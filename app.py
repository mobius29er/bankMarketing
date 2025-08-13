# app.py — Bank Marketing Targeting Dashboard (Streamlit)
# ------------------------------------------------------
# What this app does
# - Upload a bank marketing dataset (UCI bank-additional-full.csv; separator=';')
# - Train a champion classifier (Logistic Regression) WITHOUT 'duration'
# - Assign customers to clusters (KMeans) for hyper-segmentation (Objective 3)
# - Explore tiers (A/B/C/D) and filter the scored list (Objective 1)
# - Segment insights tables (Objective 2)
# - "Calculator": enter Features 1–4 + macro (7) and get recommended
#   contact method (5) and timing (6) by counterfactual scoring
#
# How to run
#   streamlit run app.py
# ------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import time
import inspect

from dataclasses import dataclass
from typing import List, Tuple, Dict

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ============ Utilities ============

CAT_CONTACT = ["cellular", "telephone"]
CAT_DOW = ["mon","tue","wed","thu","fri"]
CAT_MONTH = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]

@st.cache_data(show_spinner=False)
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file, sep=';')
    # Standardize column names just in case
    df.columns = [c.strip() for c in df.columns]
    return df

@dataclass
class FeatureSets:
    X: pd.DataFrame
    preproc: ColumnTransformer
    dense_preproc: ColumnTransformer
    cat_cols: List[str]
    num_cols: List[str]


def build_feature_sets(df: pd.DataFrame) -> FeatureSets:
    # Target
    if 'y' not in df.columns:
        raise ValueError("Expected target column 'y' in uploaded data.")
    # Drop duration (benchmark-only; not available at decision time)
    cols_drop = ['duration'] if 'duration' in df.columns else []
    X = df.drop(columns=cols_drop + ['y'])

    # Identify types
    cat_cols = X.select_dtypes(include='object').columns.tolist()
    num_cols = X.select_dtypes(exclude='object').columns.tolist()

    preproc = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', StandardScaler(), num_cols)
    ])

    # Dense preprocessor for KMeans (OneHotEncoder must output dense)
    ohe_kwargs = {'handle_unknown': 'ignore'}
    if 'sparse_output' in inspect.signature(OneHotEncoder).parameters:
        ohe_kwargs['sparse_output'] = False
    else:
        ohe_kwargs['sparse'] = False

    dense_preproc = ColumnTransformer([
        ('cat', OneHotEncoder(**ohe_kwargs), cat_cols),
        ('num', StandardScaler(), num_cols)
    ])

    return FeatureSets(X=X, preproc=preproc, dense_preproc=dense_preproc,
                       cat_cols=cat_cols, num_cols=num_cols)


@st.cache_resource(show_spinner=False)
def train_logreg(X: pd.DataFrame, y: pd.Series, preproc: ColumnTransformer) -> Pipeline:
    pipe = Pipeline([('prep', preproc), ('clf', LogisticRegression(max_iter=1000))])
    pipe.fit(X, y)
    return pipe


@st.cache_resource(show_spinner=False)
def train_kmeans(X: pd.DataFrame, dense_preproc: ColumnTransformer,
                 cand_k: List[int] = [3,4,5]) -> Tuple[KMeans, ColumnTransformer]:
    Xt = dense_preproc.fit_transform(X)
    # convert to dense if sparse leaked through
    try:
        import scipy.sparse as sp
        if sp.issparse(Xt):
            Xt = Xt.toarray()
    except Exception:
        Xt = np.asarray(Xt)

    rng = np.random.RandomState(42)
    sample_idx = rng.choice(Xt.shape[0], size=min(10000, Xt.shape[0]), replace=False)

    best_k, best_score, best_model = None, -1, None
    for k in cand_k:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(Xt)
        sil = silhouette_score(Xt[sample_idx], km.labels_[sample_idx])
        if sil > best_score:
            best_k, best_score, best_model = k, sil, km
    return best_model, dense_preproc


def assign_tiers(proba: pd.Series) -> pd.Series:
    s = pd.Series(proba)
    try:
        return pd.qcut(s, q=[0.0,0.4,0.7,0.9,1.0], labels=['D','C','B','A'])
    except ValueError:
        # fallback if many ties
        ranks = s.rank(method='first', pct=True)
        return pd.cut(ranks, bins=[0,0.4,0.7,0.9,1.0], labels=['D','C','B','A'], include_lowest=True)


def recommend_contact_and_timing(model: Pipeline,
                                 base_row: pd.Series,
                                 contacts=CAT_CONTACT,
                                 dows=CAT_DOW,
                                 months=CAT_MONTH,
                                 topn: int=5) -> pd.DataFrame:
    # Enumerate all candidate (contact, day_of_week, month) combos and score
    rows = []
    for c in contacts:
        for d in dows:
            for m in months:
                r = base_row.copy()
                for col, val in [('contact', c), ('day_of_week', d), ('month', m)]:
                    if col in r.index:
                        r[col] = val
                proba = float(model.predict_proba(pd.DataFrame([r]))[:,1])
                rows.append({'contact': c, 'day_of_week': d, 'month': m, 'predicted_rate': proba})
    tab = pd.DataFrame(rows).sort_values(['predicted_rate'], ascending=False).head(topn)
    tab['conversions_per_1000_calls'] = (tab['predicted_rate']*1000).round(0).astype(int)
    return tab


# ============ Streamlit App ============

st.set_page_config(page_title="Bank Marketing Targeting Dashboard", layout="wide")
st.title("📈 Bank Marketing Targeting Dashboard")
st.caption("Upload your dataset, score prospects, explore segments, and get contact/timing recommendations.")

with st.sidebar:
    st.header("1) Upload dataset")
    file = st.file_uploader("CSV (e.g., bank-additional-full.csv)", type=['csv'])
    st.markdown("*Note:* The file should use `;` as the separator (UCI format).")

    st.header("2) Train models")
    st.markdown("Classifier: Logistic Regression (no 'duration'). Clustering: KMeans (silhouette).")
    train_button = st.button("Train / Retrain", type="primary")

# Load data
if file is None:
    st.info("Upload the bank marketing CSV to begin.")
    st.stop()

raw = load_data(file)

# Validate target
if 'y' not in raw.columns:
    st.error("Column 'y' not found. Please upload the UCI bank-additional(-full).csv with target 'y'.")
    st.stop()

# Build feature sets (pre-call only)
fs = build_feature_sets(raw)
# Target mapping
y = (raw['y'] == 'yes').astype(int)

# Train or reuse models
if train_button or ('logreg' not in st.session_state):
    with st.spinner("Training models…"):
        t0 = time.perf_counter()
        logreg = train_logreg(fs.X, y, fs.preproc)
        kmeans, dense_prep = train_kmeans(fs.X, fs.dense_preproc)
        t1 = time.perf_counter()
    st.session_state['logreg'] = logreg
    st.session_state['kmeans'] = kmeans
    st.session_state['dense_prep'] = dense_prep
    st.session_state['train_secs'] = round(t1 - t0, 2)

logreg = st.session_state['logreg']
kmeans = st.session_state['kmeans']
dense_prep = st.session_state['dense_prep']
train_secs = st.session_state.get('train_secs', None)

# Score all rows + assign cluster
proba = logreg.predict_proba(fs.X)[:,1]
Xt_all = dense_prep.transform(fs.X)
try:
    import scipy.sparse as sp
    if sp.issparse(Xt_all): Xt_all = Xt_all.toarray()
except Exception:
    Xt_all = np.asarray(Xt_all)
clusters = kmeans.predict(Xt_all)

tiers = assign_tiers(proba)
scored = fs.X.copy()
scored['predicted_rate'] = proba
scored['cluster'] = clusters
scored['tier'] = tiers
scored['y_true'] = y

# Overview
st.subheader("Overview")
colA, colB, colC, colD = st.columns(4)
colA.metric("Rows", f"{len(scored):,}")
colB.metric("Overall conversion (actual)", f"{y.mean():.3f}")
colC.metric("Mean predicted rate", f"{proba.mean():.3f}")
if train_secs:
    colD.metric("Train time (s)", f"{train_secs}")

# Tabs
tabs = st.tabs(["Scored List", "Tiers", "Segments", "Calculator", "Upload New Contacts"])

# ============ Scored List (with filters) ============
with tabs[0]:
    st.markdown("### Scored & Clustered List")
    # Simple filters: cluster, tier, contact, month
    fcols = st.columns(5)
    f_cluster = fcols[0].multiselect("Cluster", sorted(scored['cluster'].unique().tolist()))
    f_tier = fcols[1].multiselect("Tier", ['A','B','C','D'])
    f_contact = fcols[2].multiselect("Contact", CAT_CONTACT)
    f_month = fcols[3].multiselect("Month", CAT_MONTH)
    min_prob = fcols[4].slider("Min predicted rate", 0.0, 1.0, 0.0, 0.01)

    view = scored.copy()
    if f_cluster: view = view[view['cluster'].isin(f_cluster)]
    if f_tier:    view = view[view['tier'].isin(f_tier)]
    if f_contact and 'contact' in view.columns:
        view = view[view['contact'].isin(f_contact)]
    if f_month and 'month' in view.columns:
        view = view[view['month'].isin(f_month)]
    view = view[view['predicted_rate'] >= min_prob]

    st.dataframe(view.sort_values('predicted_rate', ascending=False).head(200), use_container_width=True)

# ============ Tiers summary ============
with tabs[1]:
    st.markdown("### Tier Summary (A=top10%, B=next20%, C=next30%, D=bottom40%)")
    tier_tab = (scored.groupby('tier')['y_true']
                .agg(n='count', positives='sum', precision='mean')
                .reindex(['A','B','C','D']))
    base = y.mean()
    tier_tab['lift_vs_baseline'] = tier_tab['precision'] / base
    tier_tab['conversions_per_1000_calls'] = tier_tab['precision'] * 1000
    st.dataframe(tier_tab.round({'precision':4,'lift_vs_baseline':2,'conversions_per_1000_calls':0}), use_container_width=True)

# ============ Segment insights (one-way) ============
with tabs[2]:
    st.markdown("### Segment Insights (Predicted rate by slice)")
    seg_col = st.selectbox(
        "Choose a column to rank",
        options=[c for c in [
            'job','education','marital','default','housing','loan','contact','day_of_week','month',
            'poutcome','campaign','previous','pdays','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']
            if c in scored.columns]
    )

    # Bin numerics for readability
    def qbin(s: pd.Series, q=4):
        try:
            return pd.qcut(s, q=q, duplicates='drop').astype(str)
        except Exception:
            return s.astype(str)

    tmp = scored.copy()
    if seg_col in fs.num_cols:
        tmp[seg_col] = qbin(tmp[seg_col])

    g = tmp.groupby(seg_col)
    tab = pd.DataFrame({
        'count': g.size(),
        'predicted_rate': g['predicted_rate'].mean(),
        'actual_rate': g['y_true'].mean()
    })
    tab['lift_vs_overall'] = tab['predicted_rate'] / proba.mean()
    tab['conversions_per_1000_calls'] = tab['predicted_rate']*1000
    st.dataframe(tab.sort_values(['predicted_rate','count'], ascending=False)
                 .round({'predicted_rate':4,'actual_rate':4,'lift_vs_overall':2,'conversions_per_1000_calls':0}),
                 use_container_width=True)

# ============ Calculator ============
with tabs[3]:
    st.markdown("### Contact & Timing Calculator")
    st.caption("Enter Features 1–4 and macro (7). We'll recommend contact method (5) and timing (6).")

    # Build an empty base row with default values taken from most frequent / median
    base = {}
    # Person
    base['age'] = st.slider("Age", min_value=int(scored['age'].min()) if 'age' in scored else 18,
                            max_value=int(scored['age'].max()) if 'age' in scored else 98,
                            value=int(scored['age'].median()) if 'age' in scored else 40)
    if 'education' in scored:
        base['education'] = st.selectbox("Education", sorted(scored['education'].dropna().unique().tolist()))
    if 'marital' in scored:
        base['marital'] = st.selectbox("Marital", sorted(scored['marital'].dropna().unique().tolist()))
    # Employment
    if 'job' in scored:
        base['job'] = st.selectbox("Job", sorted(scored['job'].dropna().unique().tolist()))
    # Current loans
    if 'default' in scored:
        base['default'] = st.selectbox("Credit in default?", sorted(scored['default'].dropna().unique().tolist()))
    if 'housing' in scored:
        base['housing'] = st.selectbox("Housing loan?", sorted(scored['housing'].dropna().unique().tolist()))
    if 'loan' in scored:
        base['loan'] = st.selectbox("Personal loan?", sorted(scored['loan'].dropna().unique().tolist()))
    # History
    if 'campaign' in scored:
        base['campaign'] = st.slider("# contacts in this campaign", 1, int(scored['campaign'].max()), 1)
    if 'pdays' in scored:
        base['pdays'] = st.slider("Days since last contact (999 = none)",
                                  int(scored['pdays'].min()), int(scored['pdays'].max()), 999)
    if 'previous' in scored:
        base['previous'] = st.slider("# contacts before this campaign", 0, int(scored['previous'].max()), 0)
    if 'poutcome' in scored:
        base['poutcome'] = st.selectbox("Outcome of previous campaign", sorted(scored['poutcome'].dropna().unique().tolist()))
    # Macro (simulate using dataset ranges)
    for econ in ['emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']:
        if econ in scored:
            mn, mx = float(scored[econ].min()), float(scored[econ].max())
            val = float(scored[econ].median())
            base[econ] = st.slider(econ, mn, mx, val)

    # Contact & timing placeholders (will be set during recommendation)
    base['contact'] = CAT_CONTACT[0]
    base['day_of_week'] = CAT_DOW[0]
    base['month'] = CAT_MONTH[0]

    if st.button("Recommend contact & timing", type="primary"):
        base_row = pd.Series(base)
        recs = recommend_contact_and_timing(logreg, base_row)
        st.success("Top recommendations")
        st.dataframe(recs, use_container_width=True)

# ============ Upload new contacts for scoring ============
with tabs[4]:
    st.markdown("### Upload New Contacts to Score & Cluster")
    st.caption("Upload a CSV with the same columns (except 'y' is optional). We'll score, tier, and assign clusters.")
    new_file = st.file_uploader("New contacts CSV", type=['csv'], key="newcsv")

    if new_file is not None:
        new_df = pd.read_csv(new_file, sep=';')
        # Align to training columns (missing cols will be introduced with NaN and handled by OHE)
        for col in fs.X.columns:
            if col not in new_df.columns:
                new_df[col] = np.nan
        Xn = new_df[fs.X.columns]

        # Score
        pnew = logreg.predict_proba(Xn)[:,1]
        # Cluster
        Xn_t = dense_prep.transform(Xn)
        try:
            import scipy.sparse as sp
            if sp.issparse(Xn_t): Xn_t = Xn_t.toarray()
        except Exception:
            Xn_t = np.asarray(Xn_t)
        cnew = kmeans.predict(Xn_t)
        tnew = assign_tiers(pnew)

        out = Xn.copy()
        out['predicted_rate'] = pnew
        out['cluster'] = cnew
        out['tier'] = tnew

        st.dataframe(out.sort_values('predicted_rate', ascending=False).head(200), use_container_width=True)
        st.download_button("Download scored CSV", data=out.to_csv(index=False).encode('utf-8'),
                           file_name="scored_contacts.csv", mime="text/csv")

st.caption("\n\nBuilt with scikit-learn + Streamlit. Models exclude 'duration' for realistic use.")
