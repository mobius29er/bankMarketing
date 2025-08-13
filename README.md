# Bank Marketing Targeting Dashboard

I built this to turn the UCI Bank Marketing dataset into something my marketing, sales, and account teams can actually use. Upload the data, score prospects, explore segments, and get a concrete recommendation for how and when to contact each person.

This repo includes a Streamlit app that trains a model, assigns clusters, builds A–D tiers, and provides a simple calculator so a rep can key in a client profile and get the best channel and timing.

---

## What this project does

* Scores each contact with a probability of subscribing to a term deposit
* Assigns each contact to one of three behavioral clusters for quick personas
* Buckets contacts into actionable tiers

  * A = top 10% by predicted probability
  * B = next 20%
  * C = next 30%
  * D = bottom 40%
* Builds segment tables for messaging and planning

  * Who they are, what to say, when to call, and how to contact
* Provides a calculator

  * I enter Features 1–4 and macro context
  * The app recommends Feature 5 (contact method) and Feature 6 (timing)

I do not use the `duration` field because it is unknown before a call. That keeps the system realistic.

---

## Objectives

1. **Prioritize**
   Rank the list by conversion likelihood and create tiers so teams can spend time where it matters.

2. **Segment**
   Summarize conversion lift and volume by demographic, employment, loan status, history, channel, timing, and macro context.

3. **Hypersegment**
   Create simple clusters that look like personas marketers can understand and act on.

---

## Data I expect

Upload the UCI **bank-additional-full.csv** file or anything with the same columns and a semicolon separator.

* Target: `y` with values `yes` or `no`
* Core features I use:

  * Person: `age`, `education`, `marital`
  * Employment: `job`
  * Current loan status: `default`, `housing`, `loan`
  * History: `campaign`, `pdays`, `previous`, `poutcome`
  * Communication: `contact`
  * Timing: `day_of_week`, `month`
  * Macro: `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`
* I ignore `duration` for training and scoring

Unknowns such as `unknown` are treated as valid categories.

---

## How it works

* **Model**
  Logistic Regression using one hot encoding for categoricals and standard scaling for numerics. This is fast and well calibrated for ranking. I compared it to KNN, Decision Trees, and SVM and picked Logistic Regression as the champion here.

* **Tiers**
  I convert predicted probabilities into A, B, C, D tiers using quantiles. A is the top 10%.

* **Clusters**
  KMeans on a dense, encoded matrix. I pick k with a quick silhouette sweep over 3, 4, and 5.

* **Segment tables**
  One way slices (job, month, contact, etc.). I show predicted rate, actual rate for sanity, lift vs overall, and conversions per 1,000 calls.

* **Calculator**
  I let the user provide Features 1–4 and macro context. I grid search every combination of contact method by day of week by month for that profile and return the top recommendations by predicted conversion. Macro values are simulated from dataset ranges inside the app.

---

## Quick start

1. Create a virtual environment

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
   ```

2. Install deps

   ```bash
   pip install streamlit scikit-learn pandas numpy scipy
   ```

3. Run

   ```bash
   streamlit run app.py
   ```

4. Upload `bank-additional-full.csv`
   The file should be semicolon separated.

5. Use the tabs

   * Scored List: filter by cluster, tier, contact, month, probability
   * Tiers: A to D summary with lift and conversions per 1,000 calls
   * Segments: pick a column and see ranked lift
   * Calculator: enter a client profile and get contact and timing
   * Upload New Contacts: score and cluster a second file and download results

---

## What the outputs mean

* **Predicted rate**
  The model’s probability of subscription. This drives tiering and the calculator.

* **Conversions per 1,000 calls**
  A simple way to compare options. Predicted rate multiplied by 1,000.

* **Lift**
  Segment predicted rate divided by overall predicted rate. Helpful for choosing which segments to target when time is limited.

* **Clusters**
  Not truth. They are behaviorally coherent groups to simplify messaging and channel strategy.

---

## Ground rules and constraints

* `duration` is not used for decisions
* Macro data comes from the dataset’s observed ranges, not a live feed
* The app expects the column names above. If a column is missing I fill it with NaN and one hot encoding handles it
* Random seeds are set to 42 where it matters

---

## Extending this

* Replace Logistic Regression with a tuned Gradient Boosting model if you want more lift and you are comfortable managing calibration and speed
* Plug in a real macro data API and cache it per date
* Add a champion challenger switch with time stamped model cards
* Serve a small FastAPI endpoint for batch scoring and power a React UI

---

## Common gotchas

* Wrong separator
  The UCI file uses `;`. If you upload a comma separated version, fix the `sep` argument or resave the file.

* Missing `y`
  The app expects `y` to exist even for training. For scoring a new list it is optional.

* Empty segments
  If a filter removes too many rows, segment tables can be empty. Clear some filters.

---

## Repo map

* `app.py`
  Streamlit app with training, scoring, clustering, tiers, segments, and calculator

* `README.md`
  What you are reading

---

## Acknowledgments

Dataset from the UCI Machine Learning Repository. Original study by Sérgio Moro, Paulo Cortez, and Paulo Rita.
Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306.

---

## License

Use this for learning and internal demos. If you plan to ship this, review your data policies and local regulations first.
