# Telecom Churn Prediction — Streamlit App

A production-style Streamlit application serving the Decision Tree churn pipeline
trained in `Telecom_churn.ipynb`.

## Setup

1. Place your trained pipeline file, **`best_model.pkl`**, in this same folder
   (this is the file saved via `joblib.dump(pipeline, "best_model.pkl")` in the notebook).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Features

- **Single Prediction** — a full customer intake form (account info, plans, usage
  across day/evening/night/international) with a live churn-risk gauge, probability
  breakdown, retention recommendations, and CSV download of the result.
- **Batch Prediction** — upload a CSV of many customers and download scored results.
- **Model Insights** — feature importances and hyperparameters pulled directly from
  the loaded model.
- **Sidebar controls** — adjustable churn decision threshold and revenue-segment
  boundary settings.
- **Robust error handling** — clear messages if the model file is missing, if a
  batch CSV is missing required columns, or if a prediction fails.

## How feature alignment works

The app reproduces the notebook's feature engineering (`Total Charges`, `Total_Usage`,
`Service_Stress`, `Revenue_Segment`, and one-hot encoding for `State` /
`Revenue_Segment`), then reindexes the resulting record against
`model.feature_names_in_` — the exact column list the pipeline was fit on. This
means the app keeps working correctly even if the specific set of retained columns
changes (e.g. due to the notebook's correlation-based feature pruning step),
without you needing to hardcode the final schema.

> **Note on `Revenue_Segment`:** the notebook computes this via
> `pd.qcut(Total_Charges, q=3)` over the *entire training set*, so the exact
> tertile boundaries depend on that data. This app approximates them with
> configurable thresholds (Sidebar → Advanced) — update them if you have the
> real training-time quantile cutoffs for perfect parity.
