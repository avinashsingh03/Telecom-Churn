# Telecom Churn Prediction — Streamlit App

## 📖 Project Description

Customer churn is one of the most expensive problems in the telecom industry —
acquiring a new customer typically costs far more than retaining an existing one,
which makes early identification of at-risk customers a direct lever on revenue.
This project builds an end-to-end machine learning solution that predicts whether
a telecom customer is likely to churn, based on their account details, subscribed
plans, and usage behavior across day, evening, night, and international calls.

The full workflow is documented in `notebook.ipynb`: exploratory data analysis,
business KPI derivation (churn rate, customer lifetime value), outlier handling,
feature engineering (`Total Charges`, `Total_Usage`, `Service_Stress`, and a
qcut-based `Revenue_Segment`), and encoding of categorical variables. Several
classifiers — Logistic Regression, Naive Bayes, KNN, Random Forest, and Decision
Tree — were trained and compared on Accuracy, Precision, Recall, F1, and ROC-AUC.
The Decision Tree was selected and tuned using `GridSearchCV` and
`RandomizedSearchCV`, then wrapped in a single `scikit-learn Pipeline`
(`StandardScaler` + `DecisionTreeClassifier`) and serialized as `best_model.pkl`
so that every new prediction goes through identical preprocessing.

This repository wraps that trained pipeline in a polished, interactive Streamlit
application, allowing anyone — not just data scientists — to input a customer's
profile and instantly see their churn risk, probability score, and suggested
retention actions, or score an entire customer base at once via batch CSV upload.

## ✂️ Summary

A Streamlit web app that predicts telecom customer churn using a tuned Decision
Tree pipeline. Users enter customer details or upload a CSV to get instant churn
predictions, probability scores, risk gauges, feature insights, and downloadable
results — turning a trained ML model into a usable retention tool.

## 🚀 Live App

**[https://telecom-churn-classification.streamlit.app/](https://telecom-churn-classification.streamlit.app/)**

## 📂 Project Structure

```
Telecom-Churn/
│
├── app.py                 # Streamlit application
├── best_model.pkl         # Trained scikit-learn Pipeline (StandardScaler + DecisionTreeClassifier)
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── customer_data.csv      # Sample/reference customer dataset
└── notebook.ipynb         # Full EDA, feature engineering, model training & tuning workflow
```

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

## 🔗 Deployed Link

**[https://telecom-churn-classification.streamlit.app/](https://telecom-churn-classification.streamlit.app/)**