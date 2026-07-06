"""
Telecom Customer Churn Prediction — Streamlit Application
===========================================================
Loads a trained scikit-learn Pipeline (StandardScaler + DecisionTreeClassifier)
persisted as `best_model.pkl` from the accompanying training notebook, and
serves an interactive, production-style prediction UI.

Run with:
    streamlit run app.py
"""

import io
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "best_model.pkl"

US_STATES = [
    "AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "HI",
    "IA", "ID", "IL", "IN", "KS", "KY", "LA", "MA", "MD", "ME", "MI", "MN",
    "MO", "MS", "MT", "NC", "ND", "NE", "NH", "NJ", "NM", "NV", "NY", "OH",
    "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VA", "VT", "WA",
    "WI", "WV", "WY",
]

# --------------------------------------------------------------------------------
# STYLING
# --------------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .main { background-color: #0e1117; }
        #MainMenu, footer { visibility: hidden; }

        .app-header {
            padding: 1.75rem 2rem;
            border-radius: 14px;
            background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
            color: white;
            margin-bottom: 1.5rem;
        }
        .app-header h1 { margin: 0; font-size: 2rem; font-weight: 700; }
        .app-header p { margin: 0.35rem 0 0 0; opacity: 0.9; font-size: 0.95rem; }

        .metric-card {
            background: #1a1c24;
            border: 1px solid #2d2f3a;
            border-radius: 12px;
            padding: 1.1rem 1.3rem;
            text-align: center;
        }
        .metric-card h3 { margin: 0; font-size: 0.85rem; color: #9ca3af; font-weight: 500; }
        .metric-card p { margin: 0.25rem 0 0 0; font-size: 1.6rem; font-weight: 700; color: #f9fafb; }

        .risk-high {
            background: rgba(220, 38, 38, 0.12);
            border: 1px solid rgba(220, 38, 38, 0.5);
            border-radius: 12px;
            padding: 1.2rem 1.4rem;
        }
        .risk-low {
            background: rgba(16, 185, 129, 0.12);
            border: 1px solid rgba(16, 185, 129, 0.5);
            border-radius: 12px;
            padding: 1.2rem 1.4rem;
        }
        section[data-testid="stSidebar"] { border-right: 1px solid #2d2f3a; }

        div.stButton > button {
            border-radius: 8px;
            font-weight: 600;
            padding: 0.55rem 1.2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------
# MODEL LOADING
# --------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained model pipeline...")
def load_model(path: str):
    """Load the trained sklearn Pipeline. Raises informative exceptions on failure."""
    model = joblib.load(path)

    expected_columns = None
    for attr_holder in (model, getattr(model, "named_steps", {}).get("scaler", None)):
        if attr_holder is not None and hasattr(attr_holder, "feature_names_in_"):
            expected_columns = list(attr_holder.feature_names_in_)
            break

    return model, expected_columns


model_load_error = None
model, EXPECTED_COLUMNS = None, None
try:
    model, EXPECTED_COLUMNS = load_model(MODEL_PATH)
except FileNotFoundError:
    model_load_error = (
        f"Could not find **{MODEL_PATH}**. Place the trained pipeline file in the same "
        "directory as this app (or update `MODEL_PATH` at the top of `app.py`)."
    )
except Exception as e:  # noqa: BLE001
    model_load_error = f"Failed to load the model pipeline: {e}"


# --------------------------------------------------------------------------------
# FEATURE ENGINEERING — mirrors the notebook's preprocessing pipeline
# --------------------------------------------------------------------------------
def engineer_features(raw: dict) -> pd.DataFrame:
    """Reproduce the notebook's feature engineering for a single customer record."""
    day_charge = raw["Total day charge"]
    eve_charge = raw["Total eve charge"]
    night_charge = raw["Total night charge"]
    intl_charge = raw["Total intl charge"]

    total_charges = day_charge + eve_charge + night_charge + intl_charge
    total_usage = (
        raw["Total day minutes"]
        + raw["Total eve minutes"]
        + raw["Total night minutes"]
        + raw["Total intl minutes"]
    )
    service_stress = raw["Customer service calls"] / (raw["Account length"] + 1)

    # Business revenue tier — approximates the training-time pd.qcut(Total Charges, q=3)
    # tertile split. Adjust thresholds in the sidebar's Advanced Settings if you have
    # the exact training-data quantile boundaries.
    low_cut, high_cut = st.session_state.get("revenue_thresholds", (45.0, 75.0))
    if total_charges <= low_cut:
        revenue_segment = "Low"
    elif total_charges <= high_cut:
        revenue_segment = "Medium"
    else:
        revenue_segment = "High"

    record = {
        "Account length": raw["Account length"],
        "International plan": 1 if raw["International plan"] == "Yes" else 0,
        "Voice mail plan": 1 if raw["Voice mail plan"] == "Yes" else 0,
        "Number vmail messages": raw["Number vmail messages"],
        "Total day minutes": raw["Total day minutes"],
        "Total day calls": raw["Total day calls"],
        "Total day charge": day_charge,
        "Total eve minutes": raw["Total eve minutes"],
        "Total eve calls": raw["Total eve calls"],
        "Total eve charge": eve_charge,
        "Total night minutes": raw["Total night minutes"],
        "Total night calls": raw["Total night calls"],
        "Total night charge": night_charge,
        "Total intl minutes": raw["Total intl minutes"],
        "Total intl calls": raw["Total intl calls"],
        "Total intl charge": intl_charge,
        "Customer service calls": raw["Customer service calls"],
        "Total Charges": total_charges,
        "Total_Usage": total_usage,
        "Service_Stress": service_stress,
    }

    df = pd.DataFrame([record])

    # One-hot encode State and Revenue_Segment exactly like pd.get_dummies(drop_first=True)
    df["State"] = raw["State"]
    df["Revenue_Segment"] = revenue_segment
    df = pd.get_dummies(df, columns=["State", "Revenue_Segment"], dtype=int)

    return df


def align_to_model(df: pd.DataFrame, expected_columns):
    """Reindex engineered features to exactly match what the pipeline was trained on."""
    if not expected_columns:
        return df, [], []

    missing = [c for c in expected_columns if c not in df.columns]
    extra = [c for c in df.columns if c not in expected_columns]

    aligned = df.reindex(columns=expected_columns, fill_value=0)
    return aligned, missing, extra


def predict_single(raw: dict):
    engineered = engineer_features(raw)
    aligned, missing, extra = align_to_model(engineered, EXPECTED_COLUMNS)

    pred = model.predict(aligned)[0]
    proba = model.predict_proba(aligned)[0]
    churn_proba = float(proba[1]) if len(proba) > 1 else float(proba[0])

    return {
        "prediction": int(pred),
        "churn_probability": churn_proba,
        "retain_probability": 1 - churn_proba,
        "missing_cols": missing,
        "extra_cols": extra,
        "input_frame": engineered,
    }


# --------------------------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📡 Churn Predictor")
    st.caption("Decision Tree Pipeline · Telecom Customer Retention")
    st.divider()

    page = st.radio(
        "Navigate",
        ["🔮 Single Prediction", "📂 Batch Prediction", "📊 Model Insights", "ℹ️ About"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("### ⚙️ Settings")
    threshold = st.slider(
        "Churn decision threshold",
        min_value=0.05, max_value=0.95, value=0.50, step=0.05,
        help="Customers with predicted churn probability above this value are flagged as high risk.",
    )

    with st.expander("Advanced: Revenue segment thresholds"):
        low_cut = st.number_input("Low / Medium boundary ($)", value=45.0, step=1.0)
        high_cut = st.number_input("Medium / High boundary ($)", value=75.0, step=1.0)
        st.session_state["revenue_thresholds"] = (low_cut, high_cut)
        st.caption(
            "Approximates the notebook's `pd.qcut(Total Charges, q=3)` tertile split. "
            "Update these if you know the exact training-data quantile cutoffs."
        )

    st.divider()
    st.markdown("### 🩺 Model Status")
    if model_load_error:
        st.error("Model not loaded", icon="🚫")
    else:
        st.success("Pipeline loaded", icon="✅")
        st.caption(f"Expecting **{len(EXPECTED_COLUMNS)}** input features" if EXPECTED_COLUMNS else "Feature schema unknown")

    st.divider()
    st.caption("Built with Streamlit · scikit-learn")


# --------------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <h1>📡 Telecom Customer Churn Prediction</h1>
        <p>Predict churn risk in real time using a trained Decision Tree pipeline, and export results for retention campaigns.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if model_load_error:
    st.error(f"**Model unavailable.**\n\n{model_load_error}", icon="🚫")
    st.info(
        "This app expects `best_model.pkl` — a scikit-learn `Pipeline` containing a "
        "`StandardScaler` and a `DecisionTreeClassifier` — saved via `joblib.dump`, "
        "as produced in the `Telecom_churn.ipynb` notebook.",
        icon="💡",
    )
    st.stop()


# --------------------------------------------------------------------------------
# PAGE: SINGLE PREDICTION
# --------------------------------------------------------------------------------
if page == "🔮 Single Prediction":

    with st.form("customer_form"):
        st.subheader("Customer Profile")
        c1, c2, c3 = st.columns(3)
        with c1:
            state = st.selectbox("State", US_STATES, index=US_STATES.index("NY"))
        with c2:
            account_length = st.number_input("Account length (days)", min_value=0, max_value=400, value=100)
        with c3:
            service_calls = st.number_input("Customer service calls", min_value=0, max_value=20, value=1)

        st.subheader("Plan Subscriptions")
        p1, p2, p3 = st.columns(3)
        with p1:
            intl_plan = st.selectbox("International plan", ["No", "Yes"])
        with p2:
            vmail_plan = st.selectbox("Voice mail plan", ["No", "Yes"])
        with p3:
            vmail_messages = st.number_input("Number vmail messages", min_value=0, max_value=60, value=0)

        st.subheader("Usage Patterns")
        u1, u2, u3, u4 = st.columns(4)
        with u1:
            st.markdown("**Daytime**")
            day_minutes = st.number_input("Day minutes", min_value=0.0, max_value=400.0, value=180.0, key="dm")
            day_calls = st.number_input("Day calls", min_value=0, max_value=200, value=100, key="dc")
            day_charge = st.number_input("Day charge ($)", min_value=0.0, max_value=70.0, value=30.6, key="dch")
        with u2:
            st.markdown("**Evening**")
            eve_minutes = st.number_input("Evening minutes", min_value=0.0, max_value=400.0, value=200.0, key="em")
            eve_calls = st.number_input("Evening calls", min_value=0, max_value=200, value=100, key="ec")
            eve_charge = st.number_input("Evening charge ($)", min_value=0.0, max_value=35.0, value=17.0, key="ech")
        with u3:
            st.markdown("**Night**")
            night_minutes = st.number_input("Night minutes", min_value=0.0, max_value=400.0, value=200.0, key="nm")
            night_calls = st.number_input("Night calls", min_value=0, max_value=200, value=100, key="nc")
            night_charge = st.number_input("Night charge ($)", min_value=0.0, max_value=20.0, value=9.0, key="nch")
        with u4:
            st.markdown("**International**")
            intl_minutes = st.number_input("Intl minutes", min_value=0.0, max_value=25.0, value=10.0, key="im")
            intl_calls = st.number_input("Intl calls", min_value=0, max_value=25, value=4, key="ic")
            intl_charge = st.number_input("Intl charge ($)", min_value=0.0, max_value=10.0, value=2.7, key="ich")

        submitted = st.form_submit_button("🔮 Predict Churn Risk", use_container_width=True)

    if submitted:
        raw_input = {
            "State": state,
            "Account length": account_length,
            "International plan": intl_plan,
            "Voice mail plan": vmail_plan,
            "Number vmail messages": vmail_messages,
            "Total day minutes": day_minutes,
            "Total day calls": day_calls,
            "Total day charge": day_charge,
            "Total eve minutes": eve_minutes,
            "Total eve calls": eve_calls,
            "Total eve charge": eve_charge,
            "Total night minutes": night_minutes,
            "Total night calls": night_calls,
            "Total night charge": night_charge,
            "Total intl minutes": intl_minutes,
            "Total intl calls": intl_calls,
            "Total intl charge": intl_charge,
            "Customer service calls": service_calls,
        }

        try:
            result = predict_single(raw_input)
        except Exception as e:  # noqa: BLE001
            st.error(f"⚠️ Prediction failed: {e}", icon="🚫")
            st.stop()

        if result["missing_cols"]:
            st.warning(
                f"The model expected {len(result['missing_cols'])} feature(s) not produced by this "
                f"form (filled with 0): `{', '.join(result['missing_cols'][:8])}"
                f"{'...' if len(result['missing_cols']) > 8 else ''}`. "
                "Predictions may be slightly less precise for these edge cases.",
                icon="⚠️",
            )

        churn_prob = result["churn_probability"]
        is_high_risk = churn_prob >= threshold

        st.divider()
        st.subheader("Prediction Result")

        left, right = st.columns([1, 1.3])

        with left:
            box_class = "risk-high" if is_high_risk else "risk-low"
            label = "⚠️ Likely to CHURN" if is_high_risk else "✅ Likely to STAY"
            st.markdown(
                f"""
                <div class="{box_class}">
                    <h2 style="margin:0;">{label}</h2>
                    <p style="margin-top:0.5rem; color:#d1d5db;">
                        Churn probability: <strong>{churn_prob:.1%}</strong><br>
                        Decision threshold: {threshold:.0%}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")
            m1, m2 = st.columns(2)
            with m1:
                st.markdown(
                    f'<div class="metric-card"><h3>Churn Probability</h3><p>{churn_prob:.1%}</p></div>',
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    f'<div class="metric-card"><h3>Retention Probability</h3><p>{result["retain_probability"]:.1%}</p></div>',
                    unsafe_allow_html=True,
                )

        with right:
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=churn_prob * 100,
                    number={"suffix": "%"},
                    title={"text": "Churn Risk Gauge"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#db2777" if is_high_risk else "#10b981"},
                        "steps": [
                            {"range": [0, 40], "color": "rgba(16,185,129,0.25)"},
                            {"range": [40, 70], "color": "rgba(245,158,11,0.25)"},
                            {"range": [70, 100], "color": "rgba(220,38,38,0.25)"},
                        ],
                        "threshold": {
                            "line": {"color": "white", "width": 3},
                            "thickness": 0.85,
                            "value": threshold * 100,
                        },
                    },
                )
            )
            fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)", font_color="white")
            st.plotly_chart(fig, use_container_width=True)

        if is_high_risk:
            st.info(
                "**Suggested retention actions:** proactive outreach call, loyalty discount, "
                "review recent customer service interactions, offer plan optimization.",
                icon="💡",
            )

        # Download prediction
        st.divider()
        output_row = raw_input.copy()
        output_row.update(
            {
                "predicted_churn": result["prediction"],
                "churn_probability": round(churn_prob, 4),
                "retain_probability": round(result["retain_probability"], 4),
                "decision_threshold": threshold,
                "prediction_timestamp": datetime.now().isoformat(timespec="seconds"),
            }
        )
        result_df = pd.DataFrame([output_row])
        csv_buffer = io.StringIO()
        result_df.to_csv(csv_buffer, index=False)

        st.download_button(
            "⬇️ Download Prediction (CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"churn_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )


# --------------------------------------------------------------------------------
# PAGE: BATCH PREDICTION
# --------------------------------------------------------------------------------
elif page == "📂 Batch Prediction":
    st.subheader("Batch Prediction from CSV")
    st.caption(
        "Upload a CSV with raw customer columns (State, Account length, International plan, "
        "Voice mail plan, Number vmail messages, Total day/eve/night/intl minutes/calls/charge, "
        "Customer service calls). Engineered features are computed automatically."
    )

    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])

    required_raw_cols = [
        "State", "Account length", "International plan", "Voice mail plan",
        "Number vmail messages", "Total day minutes", "Total day calls", "Total day charge",
        "Total eve minutes", "Total eve calls", "Total eve charge",
        "Total night minutes", "Total night calls", "Total night charge",
        "Total intl minutes", "Total intl calls", "Total intl charge",
        "Customer service calls",
    ]

    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
        except Exception as e:  # noqa: BLE001
            st.error(f"⚠️ Could not read the file: {e}", icon="🚫")
            st.stop()

        missing_input_cols = [c for c in required_raw_cols if c not in batch_df.columns]
        if missing_input_cols:
            st.error(
                f"⚠️ The uploaded file is missing required column(s): `{', '.join(missing_input_cols)}`",
                icon="🚫",
            )
            st.stop()

        try:
            rows = []
            for _, row in batch_df.iterrows():
                res = predict_single(row.to_dict())
                rows.append((res["prediction"], res["churn_probability"]))

            batch_df["predicted_churn"] = [r[0] for r in rows]
            batch_df["churn_probability"] = [round(r[1], 4) for r in rows]
            batch_df["risk_level"] = np.where(
                batch_df["churn_probability"] >= threshold, "High Risk", "Low Risk"
            )
        except Exception as e:  # noqa: BLE001
            st.error(f"⚠️ Batch prediction failed: {e}", icon="🚫")
            st.stop()

        st.success(f"✅ Scored {len(batch_df)} customers successfully.", icon="✅")

        k1, k2, k3 = st.columns(3)
        k1.metric("Total Customers", len(batch_df))
        k2.metric("Predicted Churners", int(batch_df["predicted_churn"].sum()))
        k3.metric("Churn Rate", f"{batch_df['predicted_churn'].mean():.1%}")

        st.dataframe(batch_df, use_container_width=True)

        csv_buffer = io.StringIO()
        batch_df.to_csv(csv_buffer, index=False)
        st.download_button(
            "⬇️ Download Scored Results (CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"batch_churn_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.info("Upload a CSV file to run predictions on multiple customers at once.", icon="📁")


# --------------------------------------------------------------------------------
# PAGE: MODEL INSIGHTS
# --------------------------------------------------------------------------------
elif page == "📊 Model Insights":
    st.subheader("Model Insights")

    classifier = None
    if hasattr(model, "named_steps") and "classifier" in getattr(model, "named_steps", {}):
        classifier = model.named_steps["classifier"]
    elif hasattr(model, "steps"):
        classifier = model.steps[-1][1]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f'<div class="metric-card"><h3>Model Type</h3><p style="font-size:1.1rem;">{type(classifier).__name__ if classifier else "Unknown"}</p></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="metric-card"><h3>Input Features</h3><p>{len(EXPECTED_COLUMNS) if EXPECTED_COLUMNS else "—"}</p></div>',
            unsafe_allow_html=True,
        )

    st.write("")

    if classifier is not None and hasattr(classifier, "feature_importances_") and EXPECTED_COLUMNS:
        importance_df = (
            pd.DataFrame({"Feature": EXPECTED_COLUMNS, "Importance": classifier.feature_importances_})
            .sort_values("Importance", ascending=False)
            .head(15)
        )
        fig = go.Figure(
            go.Bar(
                x=importance_df["Importance"][::-1],
                y=importance_df["Feature"][::-1],
                orientation="h",
                marker_color="#7c3aed",
            )
        )
        fig.update_layout(
            title="Top 15 Feature Importances",
            height=500,
            margin=dict(l=10, r=10, t=50, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importances are not available for this model type.", icon="ℹ️")

    if hasattr(classifier, "get_params"):
        with st.expander("View model hyperparameters"):
            st.json(classifier.get_params())


# --------------------------------------------------------------------------------
# PAGE: ABOUT
# --------------------------------------------------------------------------------
elif page == "ℹ️ About":
    st.subheader("About this Application")
    st.markdown(
        """
This application operationalizes the Decision Tree churn model developed in
**`Telecom_churn.ipynb`** for the telecom customer retention use case.

**Pipeline summary**
- **Preprocessing:** `StandardScaler`
- **Model:** `DecisionTreeClassifier` (hyperparameters selected via `GridSearchCV`)
- **Persistence:** the full `sklearn.pipeline.Pipeline` was serialized with `joblib` to `best_model.pkl`

**Engineered features reproduced in this app**
| Feature | Definition |
|---|---|
| `Total Charges` | Sum of day, evening, night, and international charges |
| `Total_Usage` | Sum of day, evening, night, and international minutes |
| `Service_Stress` | Customer service calls ÷ (account length + 1) |
| `Revenue_Segment` | Low / Medium / High tier derived from `Total Charges` |
| `State`, `Revenue_Segment` | One-hot encoded (`drop_first=True`), matching training |

**How predictions stay consistent with training**
This app reads `feature_names_in_` directly from the loaded pipeline and reindexes
every engineered record to that exact column schema before calling `.predict()` —
so it adapts automatically even if the exact set of retained columns changes
(e.g. due to the correlation-pruning step in the notebook).

**Disclaimer:** Predictions are only as reliable as the training data and should
support, not replace, human judgment in retention decision-making.
        """
    )
