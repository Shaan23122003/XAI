import os
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import custom helpers
from src.utils import (
    FEATURE_COLUMNS,
    load_model,
    prepare_single_input,
    format_prediction,
    clean_dataframe,
    encode_target,
    split_features_target
)
from src.explain import (
    explain_shap_single,
    get_lime_explainer,
    explain_lime_single
)

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_PATH = "models/xgb_pipeline.pkl"
DATA_PATH = "data/Raw/XAI_HR_NUMERIC.csv"

st.set_page_config(
    page_title="HR Attrition Intelligence",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "HR Attrition Prediction · Powered by XGBoost, SHAP & LIME"
    }
)

# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    """Loads the pre-trained XGBoost pipeline."""
    if os.path.exists(MODEL_PATH):
        return load_model(MODEL_PATH)
    return None

@st.cache_data
def load_training_data():
    """
    Loads and preprocesses the training data to prepopulate
    Streamlit UI dropdowns and min/max ranges accurately.
    """
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        df = clean_dataframe(df)
        df = encode_target(df)
        X, y = split_features_target(df)
        return X
    return None

# ── Main app ──────────────────────────────────────────────────────────────────
def main():

    # ── Metric card styling ───────────────────────────────────────────────────
    st.markdown("""
    <style>
    [data-testid="metric-container"] {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1rem 1.25rem;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 1.6rem;
        font-weight: 700;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("## 🏢 HR Attrition Intelligence")
    st.caption("Powered by XGBoost · Explained with SHAP & LIME · IBM HR Analytics Dataset")
    st.divider()

    # ── Load dependencies ─────────────────────────────────────────────────────
    pipeline    = load_pipeline()
    X_train_raw = load_training_data()

    if pipeline is None or X_train_raw is None:
        st.error("Model pipeline or training data not found. Please run src/train.py first.")
        return

    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.markdown("## 🧑‍💼 Employee Profile")
    st.sidebar.caption("Fill in the employee details below, then click Predict.")
    st.sidebar.divider()

    categorical_cols = X_train_raw.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols     = X_train_raw.select_dtypes(exclude=["object", "category"]).columns.tolist()

    # Logical groupings for the sidebar expanders
    GROUPS = {
        "👤 Personal":      ["Age", "Gender", "MaritalStatus", "Education", "EducationField"],
        "💼 Job Details":   ["Department", "JobRole", "JobLevel", "JobInvolvement",
                             "BusinessTravel", "OverTime"],
        "💰 Compensation":  ["MonthlyIncome", "DailyRate", "HourlyRate", "MonthlyRate",
                             "PercentSalaryHike", "StockOptionLevel"],
        "😊 Satisfaction":  ["EnvironmentSatisfaction", "JobSatisfaction",
                             "RelationshipSatisfaction", "WorkLifeBalance"],
        "📅 Work History":  ["YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion",
                             "YearsWithCurrManager", "TotalWorkingYears", "NumCompaniesWorked",
                             "TrainingTimesLastYear", "DistanceFromHome", "PerformanceRating"],
    }

    def _render_input(col):
        """Renders the correct sidebar widget for a single feature column."""
        if col in categorical_cols:
            options = sorted(X_train_raw[col].dropna().unique().tolist())
            return st.sidebar.selectbox(col, options=options, key=col)
        elif col in numeric_cols:
            min_val  = float(X_train_raw[col].min())
            max_val  = float(X_train_raw[col].max())
            mean_val = float(X_train_raw[col].median())
            if X_train_raw[col].dtype == "int64":
                return st.sidebar.number_input(
                    col,
                    min_value=int(min_val), max_value=int(max_val),
                    value=int(mean_val), step=1, key=col
                )
            else:
                return st.sidebar.number_input(
                    col,
                    min_value=min_val, max_value=max_val,
                    value=mean_val, key=col
                )
        else:
            return st.sidebar.text_input(col, key=col)

    user_input   = {}
    grouped_cols = [col for cols in GROUPS.values() for col in cols]
    ungrouped    = [col for col in FEATURE_COLUMNS if col not in grouped_cols]

    for group_label, cols in GROUPS.items():
        with st.sidebar.expander(group_label, expanded=False):
            for col in cols:
                if col in FEATURE_COLUMNS:
                    user_input[col] = _render_input(col)

    # Safety net for any feature not assigned to a group
    if ungrouped:
        with st.sidebar.expander("🔧 Other", expanded=False):
            for col in ungrouped:
                user_input[col] = _render_input(col)

    st.sidebar.divider()
    predict_clicked = st.sidebar.button(
        "🔍 Predict Attrition", type="primary", use_container_width=True
    )

    # ── Prediction Section ────────────────────────────────────────────────────
    st.subheader("Prediction Result")

    if predict_clicked:
        with st.spinner("Running model and generating explanations…"):

            # Prepare input and run model
            df_input  = prepare_single_input(user_input)
            raw_pred  = pipeline.predict(df_input)
            raw_prob  = pipeline.predict_proba(df_input)
            result    = format_prediction(raw_pred, raw_prob)

            # ── Metric cards ──────────────────────────────────────────────────
            is_attrition = result["prediction"] == "Attrition"
            risk_pct     = result["probability"] * 100
            confidence   = (
                "High"   if result["probability"] >= 0.70 else
                "Medium" if result["probability"] >= 0.45 else
                "Low"
            )

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric(
                label="Prediction",
                value=result["prediction"],
                delta="⚠️ Risk Detected" if is_attrition else "✅ Stable",
                delta_color="inverse" if is_attrition else "normal"
            )
            mc2.metric(label="Attrition Probability", value=f"{risk_pct:.1f}%")
            mc3.metric(label="Model Confidence",      value=confidence)

            if is_attrition:
                st.error(
                    "⚠️  This employee profile shows elevated attrition risk. "
                    "Review the feature explanations below.",
                    icon="🚨"
                )
            else:
                st.success(
                    "✅  This employee profile is predicted to stay. Low attrition risk.",
                    icon="✅"
                )

            st.divider()
            st.subheader("Explainability")
            st.caption(
                "Local explanations for this specific prediction — not global model behaviour."
            )

            # ── Tabs: SHAP | LIME ─────────────────────────────────────────────
            tab1, tab2 = st.tabs(
                ["📊 SHAP — Feature Contributions", "🟢 LIME — Local Explanation"]
            )

            # SHAP TAB
            with tab1:
                st.markdown(
                    "Top 10 features driving this specific prediction. "
                    "**Red** bars push toward Attrition; **Indigo** bars push away."
                )

                shap_values, X_transformed, feature_names = explain_shap_single(
                    pipeline, df_input
                )

                # Disambiguate old vs new SHAP output format
                if isinstance(shap_values, list):
                    sv = shap_values[1][0]   # older SHAP: list of [class0, class1] arrays
                else:
                    sv = shap_values[0]      # newer SHAP: single 2-D array, row 0

                # Build top-10 DataFrame
                shap_df = pd.DataFrame({"Feature": feature_names, "SHAP Value": sv})
                shap_df["Abs_SHAP"] = shap_df["SHAP Value"].abs()
                shap_df = shap_df.sort_values(by="Abs_SHAP", ascending=False).head(10)

                # Themed Matplotlib chart
                fig, ax = plt.subplots(figsize=(8, 5))
                fig.patch.set_facecolor("none")
                ax.set_facecolor("#1E293B")
                colors = [
                    "#F43F5E" if x > 0 else "#6366F1"
                    for x in shap_df["SHAP Value"][::-1]
                ]
                ax.barh(
                    shap_df["Feature"][::-1],
                    shap_df["SHAP Value"][::-1],
                    color=colors,
                    edgecolor="none"
                )
                ax.set_xlabel("SHAP Value", color="#94A3B8")
                ax.set_title("Top 10 SHAP Feature Importances", color="#F1F5F9", pad=12)
                ax.tick_params(colors="#CBD5E1", labelsize=9)
                ax.xaxis.label.set_color("#94A3B8")
                for spine in ["top", "right"]:
                    ax.spines[spine].set_visible(False)
                for spine in ["bottom", "left"]:
                    ax.spines[spine].set_color("#334155")
                ax.axvline(0, color="#475569", linewidth=0.8, linestyle="--")
                st.pyplot(fig)
                plt.close(fig)

            # LIME TAB
            with tab2:
                st.markdown(
                    "LIME perturbs the input and fits a local linear model "
                    "to approximate the prediction around this data point."
                )

                lime_explainer, cat_maps, lime_feats, lime_cats = get_lime_explainer(
                    X_train_raw
                )
                lime_exp = explain_lime_single(
                    pipeline, lime_explainer, cat_maps, lime_feats, lime_cats, df_input
                )
                components.html(lime_exp.as_html(), height=500, scrolling=True)

    else:
        # Placeholder when no prediction has been run yet
        st.info(
            "👈  Configure the employee profile in the sidebar and click "
            "**🔍 Predict Attrition** to see the result.",
            icon="💡"
        )

if __name__ == "__main__":
    main()
