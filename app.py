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
DATA_PATH  = "data/Raw/XAI_HR_NUMERIC.csv"

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

@st.cache_resource
def build_lime_explainer(_X_train):
    """Builds and caches the LIME explainer once per session."""
    return get_lime_explainer(_X_train)

@st.cache_resource
def build_shap_explainer(_pipeline):
    """Caches the SHAP TreeExplainer once per session."""
    from src.explain import get_shap_explainer
    return get_shap_explainer(_pipeline)

# ── Main app ──────────────────────────────────────────────────────────────────
def main():

    # ── Minimal CSS ───────────────────────────────────────────────────────────
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
    .predict-hint {
        color: #64748B;
        font-size: 0.9rem;
        margin-top: 0.25rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Load dependencies ─────────────────────────────────────────────────────
    pipeline    = load_pipeline()
    X_train_raw = load_training_data()

    if pipeline is None or X_train_raw is None:
        st.error("Model pipeline or training data not found. Please run src/train.py first.")
        return

    # ── Sidebar — employee inputs ─────────────────────────────────────────────
    st.sidebar.markdown("## 🧑‍💼 Employee Profile")
    st.sidebar.caption("Fill in the employee details below, then click Predict.")
    st.sidebar.divider()

    categorical_cols = X_train_raw.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols     = X_train_raw.select_dtypes(exclude=["object", "category"]).columns.tolist()

    GROUPS = {
        "👤 Personal":     ["Age", "Gender", "MaritalStatus", "Education", "EducationField"],
        "💼 Job Details":  ["Department", "JobRole", "JobLevel", "JobInvolvement",
                            "BusinessTravel", "OverTime"],
        "💰 Compensation": ["MonthlyIncome", "DailyRate", "HourlyRate", "MonthlyRate",
                            "PercentSalaryHike", "StockOptionLevel"],
        "😊 Satisfaction": ["EnvironmentSatisfaction", "JobSatisfaction",
                            "RelationshipSatisfaction", "WorkLifeBalance"],
        "📅 Work History": ["YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion",
                            "YearsWithCurrManager", "TotalWorkingYears", "NumCompaniesWorked",
                            "TrainingTimesLastYear", "DistanceFromHome", "PerformanceRating"],
    }

    def _render_input(col):
        if col in categorical_cols:
            options = sorted(X_train_raw[col].dropna().unique().tolist())
            return st.sidebar.selectbox(col, options=options, key=col)
        elif col in numeric_cols:
            min_val  = float(X_train_raw[col].min())
            max_val  = float(X_train_raw[col].max())
            mean_val = float(X_train_raw[col].median())
            if X_train_raw[col].dtype == "int64":
                return st.sidebar.number_input(
                    col, min_value=int(min_val), max_value=int(max_val),
                    value=int(mean_val), step=1, key=col
                )
            else:
                return st.sidebar.number_input(
                    col, min_value=min_val, max_value=max_val,
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

    if ungrouped:
        with st.sidebar.expander("🔧 Other", expanded=False):
            for col in ungrouped:
                user_input[col] = _render_input(col)

    # ── Main area header ──────────────────────────────────────────────────────
    st.markdown("## 🏢 HR Attrition Intelligence")
    st.caption("Powered by XGBoost · Explained with SHAP & LIME · IBM HR Analytics Dataset")
    st.divider()

    # ── Predict button ────────────────────────────────────────────────────────
    btn_col, _ = st.columns([1, 3])
    predict_clicked = btn_col.button(
        "🔍 Run Prediction", type="primary", use_container_width=True
    )
    st.markdown(
        "<p class='predict-hint'>Configure the employee profile in the sidebar, "
        "then click Run Prediction.</p>",
        unsafe_allow_html=True
    )
    st.divider()

    # ── Results ───────────────────────────────────────────────────────────────
    if not predict_clicked:
        st.info("Results will appear here after you run a prediction.", icon="📊")
        return

    # ── Run model ─────────────────────────────────────────────────────────────
    with st.spinner("Running model…"):
        df_input = prepare_single_input(user_input)
        raw_pred = pipeline.predict(df_input)
        raw_prob = pipeline.predict_proba(df_input)
        result   = format_prediction(raw_pred, raw_prob)

    is_attrition = result["prediction"] == "Attrition"
    risk_pct     = result["probability"] * 100
    confidence   = (
        "High"   if result["probability"] >= 0.70 else
        "Medium" if result["probability"] >= 0.45 else
        "Low"
    )

    # ── Verdict + SHAP side by side ───────────────────────────────────────────
    left, right = st.columns([1, 1.6], gap="large")

    with left:
        st.subheader("Verdict")
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric(
            label="Outcome",
            value=result["prediction"],
            delta="⚠️ At Risk" if is_attrition else "✅ Stable",
            delta_color="inverse" if is_attrition else "normal"
        )
        mc2.metric(label="Probability", value=f"{risk_pct:.1f}%")
        mc3.metric(label="Confidence",  value=confidence)

        st.write("")

        if is_attrition:
            st.error(
                "This employee profile shows elevated attrition risk. "
                "See the SHAP chart for the key drivers.",
                icon="🚨"
            )
        else:
            st.success(
                "This employee is predicted to stay. Low attrition risk.",
                icon="✅"
            )

    with right:
        st.subheader("Top Feature Drivers (SHAP)")
        st.caption("Red = pushes toward Attrition · Indigo = pushes away")

        with st.spinner("Computing SHAP values…"):
            preprocessor   = pipeline.named_steps["preprocessor"]
            X_transformed  = preprocessor.transform(df_input)
            feature_names  = preprocessor.get_feature_names_out()
            feature_names  = [f.replace("cat__", "").replace("num__", "") for f in feature_names]
            shap_explainer = build_shap_explainer(pipeline)
            shap_values    = shap_explainer.shap_values(X_transformed)

        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        else:
            sv = shap_values[0]

        shap_df = pd.DataFrame({"Feature": feature_names, "SHAP Value": sv})
        shap_df["Abs_SHAP"] = shap_df["SHAP Value"].abs()
        shap_df = shap_df.sort_values(by="Abs_SHAP", ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor("none")
        ax.set_facecolor("#1E293B")
        colors = [
            "#F43F5E" if x > 0 else "#6366F1"
            for x in shap_df["SHAP Value"][::-1]
        ]
        ax.barh(shap_df["Feature"][::-1], shap_df["SHAP Value"][::-1],
                color=colors, edgecolor="none")
        ax.set_xlabel("SHAP Value", color="#94A3B8")
        ax.tick_params(colors="#CBD5E1", labelsize=9)
        ax.xaxis.label.set_color("#94A3B8")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_color("#334155")
        ax.axvline(0, color="#475569", linewidth=0.8, linestyle="--")
        st.pyplot(fig)
        plt.close(fig)

    # ── LIME — full width below ───────────────────────────────────────────────
    st.divider()
    st.subheader("🟢 LIME — Local Linear Explanation")
    st.caption(
        "LIME perturbs the input and fits a local linear model to approximate "
        "the prediction around this data point."
    )

    with st.spinner("Generating LIME explanation…"):
        lime_explainer, cat_maps, lime_feats, lime_cats = build_lime_explainer(X_train_raw)
        lime_exp = explain_lime_single(
            pipeline, lime_explainer, cat_maps, lime_feats, lime_cats, df_input
        )

    # ── LIME class probabilities ──────────────────────────────────────────
    lime_probs = lime_exp.predict_proba
    lp1, lp2, _ = st.columns([1, 1, 2])
    lp1.metric(label="No Attrition (LIME)", value=f"{lime_probs[0] * 100:.1f}%")
    lp2.metric(label="Attrition (LIME)",    value=f"{lime_probs[1] * 100:.1f}%")

    st.write("")

    # ── LIME feature contribution chart ───────────────────────────────────
    lime_list = lime_exp.as_list()
    lime_features = [item[0] for item in lime_list]
    lime_weights  = [item[1] for item in lime_list]

    lime_chart_df = pd.DataFrame({
        "Feature": lime_features,
        "Weight": lime_weights
    })
    lime_chart_df["Abs_W"] = lime_chart_df["Weight"].abs()
    lime_chart_df = lime_chart_df.sort_values(by="Abs_W", ascending=True)

    fig2, ax2 = plt.subplots(figsize=(8, max(4, len(lime_chart_df) * 0.45)))
    fig2.patch.set_facecolor("none")
    ax2.set_facecolor("#1E293B")

    lime_colors = [
        "#F43F5E" if w > 0 else "#22D3EE"
        for w in lime_chart_df["Weight"]
    ]
    bars = ax2.barh(
        lime_chart_df["Feature"],
        lime_chart_df["Weight"],
        color=lime_colors,
        edgecolor="none",
        height=0.65
    )

    # Value labels on bars
    for bar, val in zip(bars, lime_chart_df["Weight"]):
        x_pos = bar.get_width()
        ha = "left" if x_pos >= 0 else "right"
        offset = 0.002 if x_pos >= 0 else -0.002
        ax2.text(
            x_pos + offset, bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}", va="center", ha=ha,
            color="#F1F5F9", fontsize=8, fontweight="600"
        )

    ax2.set_xlabel("Feature Weight", color="#94A3B8")
    ax2.set_title(
        "LIME Feature Contributions",
        color="#F1F5F9", pad=14, fontsize=13, fontweight="bold"
    )
    ax2.tick_params(colors="#CBD5E1", labelsize=9)
    ax2.xaxis.label.set_color("#94A3B8")
    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax2.spines[spine].set_color("#334155")
    ax2.axvline(0, color="#475569", linewidth=0.8, linestyle="--")
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    # Legend
    st.markdown(
        "<div style='text-align:center; margin-top:0.5rem;'>"
        "<span style='color:#F43F5E; font-weight:600;'>■</span> "
        "<span style='color:#94A3B8; font-size:0.85rem;'>Pushes toward Attrition</span>"
        "&nbsp;&nbsp;&nbsp;&nbsp;"
        "<span style='color:#22D3EE; font-weight:600;'>■</span> "
        "<span style='color:#94A3B8; font-size:0.85rem;'>Pushes toward Retention</span>"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
