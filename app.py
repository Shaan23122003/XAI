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

# Configuration
MODEL_PATH = "models/xgb_pipeline.pkl"
DATA_PATH = "data/Raw/XAI_HR_NUMERIC.csv"

st.set_page_config(page_title="HR Attrition Prediction", layout="wide")

@st.cache_resource
def load_pipeline():
    """Loads the pre-trained XGBoost pipeline"""
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

def main():
    st.title("🏃‍♂️ HR Attrition Prediction & Explainability")
    
    # 1. Load dependencies
    pipeline = load_pipeline()
    X_train_raw = load_training_data()
    
    if not pipeline or X_train_raw is None:
        st.error("Model pipeline or training data not found. Please run src/train.py first.")
        return

    # 2. Setup the UI Sidebar dynamically based on feature types
    st.sidebar.header("Employee Attributes")
    user_input = {}
    
    categorical_cols = X_train_raw.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X_train_raw.select_dtypes(exclude=["object", "category"]).columns.tolist()
    
    for col in FEATURE_COLUMNS:
        if col in categorical_cols:
            options = sorted(X_train_raw[col].dropna().unique().tolist())
            user_input[col] = st.sidebar.selectbox(f"{col}", options=options)
        
        elif col in numeric_cols:
            min_val = float(X_train_raw[col].min())
            max_val = float(X_train_raw[col].max())
            mean_val = float(X_train_raw[col].median())
            
            if X_train_raw[col].dtype == 'int64':
                user_input[col] = st.sidebar.number_input(f"{col}", min_value=int(min_val), max_value=int(max_val), value=int(mean_val), step=1)
            else:
                user_input[col] = st.sidebar.number_input(f"{col}", min_value=min_val, max_value=max_val, value=mean_val)
        else:
            # Fallback for unclassified types
            user_input[col] = st.sidebar.text_input(f"{col}")

    # 3. Prediction Section
    st.subheader("1. Prediction")
    
    if st.button("Predict Attrition", type="primary"):
        with st.spinner("Processing Prediction and Explanations..."):
            
            # Use src.utils to format input and predict
            df_input = prepare_single_input(user_input)
            
            raw_pred = pipeline.predict(df_input)
            raw_prob = pipeline.predict_proba(df_input)
            result = format_prediction(raw_pred, raw_prob)
            
            # Show output prominently
            if result['prediction'] == 'Attrition':
                st.error(f"Prediction: **{result['prediction']}** (Probability: {result['probability'] * 100:.2f}%)")
            else:
                st.success(f"Prediction: **{result['prediction']}** (Probability: {result['probability'] * 100:.2f}%)")
            
            st.markdown("---")
            st.subheader("2. Explainability (SHAP & LIME)")
            
            col1, col2 = st.columns(2)
            
            # SHAP EXPLANATION
            with col1:
                st.markdown("#### Local Feature Contributions (SHAP)")
                shap_values, X_transformed, feature_names = explain_shap_single(pipeline, df_input)
                
                # Depending on the shap tree explainer version for binary class, it returns list or array
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                
                # Build a DataFrame of the top 10 features influencing this specific prediction
                shap_df = pd.DataFrame({
                    "Feature": feature_names,
                    "SHAP Value": shap_values
                })
                shap_df["Abs_SHAP"] = shap_df["SHAP Value"].abs()
                shap_df = shap_df.sort_values(by="Abs_SHAP", ascending=False).head(10)
                
                # Plot in Matplotlib for Streamlit
                fig, ax = plt.subplots(figsize=(6, 4))
                colors = ['red' if x > 0 else 'blue' for x in shap_df["SHAP Value"][::-1]]
                ax.barh(shap_df["Feature"][::-1], shap_df["SHAP Value"][::-1], color=colors)
                ax.set_xlabel("SHAP Value (Red = Increases Attrition Risk)")
                ax.set_title("Top 10 SHAP Feature Importances")
                st.pyplot(fig)
                
            # LIME EXPLANATION
            with col2:
                st.markdown("#### LIME Explanation")
                # Initialize LIME Explainer
                lime_explainer, cat_maps, lime_feats, lime_cats = get_lime_explainer(X_train_raw)
                
                # Generate explanation
                lime_exp = explain_lime_single(
                    pipeline, lime_explainer, cat_maps, lime_feats, lime_cats, df_input
                )
                
                # Render the raw HTML from LIME into the Streamlit app
                components.html(lime_exp.as_html(), height=400, scrolling=True)

if __name__ == "__main__":
    main()
