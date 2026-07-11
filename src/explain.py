import pandas as pd
import numpy as np
import shap
from lime.lime_tabular import LimeTabularExplainer

# ==========================================
# SHAP Helpers
# ==========================================

def get_shap_explainer(pipeline):
    """
    Returns a SHAP TreeExplainer for the XGBoost model within the pipeline.
    """
    xgb_model = pipeline.named_steps["model"]
    return shap.TreeExplainer(xgb_model)

def explain_shap_global(pipeline, X_data):
    """
    Computes global SHAP values for a given dataset (e.g., X_test).
    Returns shap_values, transformed data, and clean feature names for plotting.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    X_transformed = preprocessor.transform(X_data)
    
    explainer = get_shap_explainer(pipeline)
    shap_values = explainer.shap_values(X_transformed)
    
    # Get and clean feature names
    feature_names = preprocessor.get_feature_names_out()
    feature_names = [f.replace("cat__", "").replace("num__", "") for f in feature_names]
    
    return shap_values, X_transformed, feature_names

def explain_shap_single(pipeline, X_single):
    """
    Computes SHAP values for a single instance.
    X_single should be a raw DataFrame with 1 row.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    X_transformed = preprocessor.transform(X_single)
    
    explainer = get_shap_explainer(pipeline)
    shap_values = explainer.shap_values(X_transformed)
    
    feature_names = preprocessor.get_feature_names_out()
    feature_names = [f.replace("cat__", "").replace("num__", "") for f in feature_names]
    
    return shap_values, X_transformed[0], feature_names


# ==========================================
# LIME Helpers
# ==========================================

def get_lime_explainer(X_train):
    """
    Initializes a LimeTabularExplainer using the raw training data.
    Returns the explainer, category mappings (needed for predictions), 
    and lists of feature names/categorical columns.
    """
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    categorical_indices = [X_train.columns.get_loc(col) for col in categorical_cols]
    
    lime_X_train_encoded = X_train.copy()
    category_mappings = {}
    
    # Convert string categoricals to numerical codes for LIME's internal math
    for col in categorical_cols:
        cat_series = X_train[col].astype("category")
        lime_X_train_encoded[col] = cat_series.cat.codes
        category_mappings[col] = dict(enumerate(cat_series.cat.categories))

    feature_names = lime_X_train_encoded.columns.tolist()
    
    explainer = LimeTabularExplainer(
        training_data=lime_X_train_encoded.values,
        feature_names=feature_names,
        class_names=["No Attrition", "Attrition"],
        categorical_features=categorical_indices,
        mode="classification"
    )
    
    return explainer, category_mappings, feature_names, categorical_cols


def explain_lime_single(pipeline, explainer, category_mappings, feature_names, categorical_cols, X_single):
    """
    Generates a LIME explanation for a single instance.
    """
    # X_single needs to be numerically encoded first for LIME's explain_instance input
    X_single_encoded = X_single.copy()
    
    for col in categorical_cols:
        mapping = {v: k for k, v in category_mappings[col].items()}
        val = X_single_encoded[col].iloc[0]
        # Assign numeric code, fallback to -1 if somehow unseen
        X_single_encoded[col] = mapping.get(val, -1) 
    
    # Define a prediction function that maps LIME's perturbed numerical arrays back to strings for the pipeline
    def predict_fn_lime(x_numpy):
        x_df = pd.DataFrame(x_numpy, columns=feature_names)
        for col in categorical_cols:
            x_df[col] = x_df[col].round().astype(int)
            x_df[col] = x_df[col].map(category_mappings[col])
        return pipeline.predict_proba(x_df)
    
    exp = explainer.explain_instance(
        data_row=X_single_encoded.iloc[0].values,
        predict_fn=predict_fn_lime
    )
    
    return exp


# ==========================================
# DiCE Helpers
# ==========================================

def get_dice_explainer(pipeline, df_train_full, continuous_features=None):
    """
    Initializes a DiCE explainer. 
    df_train_full MUST contain both the features AND the target column ('Attrition').
    """
    import dice_ml
    from dice_ml import Dice
    if continuous_features is None:
        continuous_features = ["Age", "MonthlyIncome", "YearsAtCompany", "DistanceFromHome"]
        
    data_dice = dice_ml.Data(
        dataframe=df_train_full,
        continuous_features=continuous_features,
        outcome_name="Attrition"
    )
    
    model_dice = dice_ml.Model(
        model=pipeline,
        backend="sklearn"
    )
    
    dice_exp = Dice(data_dice, model_dice, method="random")
    return dice_exp


def explain_dice_single(dice_explainer, X_single, total_CFs=3):
    """
    Generates counterfactuals for a single instance using DiCE.
    """
    counterfactuals = dice_explainer.generate_counterfactuals(
        X_single,
        total_CFs=total_CFs,
        desired_class="opposite"
    )
    return counterfactuals
