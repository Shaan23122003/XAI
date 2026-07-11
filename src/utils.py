import pandas as pd
import numpy as np
import pickle

# --- Constants ---
TARGET_COL = "Attrition"

# Columns dropped in the notebook before training
DROP_COLS = [
    "EmployeeCount", 
    "EmployeeNumber", 
    "Over18", 
    "StandardHours", 
    "Unnamed: 0"
]

# Extracted from data/raw/XAI_HR_NUMERIC.csv (excluding target and dropped cols)
FEATURE_COLUMNS = [
    "Age",
    "BusinessTravel",
    "DailyRate",
    "Department",
    "DistanceFromHome",
    "Education",
    "EducationField",
    "EnvironmentSatisfaction",
    "Gender",
    "HourlyRate",
    "JobInvolvement",
    "JobLevel",
    "JobRole",
    "JobSatisfaction",
    "MaritalStatus",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "OverTime",
    "PercentSalaryHike",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager"
]

# --- Functions ---

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops useless / identifier-like columns from the dataset.
    """
    df_cleaned = df.copy()
    cols_to_drop = [col for col in DROP_COLS if col in df_cleaned.columns]
    return df_cleaned.drop(columns=cols_to_drop)

def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes the target column from 'No'/'Yes' to 0/1.
    """
    df_encoded = df.copy()
    if TARGET_COL in df_encoded.columns:
        df_encoded[TARGET_COL] = df_encoded[TARGET_COL].map({"No": 0, "Yes": 1})
    return df_encoded

def split_features_target(df: pd.DataFrame):
    """
    Splits the dataframe into features (X) and target (y).
    """
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataframe.")
    
    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]
    return X, y

def prepare_single_input(input_dict: dict) -> pd.DataFrame:
    """
    Converts a dictionary of input features into a DataFrame suitable for prediction.
    Ensures column ordering strictly matches FEATURE_COLUMNS.
    """
    df_input = pd.DataFrame([input_dict])
    
    if FEATURE_COLUMNS:
        # Reorder and ensure all required columns are present
        for col in FEATURE_COLUMNS:
            if col not in df_input.columns:
                df_input[col] = np.nan
        df_input = df_input[FEATURE_COLUMNS]
        
    return df_input

def load_model(model_path: str):
    """
    Loads a serialized model pipeline from the given path.
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def format_prediction(pred, prob) -> dict:
    """
    Formats the raw prediction and probabilities into a readable dictionary.
    Assumes binary classification where 1 = Attrition, 0 = No Attrition.
    """
    # Depending on model/pipeline, pred might be a numpy array or a scalar
    pred_value = int(pred[0]) if isinstance(pred, (np.ndarray, list)) else int(pred)
    
    prediction_label = "Attrition" if pred_value == 1 else "No Attrition"
    
    # prob is usually a 2D array from predict_proba: [[prob_class_0, prob_class_1]]
    if isinstance(prob, (np.ndarray, list)) and len(prob.shape) == 2:
        probability = float(prob[0][1]) if pred_value == 1 else float(prob[0][0])
    else:
        # Fallback if probability is a 1D array
        probability = float(prob[1]) if pred_value == 1 else float(prob[0])
    
    return {
        "prediction": prediction_label,
        "probability": round(probability, 4)
    }
