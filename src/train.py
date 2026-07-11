import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# Import helpers from utils.py
from src.utils import clean_dataframe, encode_target, split_features_target

def main():
    print("Loading raw data...")
    # Assume script is run from project root
    data_path = "data/raw/XAI_HR_NUMERIC.csv"
    df = pd.read_csv(data_path)

    print("Cleaning and preparing data...")
    df = clean_dataframe(df)
    df = encode_target(df)
    
    # Split into features and target
    X, y = split_features_target(df)

    print("Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Building preprocessing pipeline...")
    # Dynamically identify categorical and numeric columns
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols)
        ]
    )

    print("Computing scale_pos_weight...")
    # Calculate scale_pos_weight based on the training set class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    print("Initializing XGBoost Pipeline...")
    xgb_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            random_state=42,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            base_score=0.5
        ))
    ])

    print("Training model...")
    xgb_pipeline.fit(X_train, y_train)

    print("Saving full pipeline to models/xgb_pipeline.pkl...")
    os.makedirs("models", exist_ok=True)
    with open("models/xgb_pipeline.pkl", "wb") as f:
        pickle.dump(xgb_pipeline, f)
        
    print("Training complete!")

if __name__ == "__main__":
    main()
