import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_selector as selector

# --- Configuration ---
# Use pathlib for robust cross-platform paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "housing.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "housing_model.pkl"
TEST_DATA_PATH = BASE_DIR / "data" / "test_data.pkl"

def load_data(path):
    """
    Loads data and prepares initial X/y split.
    """
    print(f"\nLoading data from {path}...")
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    data = pd.read_csv(path)
    
    # Drop rows ONLY where the target is missing
    # We leave other missing values for the Imputer in the Pipeline
    data = data.dropna(subset=['median_house_value'])
    
    print(f"Data loaded: {data.shape[0]} rows, {data.shape[1]} columns") 
    
    X = data.drop(columns=['median_house_value'])
    y = data['median_house_value']
    
    # Log-transform the target variable
    y = np.log1p(y) 
    
    return X, y

def build_pipeline():
    """
    Constructs the preprocessing and modeling pipeline.
    """
    # 1. Numerical Pipeline
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")), 
        ("scaler", StandardScaler())
    ])

    # 2. Categorical Pipeline
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # 3. Column Transformer (The Traffic Controller)
    # Using selectors makes it robust to new/changed columns
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, selector(dtype_exclude='object')),
        ("cat", categorical_transformer, selector(dtype_include='object')),
    ])

    # 4. Final Pipeline
    clf = Pipeline(steps=[
        ("preprocessor", preprocessor), 
        ("classifier", GradientBoostingRegressor(random_state=42))
    ])    

    return clf

def train_model(pipeline, X_train, y_train):
    print("Training model...")
    model = pipeline.fit(X_train, y_train)
    return model

def save_artifacts(model, X_test, y_test):
    """
    Saves the trained model and test data for later evaluation.
    """
    # Ensure directory exists
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save Model
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # Save Test Data
    joblib.dump((X_test, y_test), TEST_DATA_PATH)
    print(f"Test data saved to {TEST_DATA_PATH}")

def main():
    print("=== House Price Prediction Project (Refactored) ===")
    
    # 1. Load
    X, y = load_data(DATA_PATH)

    # 2. Split
    # Stratify is not typically used for regression unless mapped to bins, 
    # so simple random split is fine here.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Build
    model_pipeline = build_pipeline()
    
    # 4. Train
    trained_model = train_model(model_pipeline, X_train, y_train) 
    
    # 5. Save
    save_artifacts(trained_model, X_test, y_test)

    print("=== Pipeline finished successfully. ===")

if __name__=="__main__":
    main()