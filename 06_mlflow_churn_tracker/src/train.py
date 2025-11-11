import pandas as pd
import numpy as np
import argparse
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Set random seed for reproducibility
np.random.seed(42)

# Define the database URI. This tells MLflow to use a file named 'mlflow.db'.
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

def load_data(path):
    """
    Loads data, preprocesses it, and splits it.
    """
    df = pd.read_csv(path)
    
    # --- Data Cleaning ---
    # Drop customerID
    df = df.drop("customerID", axis=1)
    
    # Convert 'TotalCharges' to numeric, forcing errors (like spaces) to NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Convert target 'Churn' to binary
    df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})
    
    # Drop rows where target is missing (if any)
    df = df.dropna(subset=['Churn'])
    
    # Define features (X) and target (y)
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Identify numerical and categorical features
    # Numerical features:
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    
    # Categorical features:
    # We explicitly exclude the numerical ones we just found
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test, numerical_features, categorical_features

def get_model_pipeline(model_name, num_features, cat_features):
    """
    Creates a full preprocessing and modeling pipeline.
    This combines ColumnTransformer with a model.
    """
    
    # --- Create Preprocessing Pipelines ---
    
    # Numerical pipeline: Impute missing values (e.g., in TotalCharges) then scale
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline: Impute missing values then one-hot encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # --- Create the ColumnTransformer ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, num_features),
            ('cat', categorical_transformer, cat_features)
        ],
        remainder='passthrough'
    )
    
    # --- Get the selected model ---
    if model_name == "logistic_regression":
        model = LogisticRegression(random_state=42, max_iter=1000)
    elif model_name == "random_forest":
        model = RandomForestClassifier(random_state=42, n_estimators=100)
    elif model_name == "lightgbm":
        model = LGBMClassifier(random_state=42, n_estimators=100)
    else:
        raise ValueError(f"Unknown model_name: {model_name}.")

    # --- Create the Final Pipeline ---
    final_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    return final_pipeline

def main(model_name):
    """
    Main function to train model and log experiment with MLflow.
    """
    # Load and preprocess data
    X_train, X_test, y_train, y_test, num_features, cat_features = load_data("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    # --- MLflow Setup ---
    # This is the KEY UPGRADE. We tell MLflow to use our SQLite database file.
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Telco Customer Churn")

    # Start an MLflow run
    with mlflow.start_run():
        
        # Get model pipeline
        model = get_model_pipeline(model_name, num_features, cat_features)
        
        print(f"Training model: {model_name}...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # --- MLflow Logging ---
        
        # 1. Log Parameters
        mlflow.log_param("model_name", model_name)
        if hasattr(model.named_steps['model'], 'n_estimators'):
             mlflow.log_param("n_estimators", model.named_steps['model'].n_estimators)
        
        # 2. Log Metrics (F1 is key for imbalanced data)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        
        print(f"Metrics for {model_name}:")
        print(f"- Accuracy: {accuracy:.4f}")
        print(f"- F1 Score: {f1:.4f}")

        # 3. Log the Model (as an artifact)
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Run for {model_name} logged successfully to {MLFLOW_TRACKING_URI}.")


if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Train and log a churn model with MLflow.")
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="logistic_regression",
        help="Model to train: 'logistic_regression', 'random_forest', 'lightgbm'."
    )
    
    args = parser.parse_args()
    
    # Run the main function
    main(args.model_name)