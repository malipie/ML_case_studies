import pandas as pd
import numpy as np
import argparse
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

# Set random seed for reproducibility
np.random.seed(42)

def load_data(path):
    """
    Loads data, preprocesses it, and splits it.
    """
    # Load data
    df = pd.read_csv(path)
    
    # Create binary target variable: 1 for 'good' (quality > 6), 0 for 'bad' (quality <= 6)
    df['quality_label'] = (df['quality'] > 6).astype(int)
    df = df.drop('quality', axis=1)
    
    # Define features (X) and target (y)
    X = df.drop('quality_label', axis=1)
    y = df['quality_label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test

def get_model(model_name):
    """
    Returns a scikit-learn model pipeline based on the model_name.
    """
    if model_name == "logistic_regression":
        model = LogisticRegression(random_state=42, max_iter=1000)
        
    elif model_name == "random_forest":
        model = RandomForestClassifier(random_state=42, n_estimators=100)
        
    elif model_name == "lightgbm":
        model = LGBMClassifier(random_state=42, n_estimators=100)
        
    else:
        raise ValueError(f"Unknown model_name: {model_name}. Choose from 'logistic_regression', 'random_forest', 'lightgbm'.")

    # Create a pipeline with scaling and the model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    return pipeline

def main(model_name):
    """
    Main function to train model and log experiment with MLflow.
    """
    # Load data
    X_train, X_test, y_train, y_test = load_data("data/winequality-red.csv")

    # Start an MLflow experiment run
    with mlflow.start_run():
        
        # Get model pipeline
        model = get_model(model_name)
        
        print(f"Training model: {model_name}...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # --- MLflow Logging ---
        
        # 1. Log Parameters
        mlflow.log_param("model_name", model_name)
        
        # Log model-specific parameters
        if model_name != "logistic_regression":
             mlflow.log_param("n_estimators", 100)
        
        # 2. Log Metrics
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

        # 3. Log the Model
        # This saves the model file (model.pkl) inside the mlruns directory
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Run for {model_name} logged successfully.")


if __name__ == "__main__":
    # --- Argument Parser ---
    # This allows us to run the script from the command line
    # like: python src/train.py --model_name random_forest
    
    parser = argparse.ArgumentParser(description="Train and log a model with MLflow.")
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="logistic_regression",
        help="Name of the model to train. Choose from: 'logistic_regression', 'random_forest', 'lightgbm'."
    )
    
    args = parser.parse_args()
    
    # Set experiment name in MLflow
    mlflow.set_experiment("Wine Quality Classification")
    
    # Run the main function
    main(args.model_name)