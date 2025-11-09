import mlflow
import pandas as pd
import argparse
import sys

# Define the feature names, matching the training data
FEATURE_NAMES = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol'
]
EXPERIMENT_NAME = "Wine Quality Classification"

def load_best_model_auto():
    """
    Finds the best run based on 'f1_score' (our default metric)
    and loads its model.
    """
    print(f"Strategy: Automatic. Finding best model in experiment '{EXPERIMENT_NAME}' based on F1 score.")
    
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Search all runs, order by f1_score descending
    runs = mlflow.search_runs(
        order_by=["metrics.f1_score DESC"],
        max_results=1
    )
    
    if len(runs) == 0:
        raise Exception(f"No runs found in experiment '{EXPERIMENT_NAME}'")
        
    best_run = runs.iloc[0]
    best_run_id = best_run.run_id
    best_f1 = best_run["metrics.f1_score"]
    best_model_name = best_run["params.model_name"]
    
    print(f"--- Best Model (Auto) ---")
    print(f"  Model: {best_model_name}")
    print(f"  F1 Score: {best_f1:.4f}")
    
    return load_model_by_id(best_run_id)

def load_model_by_id(run_id):
    """
    Loads a specific model from MLflow using its Run ID.
    """
    print(f"Strategy: Manual. Loading model from Run ID: {run_id}")
    try:
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model from Run ID '{run_id}'.")
        print(f"Error details: {e}")
        return None

def main(strategy, run_id, sample_data):
    """
    Main function to load the model (based on strategy) and make a prediction.
    """
    model = None
    try:
        if strategy == "manual":
            if run_id is None:
                print("Error: Strategy 'manual' requires a --run_id to be provided.")
                sys.exit(1)
            model = load_model_by_id(run_id)
        
        elif strategy == "auto":
            model = load_best_model_auto()
            
    except Exception as e:
        print(f"Error: {e}")
        return

    if model is None:
        print("Model could not be loaded. Exiting.")
        return

    # Create a DataFrame from the sample data
    try:
        df = pd.DataFrame([sample_data], columns=FEATURE_NAMES)
    except ValueError:
        print(f"Error: Sample data has wrong number of features. Expected {len(FEATURE_NAMES)}.")
        return

    # Make a prediction
    prediction = model.predict(df)
    probability = model.predict_proba(df)
    
    result_label = "Good Quality (1)" if prediction[0] == 1 else "Bad Quality (0)"
    confidence = probability[0][prediction[0]] * 100

    print("\n--- Prediction Result ---")
    print(f"Input data: {sample_data}")
    print(f"Predicted Label: {result_label}")
    print(f"Confidence: {confidence:.2f}%")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Predict wine quality using a model from MLflow.")
    
    # --- Strategy Arguments ---
    parser.add_argument(
        "--strategy",
        type=str,
        default="auto",
        choices=["auto", "manual"],
        help="Strategy to select model: 'auto' (best F1 score) or 'manual' (requires --run_id)."
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="The Run ID to use (required if strategy is 'manual')."
    )
    
    # --- Sample Data Arguments ---
    parser.add_argument("--fixed_acidity", type=float, default=7.4)
    parser.add_argument("--volatile_acidity", type=float, default=0.7)
    parser.add_argument("--citric_acid", type=float, default=0.0)
    parser.add_argument("--residual_sugar", type=float, default=1.9)
    parser.add_argument("--chlorides", type=float, default=0.076)
    parser.add_argument("--free_sulfur_dioxide", type=float, default=11.0)
    parser.add_argument("--total_sulfur_dioxide", type=float, default=34.0)
    parser.add_argument("--density", type=float, default=0.9978)
    parser.add_argument("--pH", type=float, default=3.51)
    parser.add_argument("--sulphates", type=float, default=0.56)
    parser.add_argument("--alcohol", type=float, default=9.4)


    args = parser.parse_args()
    
    sample_data = [
        args.fixed_acidity, args.volatile_acidity, args.citric_acid,
        args.residual_sugar, args.chlorides, args.free_sulfur_dioxide,
        args.total_sulfur_dioxide, args.density, args.pH,
        args.sulphates, args.alcohol
    ]
    
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    main(args.strategy, args.run_id, sample_data)