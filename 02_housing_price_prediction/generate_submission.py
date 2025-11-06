import joblib
import pandas as pd
import numpy as np
import sys

# --- Configuration ---
MODEL_PATH = "models/best_model.pkl"
KAGGLE_TEST_DATA_PATH = "data/test.csv"
SUBMISSION_PATH = "submission.csv"  # Save in root folder

def main():
    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please run 'python src/train.py' first to train and save the model.")
        sys.exit(1)

    print(f"Loading Kaggle test data from {KAGGLE_TEST_DATA_PATH}...")
    try:
        df_kaggle_test = pd.read_csv(KAGGLE_TEST_DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Test data not found at {KAGGLE_TEST_DATA_PATH}")
        print("Please download 'test.csv' from the Kaggle competition page "
              "and place it in the 'data/' folder.")
        sys.exit(1)

    # Keep 'Id' for the final submission file
    test_ids = df_kaggle_test['Id']
    
    # Use *exactly the same* columns as in training
    # (Drop 'Id' as it wasn't in X_train)
    X_kaggle_test = df_kaggle_test.drop('Id', axis=1)

    # --- Generate Predictions ---
    # The Pipeline automatically handles all preprocessing
    print("Making predictions on Kaggle's test.csv...")
    y_pred_log_kaggle = model.predict(X_kaggle_test)

    # Revert the log-transform to get actual dollar values
    y_pred_kaggle = np.expm1(y_pred_log_kaggle)
    print("Predictions generated and log-transform reverted.")

    # --- Create Submission File ---
    submission_df = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': y_pred_kaggle
    })

    submission_df.to_csv(SUBMISSION_PATH, index=False)
    
    print(f"\nSuccess! Submission file created at: {SUBMISSION_PATH}")
    print(submission_df.head())

if __name__ == "__main__":
    main()