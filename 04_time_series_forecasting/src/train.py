import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_PATH = 'data/train.csv'
MODEL_DIR = 'models'
MODEL_PATH = f'{MODEL_DIR}/lgbm_model.pkl'
VALIDATION_DATA_PATH = f'{MODEL_DIR}/validation_data.pkl'
np.random.seed(42)

# --- Feature Engineering Function ---

def create_time_features(df):
    """
    Creates time-series features based on a date column.
    """
    df_copy = df.copy()
    df_copy['date'] = pd.to_datetime(df_copy['date'])
    
    # Basic Time Features
    df_copy['month'] = df_copy['date'].dt.month
    df_copy['year'] = df_copy['date'].dt.year
    df_copy['dayofweek'] = df_copy['date'].dt.dayofweek
    df_copy['dayofmonth'] = df_copy['date'].dt.day
    df_copy['dayofyear'] = df_copy['date'].dt.dayofyear
    
    # Lag Features (Sales from previous periods)
    # We will sort by store, item, then date to ensure lags are correct
    df_copy = df_copy.sort_values(by=['store', 'item', 'date'], ascending=True)
    
    print("Creating lag features...")
    # Group by store and item to calculate lags
    g = df_copy.groupby(['store', 'item'])
    
    # Sales 7 days ago
    df_copy['lag_7'] = g['sales'].shift(7)
    # Sales 28 days ago
    df_copy['lag_28'] = g['sales'].shift(28)
    
    # Rolling Mean (Average sales over a window)
    print("Creating rolling mean features...")
    # Average sales in the last 7 days (not including today)
    df_copy['rolling_mean_7'] = g['sales'].shift(1).rolling(7).mean()
    # Average sales in the last 28 days (not including today)
    df_copy['rolling_mean_28'] = g['sales'].shift(1).rolling(28).mean()
    
    # Drop rows with NaN (created by lags/rolling)
    df_copy = df_copy.dropna()
    
    return df_copy

# --- Main Execution ---

def main():
    # 1. Load Data
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # 2. Create Time Features
    print("Starting feature engineering...")
    df_features = create_time_features(df)
    
    # 3. Define Features (X) and Target (y)
    target = 'sales'
    
    # We drop the original 'date' and 'sales' columns
    features = [col for col in df_features.columns if col not in ['date', 'sales']]
    
    X = df_features[features]
    y = df_features[target]
    
    # 4. Create Time-Based Validation Split
    # We cannot shuffle! We must validate on the "future".
    # Let's use 2017 as our validation set.
    
    print("Creating time-based train/validation split...")
    
    # Data from 2017
    X_val = X[df_features['year'] == 2017]
    y_val = y[df_features['year'] == 2017]
    
    # Data before 2017
    X_train = X[df_features['year'] < 2017]
    y_train = y[df_features['year'] < 2017]
    
    # Save the validation set for the notebook
    # We also save the dates for plotting
    val_dates = df_features[df_features['year'] == 2017]['date']
    joblib.dump((X_val, y_val, val_dates), VALIDATION_DATA_PATH)
    
    print(f"Training data size: {len(X_train)}")
    print(f"Validation data size: {len(X_val)}")
    
    # 5. Train the LightGBM Model
    print("Training LightGBM model...")
    
    # Define categorical features for LightGBM
    categorical_features = ['store', 'item', 'month', 'dayofweek']
    
    lgb_model = lgb.LGBMRegressor(
        random_state=42,
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        n_jobs=-1,
        verbose=-1
    )
    
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(100, verbose=True)],
        categorical_feature=categorical_features
    )
    
    # 6. Evaluate on Validation Set
    y_pred = lgb_model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"\nValidation RMSE: {rmse:.4f}")
    
    # 7. Save the Best Model
    print(f"Saving best model to {MODEL_PATH}...")
    joblib.dump(lgb_model, MODEL_PATH)

    print("\nTraining script finished successfully.")

if __name__ == "__main__":
    main()