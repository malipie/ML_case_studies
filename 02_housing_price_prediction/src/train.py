import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import randint, uniform

# --- Configuration ---
DATA_PATH = 'data/train.csv'
MODEL_DIR = 'models'
MODEL_PATH = f'{MODEL_DIR}/best_model.pkl'
TEST_DATA_PATH = f'{MODEL_DIR}/test_data.pkl'
TARGET_COLUMN = 'SalePrice'
np.random.seed(42)

def main():
    # 1. Load Data
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    # 2. Prepare Data
    print("Preparing data...")
    # Drop the Id column
    df = df.drop('Id', axis=1)
    
    # Separate features and target
    X = df.drop(TARGET_COLUMN, axis=1)
    # Apply log transform to target variable to normalize it
    y = np.log1p(df[TARGET_COLUMN])

    # Save list in correct order
    feature_list = X.columns.tolist()
    joblib.dump(feature_list, f'{MODEL_DIR}/feature_list.pkl')
    print("Feature list saved to models/feature_list.pkl")

    # 3. Identify Column Types
    # Identify numerical and categorical features
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()
    print(f"Found {len(numerical_features)} numerical features.")
    print(f"Found {len(categorical_features)} categorical features.")

    # 4. Split Data (before preprocessing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Define Preprocessing Pipelines
    
    # Pipeline for NUMERICAL features:
    # 1. Impute missing values with the median
    # 2. Scale features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Pipeline for CATEGORICAL features:
    # 1. Impute missing values with a constant "missing" string
    # 2. One-hot encode the categories
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # 6. Create the ColumnTransformer
    # This transformer applies the correct pipeline to the correct columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Keep any columns not listed
    )

    # 7. Create the Final Pipeline (Preprocessor + Model)
    # We will use XGBoost as our model
    main_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', XGBRegressor(random_state=42, objective='reg:squarederror'))
    ])

    # 8. Define Parameter Grid for RandomizedSearch
    # Tuning hyperparameters for XGBoost
    param_dist = {
        'model__n_estimators': randint(100, 1000),
        'model__learning_rate': uniform(0.01, 0.3),
        'model__max_depth': randint(3, 10),
        'model__subsample': uniform(0.7, 0.3),
        'model__colsample_bytree': uniform(0.7, 0.3)
    }

    # 9. Run RandomizedSearchCV
    print("Running RandomizedSearch to find best XGBoost parameters...")


    
    random_search = RandomizedSearchCV(
        main_pipeline,
        param_distributions=param_dist,
        n_iter=20, # 20 combinations
        cv=5,
        verbose=2,
        random_state=42,
        n_jobs=-1,
        scoring='neg_root_mean_squared_error' # Optimize for RMSE
    )

    random_search.fit(X_train, y_train)

    # 10. Print Best Results
    print("\nSearch complete.")
    print(f"Best cross-validation score (neg RMSE): {random_search.best_score_:.4f}")
    print("Best parameters found:")
    print(random_search.best_params_)

    # 11. Save the Best Model
    best_model = random_search.best_estimator_
    print(f"\nSaving best model to {MODEL_PATH}...")
    joblib.dump(best_model, MODEL_PATH)

    # 12. Save Test Data for Notebook
    print(f"Saving test data to {TEST_DATA_PATH}...")
    joblib.dump((X_test, y_test), TEST_DATA_PATH)

    print("\nTraining script finished successfully.")

if __name__ == "__main__":
    main()