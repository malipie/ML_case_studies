import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional

# --- 1. Configuration and Model Loading ---

# Initialize FastAPI application
app = FastAPI(title="Housing Price Prediction API",
              description="API for predicting house prices in Ames, Iowa using an XGBoost model.",
              version="1.0")

# Load our finished pipeline
# The model is loaded ONLY ONCE at startup.
try:
    model = joblib.load("app/model/best_model.pkl")
    model_features = joblib.load("app/model/feature_list.pkl")
    print("Model and feature list loaded successfully.")
except Exception as e:
    print(f"Error loading model or features: {e}")
    model = None # Set model to None if loading fails
    model_features = None

# --- 2. Data Model Definition (API Contract) ---

# Pydantic is used to define EXACTLY what data
# /predict endpoint expects.
# ColumnTransformer pipeline was trained on 79 features.
# We must define them all here.

# IMPORTANT:
# "Optional[type] = None", which means the field is optional.
# Because  pipeline has a 'SimpleImputer' (to fill missing values),
# the user doesn't have to send all 79 features. They can send only
# the ones they know, and the pipeline will handle the rest.

# Script will make the two most important features REQUIRED to show how.

class HouseFeatures(BaseModel):
    # Required features (no default value)
    OverallQual: int
    GrLivArea: int
    
    # --- Optional Features (default value = None) ---
    MSSubClass: Optional[int] = None
    MSZoning: Optional[str] = None
    LotFrontage: Optional[float] = None
    LotArea: Optional[int] = None
    Street: Optional[str] = None
    Alley: Optional[str] = None
    LotShape: Optional[str] = None
    LandContour: Optional[str] = None
    Utilities: Optional[str] = None
    LotConfig: Optional[str] = None
    LandSlope: Optional[str] = None
    Neighborhood: Optional[str] = None
    Condition1: Optional[str] = None
    Condition2: Optional[str] = None
    BldgType: Optional[str] = None
    HouseStyle: Optional[str] = None
    OverallCond: Optional[int] = None
    YearBuilt: Optional[int] = None
    YearRemodAdd: Optional[int] = None
    RoofStyle: Optional[str] = None
    RoofMatl: Optional[str] = None
    Exterior1st: Optional[str] = None
    Exterior2nd: Optional[str] = None
    MasVnrType: Optional[str] = None
    MasVnrArea: Optional[float] = None
    ExterQual: Optional[str] = None
    ExterCond: Optional[str] = None
    Foundation: Optional[str] = None
    BsmtQual: Optional[str] = None
    BsmtCond: Optional[str] = None
    BsmtExposure: Optional[str] = None
    BsmtFinType1: Optional[str] = None
    BsmtFinSF1: Optional[int] = None
    BsmtFinType2: Optional[str] = None
    BsmtFinSF2: Optional[int] = None
    BsmtUnfSF: Optional[int] = None
    TotalBsntSF: Optional[int] = None
    Heating: Optional[str] = None
    HeatingQC: Optional[str] = None
    CentralAir: Optional[str] = None
    Electrical: Optional[str] = None
    BsmtFullBath: Optional[int] = None
    BsmtHalfBath: Optional[int] = None
    FullBath: Optional[int] = None
    HalfBath: Optional[int] = None
    BedroomAbvGr: Optional[int] = None
    KitchenAbvGr: Optional[int] = None
    KitchenQual: Optional[str] = None
    TotRmsAbvGrd: Optional[int] = None
    Functional: Optional[str] = None
    Fireplaces: Optional[int] = None
    FireplaceQu: Optional[str] = None
    GarageType: Optional[str] = None
    GarageYrBlt: Optional[float] = None
    GarageFinish: Optional[str] = None
    GarageCars: Optional[int] = None
    GarageArea: Optional[int] = None
    GarageQual: Optional[str] = None
    GarageCond: Optional[str] = None
    PavedDrive: Optional[str] = None
    WoodDeckSF: Optional[int] = None
    OpenPorchSF: Optional[int] = None
    EnclosedPorch: Optional[int] = None
    ScreenPorch: Optional[int] = None
    PoolArea: Optional[int] = None
    PoolQC: Optional[str] = None
    Fence: Optional[str] = None
    MiscFeature: Optional[str] = None
    MiscVal: Optional[int] = None
    MoSold: Optional[int] = None
    YrSold: Optional[int] = None
    SaleType: Optional[str] = None
    SaleCondition: Optional[str] = None

    # --- Special handling for "bad" column names ---
    # The original data had columns like '1stFlrSF'.
    # This is not a valid Python variable name.
    # Script uses `Field(alias=...)` to map our "clean"
    # Python name (FirstFlrSF) to the "dirty" JSON name (1stFlrSF).
    
    FirstFlrSF: Optional[int] = Field(None, alias='1stFlrSF')
    SecondFlrSF: Optional[int] = Field(None, alias='2ndFlrSF')
    ThreeSsnPorch: Optional[int] = Field(None, alias='3SsnPorch')

    class Config:
        populate_by_name = True # Allows using the aliases


# --- 3. API Endpoint Definition ---

@app.get("/")
def read_root():
    """Main endpoint that just welcomes the user."""
    return {"message": "Welcome to the Ames Housing Price Prediction API!"}


@app.post("/predict")
def predict_price(features: HouseFeatures):
    """
    Endpoint for predicting house price.
    Accepts house data in JSON format and returns the predicted price.
    """
    if model is None:
        return {"error": "Model not loaded. Please check server logs."}, 500

    try:
        # 1. Convert Pydantic data to a dictionary
        # We use `by_alias=True` to correctly handle '1stFlrSF', etc.
        data_dict = features.model_dump(by_alias=True)
        
        # 2. Convert dictionary to a 1-row DataFrame
        # Our scikit-learn pipeline expects a DataFrame
        df = pd.DataFrame.from_records([data_dict], columns=model_features)
        
        # 3. Use the pipeline to predict
        # model.predict() will run the FULL pipeline:
        # (Imputer -> Scaler -> OneHotEncoder -> XGBoost)
        y_pred_log = model.predict(df)
        
        # 4. Revert the log-transform
        y_pred = np.expm1(y_pred_log[0])
        
        # 5. Return the result
        return {"predicted_price": float(round(y_pred, 2))}

    except Exception as e:
        return {"error": f"Error during prediction: {str(e)}"}, 400