# California Housing Price Prediction

This project demonstrates a clean, production-ready Machine Learning pipeline for regression tasks. It predicts median house values in California districts using the 1990 Census data.

The focus of this project is on Software Engineering standards in Data Science: modularity, reproducibility, and preventing data leakage.

# Key Features

Robust Pipeline: Uses sklearn.pipeline.Pipeline and ColumnTransformer to encapsulate all preprocessing steps (Imputation, Scaling, Encoding).

Leakage Prevention: Strict separation of training and testing data before any processing occurs.

Dynamic Preprocessing: Uses make_column_selector to automatically handle numerical and categorical columns, making the code resilient to schema changes.

Log-Transformation: Applies np.log1p to the target variable to handle the skewed distribution of house prices.

Model: Utilizes GradientBoostingRegressor for capturing non-linear relationships in the data.

# Project Structure

final_exam/
├── data/
│   └── housing.csv          (Raw data)
├── models/
│   └── housing_model.pkl    (Trained pipeline)
│   └── test_data.pkl        (Test set for evaluation)
├── src/
│   ├── train.py             (Main training script)
├── notebooks/
│   └── analysis.ipynb       (Evaluation and diagnostics)
├── requirements.txt
└── README.md


# How to Run

### Install dependencies:

pip install -r requirements.txt


### Train the model:
This script handles loading, splitting, preprocessing, training, and saving artifacts.

python src/train.py


### Evaluate:
Open notebooks/notebook.ipynb to visualize performance metrics, residual plots, and feature importance.

# Analysis & Key Findings

The model evaluation revealed several critical insights about the housing market and the model's performance:

### 1. The Power of Log-Transformation

Applying np.log1p() to the target variable (median_house_value) was crucial. Without it, the model would be heavily biased by high-priced outliers. The log-transform normalized the distribution, allowing the model to focus on relative (percentage) errors rather than absolute dollar errors.

### 2. Location is Key (but complex)

The Feature Importance analysis and correlation heatmap show that median_income and ocean_proximity are highly predictive features.

Insight: This confirms that location relative to the coast and the wealth of the neighborhood are primary drivers of value. However, the geographic visualization reveals non-linear patterns (price hotspots) that require tree-based models like Gradient Boosting to capture effectively.

### 3. Limitations in High-Value Areas

The Regression Plot reveals a systematic under-prediction for high-value properties (capped at $500k in the dataset).

Observation: The model performs well for the mass market but hits a "ceiling" for luxury homes.

Reason: The dataset contains a cap on the target variable ($500,001), which introduces artificial bias for expensive homes.

Recommendation: Future iterations should treat capped values separately or use a model robust to censored data.