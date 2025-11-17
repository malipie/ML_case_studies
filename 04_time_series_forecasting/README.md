# 04_time_series_forecasting

Short project README for time series forecasting experiments and case studies.

## Project summary
This repository contains code, notebooks and experiments for time series forecasting. It focuses on data preparation, feature engineering, baseline and advanced forecasting models, evaluation and reproducibility of results.

## Goals
- Explore and preprocess time series data
- Implement baseline and ML/DL forecasting models
- Evaluate forecasts with standard metrics (MAE, RMSE, MAPE)
- Compare approaches and document findings

## Repository structure (typical)
- data/                — raw and processed datasets (do not commit large raw files)
- notebooks/           — exploratory analysis and experiment notebooks
- src/                 — source code (data loaders, features, models, training, evaluation)
- models/              — saved model checkpoints and serialized artifacts
- requirements.txt     — Python dependencies
- README.md            — this file

Adjust paths above to match the actual layout in the repository.

## Requirements
- Python 3.8+
- JupyterLab or Jupyter Notebook
- Typical libraries: numpy, pandas, matplotlib, scikit-learn, statsmodels, torch/keras (if deep learning used)
- Install from requirements.txt:
    pip install -r requirements.txt


## Usage
- Open and run notebooks in notebooks/ to reproduce analyses and plots:
    jupyter lab notebooks/
- Run training or evaluation scripts from src/ (example):
    python src/train.py


Replace example commands with actual script names and config paths present in the project.



# Analysis & Key Findings

After training the LGBM model, the Feature Importance plot provides several critical business insights:

## 1. Sales are Driven by Trends, Not Long-Term Growth

The most important features by far are rolling_mean_28 and rolling_mean_7.

Business Insight: This indicates that the business is highly auto-regressive. The best predictor of future sales is the sales performance over the last 7-28 days. This suggests strong, consistent sales patterns rather than erratic spikes.

The low importance of the year feature suggests there is no strong long-term upward or downward trend across the entire business.

## 2. Product-Level Seasonality is Key

The item ID is the third most important feature.

Business Insight: This proves that different items have vastly different sales patterns (i.e., strong product-level seasonality). The model learned that it must know which item is being sold to make an accurate prediction (e.g., item 5 sells differently in summer than item 10).

## 3. Location is Not a Major Factor

One of the most interesting findings is the very low importance of the store feature.

Business Insight: This strongly suggests that sales performance is consistent across all 10 store locations. The primary drivers of sales are the product itself and its recent trend, not where it's being sold. This could imply a uniform customer base or similar assortment and pricing strategies across all stores.

