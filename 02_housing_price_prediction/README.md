Short README for the Housing Price Prediction project.

## Project goal
Build and evaluate a regression model to predict housing prices from property attributes and local/environmental features. The workflow includes data ingestion, exploratory data analysis (EDA), feature engineering, model training, evaluation and prediction on new data.

## Repository structure (example)
- data/
    - raw/               — raw source files (e.g. `housing.csv`)
    - processed/         — cleaned and feature-engineered datasets used for training
- notebooks/           — Jupyter notebooks for EDA, visualization and experiments
- src/
    - data.py            — data loading and preprocessing utilities
    - features.py        — feature engineering functions
    - train.py           — training pipeline and model persistence
    - predict.py         — inference script for new samples
    - evaluate.py        — metrics and evaluation utilities
- models/              — saved model artifacts (e.g. `.pkl`, `.joblib`)
- requirements.txt     — Python dependencies
- README.md            — this file

## Requirements
- Python 3.8+
- See `requirements.txt` for packages (typical: numpy, pandas, scikit-learn, matplotlib, seaborn, joblib)

Installation (recommended inside a virtual environment):
```
python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Data
Place original raw data in `data/raw/`. Processed datasets produced by preprocessing scripts are stored in `data/processed/`. Keep sensitive or large datasets out of version control and document their source.

## Typical workflow
1. Inspect data and perform EDA using notebooks in `notebooks/`.
2. Run preprocessing to produce cleaned datasets:
     ```
     python src/data.py --input data/raw/housing.csv --output data/processed/train.csv
     ```
3. Create or modify feature engineering code in `src/features.py`.
4. Train model:
     ```
     python src/train.py --data data/processed/train.csv --model-path models/housing_model.pkl
     ```
5. Evaluate model:
     ```
     python src/evaluate.py --model models/housing_model.pkl --test-data data/processed/test.csv
     ```
6. Predict on new data:
     ```
     python src/predict.py --model models/housing_model.pkl --input new_samples.csv --output predictions.csv
     ```

## Evaluation
Use regression metrics such as RMSE, MAE and R² to assess performance. Include cross-validation and hold-out test sets in experiments. Log hyperparameters and results for reproducibility.

## Reproducibility and experiments
- Pin package versions in `requirements.txt`.
- Save preprocessing steps and model artifacts to `models/`.
- Use notebooks for exploratory work and scripts for reproducible runs.


# Analysis & Key Findings

The model evaluation revealed several critical insights about the housing market and the model's performance:

## 1. The Power of Log-Transformation

Applying np.log1p() to the target variable (SalePrice) was crucial. Without it, the model would be heavily biased by high-priced outliers. The log-transform normalized the distribution, allowing the model to focus on relative (percentage) errors rather than absolute dollar errors, significantly improving performance across the entire price range.

## 2. Expert Assessment Matters Most

The Feature Importance analysis shows that OverallQual (Overall Quality) is by far the most predictive feature.

Insight: This indicates that a single, expert assessment of the property's condition and finish is more valuable than dozens of raw technical metrics (like heating type or roof material). It suggests that "intangible" quality is the primary driver of value.

## 3. Limitations in the Luxury Segment

The Regression Plot reveals a systematic under-prediction for high-value properties (>$450k).

Observation: The model performs excellently for the mass market but struggles with outliers on the high end.

Reason: There are very few luxury homes in the training data, making it hard for the model to learn what drives extreme value.

Recommendation: For accurate pricing of luxury estates, a separate model or manual appraisal is recommended, as this general-purpose model is too conservative for that segment.