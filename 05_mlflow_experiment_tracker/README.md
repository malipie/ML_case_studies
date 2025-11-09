# ML Experiment Tracker (MLflow Project)
This project demonstrates a professional MLOps workflow for experiment tracking using MLflow.The goal is to train multiple classification models (Logistic Regression, Random Forest, LightGBM) on the Red Wine Quality dataset and use MLflow to log, compare, and identify the best-performing model.This project shows how to manage the machine learning lifecycle, ensuring all experiments are reproducible and easy to analyze.🚀 Key FeaturesMLflow: Implements the core MLflow API (start_run, log_param, log_metric, log_model) to track experiments.Model Comparison: Trains and compares three different classifiers.Reproducibility: Saves parameters, metrics, and model artifacts for each run.Interactive Dashboard: Uses the mlflow ui to visualize and compare results.Flexible Prediction: Includes a prediction script that can either auto-select the best model or use a manually specified model (run_id).📁 

# Project Structure

mlflow_experiment_tracker/
├── data/
│   └── winequality-red.csv
├── mlruns/
│   └── (This directory is created automatically by MLflow)
├── src/
│   ├── __init__.py
│   ├── train.py
│   └── predict.py
├── requirements.txt
└── README.md

# ⚙️ Setup
Get the Data:Download the "Red Wine Quality" dataset from Kaggle: link
Place the winequality-red.csv file inside the data/ folder.Create Virtual Environment:python -m venv venv
source venv/bin/activate  # (On Windows: venv\Scripts\activate)
Install Dependencies:pip install -r requirements.txt


# 💡 How to UseThis project has a 3-step workflow: Experiment, Analyze, and Predict.

# Step 1: Run the ExperimentsRun the train.py script multiple times, once for each model:# Run 1: Logistic Regression
python src/train.py --model_name logistic_regression

# Run 2: Random Forest
python src/train.py --model_name random_forest

# Run 3: LightGBM
python src/train.py --model_name lightgbm
Step 2: Analyze and Choose Your ModelStart the MLflow Dashboard:mlflow ui
Open in Browser:Open http://127.0.0.1:5000. In the UI, compare your models.Make a Business Decision:Analyze the metrics. Notice that different models excel in different areas (e.g., one has high precision, another has high recall).Copy the Run ID:Once you've chosen your model, click on its name in the MLflow UI and copy its Run ID.Step 3: Use the Model for PredictionThe predict.py script supports two strategies, reflecting different business needs.Strategy A: Automatic (Default, Best F1-Score)This is for a balanced use case. It will find the run with the highest f1_score and use it.# Run prediction with default sample data
python src/predict.py


Strategy B: Manual (Business-Driven)This is for a specific business case. You must provide the specific run_id you chose from the MLflow UI.# Example using the --strategy manual flag (replace with YOUR ID)
python src/predict.py --strategy manual --run_id 8f5a2a3b4c1e4f5a8a7b6c3d4e5f6a7b

# You can also pass custom wine data
python src/predict.py --strategy manual --run_id 8f5a2a3b4c1e4f5a8a7b6c3d4e5f6a7b --alcohol 12.5


# 📈 Business Case: 

The Wine WholesalerThis project highlights a critical data science concept: the "best" model depends on the business goal.The Problem: Our dataset is imbalanced (far more 'bad' wines than 'good' wines).The Trade-off: We must choose between:High Precision (e.g., random_forest): Very trustworthy. When it says a wine is "Good", it's almost certainly right. But it will miss many other good wines (low recall).High Recall (e.g., lightgbm): A great "discoverer". It finds most of the "Good" wines. But it will also incorrectly label some "Bad" wines as "Good" (low precision).Our Choice: The Wholesaler ModelFor a wine wholesaler, the cost of missing out on a good wine (a "False Negative") is very high—they lose a potential product. The cost of accidentally buying one mediocre wine (a "False Positive") is low.Therefore, the wholesaler's goal is High Recall.Based on our MLflow results, the lightgbm model is the clear winner for this use case, as it has the highest f1_score (the best balance) and the highest recall (it finds the most good wine). We would manually select its run_id using the --strategy manual flag for this scenario.