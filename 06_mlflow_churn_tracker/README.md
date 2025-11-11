# ML Experiment Tracker (MLflow + SQLite)

This project demonstrates a professional, robust MLOps workflow for experiment tracking. It trains multiple models to predict Telco Customer Churn and uses **MLflow** to log all experiments to a persistent **SQLite database backend**.

This approach solves the problem of the default `mlruns` filesystem backend, which is not suitable for production use. Using a database ensures all experiment data is stored in a single, queryable file (`mlflow.db`).

## 🚀 Key Features

* **MLflow with Database Backend:** Configured to log all results to `sqlite:///mlflow.db` instead of the default filesystem.
* **Complex Preprocessing:** Uses a full `ColumnTransformer` pipeline (from Project 4) to handle mixed data types (numerical and categorical) in the Telco dataset.
* **Model Comparison:** Trains and logs three different classifiers (`LogisticRegression`, `RandomForest`, `LGBMClassifier`) for easy comparison.
* **Reproducibility:** Logs parameters, metrics (like `f1_score`), and the full model pipeline as artifacts for each run.

## 📁 Project Structure
mlflow_churn_tracker/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── src/
│   ├── __init__.py
│   └── train.py
├── .gitignore
├── requirements.txt
└── README.md

## ⚙️ Setup

1.  **Get the Data:**
    * Download the "Telco Customer Churn" dataset from Kaggle: [link](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
    * Place the `WA_Fn-UseC_-Telco-Customer-Churn.csv` file inside the `data/` folder.

2.  **Create Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # (On Windows: venv\Scripts\activate)
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 💡 How to Use

### Step 1: Run the Experiments


Run the `train.py` script multiple times, once for each model:

# 

'''bash
# Run 1: Logistic Regression
python src/train.py --model_name logistic_regression

# Run 2: Random Forest
python src/train.py --model_name random_forest

# Run 3: LightGBM
python src/train.py --model_name lightgbm
'''

### Step 2: Analyze the Results in MLflow UI
1. Start the MLflow Dashboard: You must tell the mlflow ui command to use your database as its backend:
mlflow ui --backend-store-uri sqlite:///mlflow.db

2. Open in Browser: Open http://127.0.0.1:5000. You will see the MLflow dashboard, now powered by a robust database, ready to compare your runs.


## 📈 Analysis & Conclusion

After running all three models (`logistic_regression`, `random_forest`, `lightgbm`), the results were logged to MLflow and analyzed.

### 1. Identifying the Key Metric

The Telco Churn dataset is **imbalanced** (most customers do not churn). Therefore, `accuracy` is a misleading metric. A model that always predicts "No Churn" would have high accuracy but zero business value.

The primary metrics for this problem are:
* **Precision:** How trustworthy is a "Churn" prediction? (Minimizes cost of false positives).
* **Recall:** How many *actual* churners did we find? (Minimizes cost of false negatives).
* **F1-Score:** The harmonic mean (balance) of Precision and Recall.

### 2. The Clear Winner

Unlike other projects that may present a trade-off, the results here were unambiguous. The `logistic_regression` model was the clear and undisputed winner.

| Model | F1-Score (Best) | Precision (Best) | Recall (Best) |
| :--- | :--- | :--- | :--- |
| **`logistic_regression`** | **0.604** | **0.657** | **0.559** |
| `lightgbm` | 0.570 | 0.621 | 0.527 |
| `random_forest` | 0.535 | 0.612 | 0.476 |

*(Results based on default model parameters)*

### 3. Final Conclusion

The `LogisticRegression` model outperformed both `RandomForest` and `LightGBM` across **all three critical metrics**.

Regardless of the specific business goal (whether to prioritize minimizing costs via Precision or maximizing customer retention via Recall), `LogisticRegression` provides the superior baseline. Therefore, it would be the model selected for further tuning and production use.




