import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from scipy.stats import uniform, randint

# --- Configuration ---
DATA_PATH = 'data/IMDB Dataset.csv'
MODEL_DIR = 'models'
MODEL_PATH = f'{MODEL_DIR}/best_sentiment_model.pkl'
TEST_DATA_PATH = f'{MODEL_DIR}/test_data.pkl'
np.random.seed(42)

def main():
    # 1. Load Data
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    # 2. Prepare Data
    print("Preparing data...")
    # Convert labels to binary (0 for negative, 1 for positive)
    df['sentiment'] = df['sentiment'].map({'negative': 0, 'positive': 1})
    
    X = df['review']
    y = df['sentiment']

    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training data size: {len(X_train)}")
    print(f"Test data size: {len(X_test)}")

    # 4. Create the Pipeline
    # Chain the vectorizer and the classifier together
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', LogisticRegression(solver='saga', max_iter=1000, random_state=42))
    ])

    # 5. Define Parameter Grid for RandomizedSearch
    # Define a distribution for each hyperparameter we want to tune.
    # 'tfidf__ngram_range': Test 1-grams vs 1- and 2-grams
    # 'tfidf__max_features': Test different vocabulary sizes
    # 'clf__C': Test different strengths of regularization
    param_dist = {
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'tfidf__max_features': randint(5000, 20000),
        'clf__C': uniform(0.1, 10)
    }

    # 6. Run RandomizedSearchCV
    # n_iter=10 means it will test 10 random combinations.
    # n_jobs=-1 uses all available CPU cores.
    # cv=3 uses 3-fold cross-validation.
    print("Running RandomizedSearch to find best parameters...")
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=10,
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)

    # 7. Print Best Results
    print("\nSearch complete.")
    print(f"Best cross-validation score: {random_search.best_score_:.4f}")
    print("Best parameters found:")
    print(random_search.best_params_)

    # 8. Save the Best Model
    # random_search.best_estimator_ is the *entire* best Pipeline
    best_model = random_search.best_estimator_
    print(f"\nSaving best model to {MODEL_PATH}...")
    joblib.dump(best_model, MODEL_PATH)

    # 9. Save Test Data for Notebook
    print(f"Saving test data to {TEST_DATA_PATH}...")
    joblib.dump((X_test, y_test), TEST_DATA_PATH)

    print("\nTraining script finished successfully.")

if __name__ == "__main__":
    main()