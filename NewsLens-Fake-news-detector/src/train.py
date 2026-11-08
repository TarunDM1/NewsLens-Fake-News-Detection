"""Train and compare TF-IDF news classifiers.

Expected files:
    dataset/Fake.csv
    dataset/True.csv

Run from the repository root:
    python -m src.train
"""
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.preprocessing import combine_title_and_body

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"
MODEL_DIR = ROOT / "models"
RANDOM_STATE = 42


def load_dataset() -> pd.DataFrame:
    fake_path = DATA_DIR / "Fake.csv"
    real_path = DATA_DIR / "True.csv"
    if not fake_path.exists() or not real_path.exists():
        raise FileNotFoundError(
            "Dataset files not found. Add dataset/Fake.csv and dataset/True.csv. "
            "See dataset/README.md for guidance."
        )

    fake = pd.read_csv(fake_path)
    real = pd.read_csv(real_path)

    for frame in (fake, real):
        if "text" not in frame.columns:
            raise ValueError("Each dataset CSV must contain a 'text' column.")

    fake["title"] = fake.get("title", "")
    real["title"] = real.get("title", "")
    fake["label"] = 0
    real["label"] = 1

    data = pd.concat([fake, real], ignore_index=True)
    data["combined_text"] = [
        combine_title_and_body(title, body)
        for title, body in zip(data["title"], data["text"])
    ]
    data = data[data["combined_text"].str.len() > 0]
    data = data.drop_duplicates(subset=["combined_text"])
    return data[["combined_text", "label"]]


def build_pipeline(classifier) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=100_000,
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )),
        ("classifier", classifier),
    ])


def main() -> None:
    data = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        data["combined_text"],
        data["label"],
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=data["label"],
    )

    classifiers = {
        "Logistic Regression": LogisticRegression(
            max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
        "Calibrated Linear SVM": CalibratedClassifierCV(
            estimator=LinearSVC(class_weight="balanced", random_state=RANDOM_STATE),
            cv=3,
        ),
    }

    results = {}
    fitted_models = {}

    for name, classifier in classifiers.items():
        print(f"Training {name}...")
        pipeline = build_pipeline(classifier)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        results[name] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
        }
        fitted_models[name] = pipeline
        print(f"  Accuracy: {results[name]['accuracy']:.4f} | F1: {results[name]['f1']:.4f}")

    best_name = max(results, key=lambda name: results[name]["f1"])
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted_models[best_name], MODEL_DIR / "best_model.joblib")

    report = {
        "best_model": best_name,
        "test_size": int(len(X_test)),
        "training_size": int(len(X_train)),
        "class_counts": {
            "fake": int((data["label"] == 0).sum()),
            "real": int((data["label"] == 1).sum()),
        },
        "model_results": results,
        "random_state": RANDOM_STATE,
        "note": "Metrics are dataset-specific and are not a guarantee of real-world fact verification.",
    }
    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print(f"\nBest model by F1-score: {best_name}")
    print(f"Saved model to: {MODEL_DIR / 'best_model.joblib'}")
    print(f"Saved metrics to: {MODEL_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
