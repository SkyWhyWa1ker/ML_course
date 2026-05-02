"""
train_model.py
--------------
Trains a RandomForestClassifier on the Iris dataset
and saves the model to disk as model.pkl.

Usage:
    python train_model.py
"""

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

MODEL_PATH = "model.pkl"


def train() -> None:
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"[train] Accuracy : {acc:.4f}")
    print("[train] Classification report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    # Attach class names so batch_predict.py can decode predictions
    clf.class_names_ = iris.target_names

    joblib.dump(clf, MODEL_PATH)
    print(f"[train] Model saved to '{MODEL_PATH}'")


if __name__ == "__main__":
    train()
