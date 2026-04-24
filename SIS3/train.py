from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
import mlflow
import mlflow.sklearn

# MLflow setup
mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("iris-classification")

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Hyperparameters
params = {
    "n_estimators": 100,
    "max_depth": None,
    "random_state": 42
}

with mlflow.start_run():
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Log parameters
    mlflow.log_params(params)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)

    # Save model artifact locally
    joblib.dump(model, "model.joblib")

    # Log model to registry
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="IrisClassifier"
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Model registered in MLflow Model Registry as 'IrisClassifier'")

print("Model saved successfully as model.joblib")
