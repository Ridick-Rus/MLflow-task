import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import os

if __name__ == "__main__":
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    experiment_name = "Iris_Logistic_Regression"
    mlflow.set_experiment(experiment_name)

    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    C_param = 0.5
    solver_param = 'liblinear'

    with mlflow.start_run(run_name="Training_Run_LR") as run:
        print(f"MLflow Run ID: {run.info.run_id}")
        print(f"MLflow Experiment ID: {run.info.experiment_id}")
        print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
        print(f"MLflow Artifact URI: {mlflow.get_artifact_uri()}")

        mlflow.log_param("C", C_param)
        mlflow.log_param("solver", solver_param)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("test_size", 0.2)

        model = LogisticRegression(C=C_param, solver=solver_param, random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted')
        recall = recall_score(y_test, predictions, average='weighted')
        f1 = f1_score(y_test, predictions, average='weighted')

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        mlflow.sklearn.log_model(model, "sk_model")
        print("Model logged to MLflow")

        model_uri = f"runs:/{run.info.run_id}/sk_model"
        registered_model_name = "IrisLogisticRegressionModel"
        try:
            mlflow.register_model(model_uri=model_uri, name=registered_model_name)
            print(f"Model registered as '{registered_model_name}'")
        except Exception as e:
            print(f"Could not register model: {e}")

        with open("description.txt", "w") as f:
            f.write("This is a simple logistic regression model for the Iris dataset.\n")
            f.write(f"Trained with C={C_param}, solver={solver_param}.\n")
            f.write(f"Accuracy: {accuracy:.4f}")
        mlflow.log_artifact("description.txt", artifact_path="run_details")
        print("Description artifact logged.")

    print("Training script finished.")
