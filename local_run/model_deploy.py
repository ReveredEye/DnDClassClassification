import os
import pickle
import pandas as pd
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def get_model_from_registry(version_no = None):
    mlflow.set_tracking_uri('sqlite:///mlflow.db')
    model_name = 'best-DnD-RFModel'
    if version_no is not None:
        model_version = version_no
    else:
        client = MlflowClient(tracking_uri = "sqlite:///mlflow.db")
        # Below will return warning as get_latest_versions is deprecated in mlflow since 2.9.0
        meta_data = client.get_latest_versions(model_name, stages = ["None"])
        model_version = meta_data[0].version
    return mlflow.pyfunc.load_model(model_uri = f"models:/{model_name}/{model_version}")

with open('./processedData/test.pkl', 'rb') as testFile:
    X_test, y_test = pickle.load(testFile)

best_RFModel = get_model_from_registry()
y_pred = best_RFModel.predict(X_test)
print(accuracy_score(y_pred, y_test))