import os
import pickle
import pandas as pd
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from flask import Flask, request, jsonify

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

def prepare_features(stats):
    '''
    stats is a serializable object containing the following;
        {'HP': int, 'AC': int, 'Str': int, 'Dex': int, 
        'Con': int, 'Int': int, 'Wis':int, 'Cha':int , 'level': int}
    '''
    return pd.DataFrame([stats])

def predict_endpoint(input):
    '''
    input can either be a single serialized object or a list of serialized object for batch predictions
    '''
    model = get_model_from_registry()
    if type(input) == dict:
        X = prepare_features(input)
    elif type(input) == list:
        X = pd.DataFrame(input)
    preds = model.predict(X)
    if len(preds) == 1:
        return preds[0]
    return preds


