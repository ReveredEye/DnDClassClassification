"""
To deploy as web-service:

```bash
docker build -t dnd-class-prediction-service:v1 .
```

```bash
docker run -it --rm -p 9696:9696 dnd-class-prediction-service:v1 .
```
"""

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
    try: 
        model = mlflow.pyfunc.load_model(model_uri = f"models:/{model_name}/{model_version}")
    except OSError as ose:
        metadata_source = meta_data[0].source
        idx = metadata_source.index('/mlruns')
        path = os.getcwd() + metadata_source[idx:] + '/model.pkl'
        with open(path, 'rb') as modelFile:
            model = pickle.load(modelFile)
    return model

def prepare_features(stats):
    '''
    stats is a serializable object containing the following;
        {'HP': int, 'AC': int, 'Str': int, 'Dex': int, 
        'Con': int, 'Int': int, 'Wis':int, 'Cha':int , 'level': int}
    '''
    return pd.DataFrame([stats])

def predict(input):
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

app = Flask('DnD-class-prediction')

@app.route('/predict', methods = ['POST'])
def predict_endpoint():
    character = request.get_json()
    pred = predict(character)

    result = {
        'target': pred
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug = True, host = '0.0.0.0', port = 9696)