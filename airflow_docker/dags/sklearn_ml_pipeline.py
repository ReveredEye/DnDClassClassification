import os, sys
from difflib import SequenceMatcher
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

"""
hyperopt is a library that uses bayesian methods to find the best set of hyper parameters.
    - STATUS_OK -> a signal to tell hyperopt that each run has executed successfully or not.
    - Trails -> an object that keeps track of information in each run.
    - fmin -> a function that aims to minimise the objective/loss function.
    - hp -> a library to contains the search space (e.g. the available values for hyper parameters).
    - tpe -> an algorithm that controls the optimisation logic please see:
        https://hyperopt.github.io/hyperopt/
        https://proceedings.neurips.cc/paper_files/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf
"""
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope

import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from airflow.models import DAG
from airflow.operators.python import PythonOperator

def mlflow_set_tracking():
    import mlflow
    mlflow.set_tracking_uri('sqlite:///mlflow.db')

# Function to identify the class with the most levels in and use that as the target
def dominantClass(classStr, justClass):
    i = 0
    totalLvl = 0
    if '|' not in classStr:
        if 'Revised Ranger' in classStr:
            return 'Ranger', int(classStr.split(' ')[-1])
        else:
            return classStr.split(' ')[0], int(classStr.split(' ')[-1])
    dClass = ''
    for c_lvl in classStr.split('|'):
        if 'Revised Ranger' in c_lvl:
            c = 'Ranger'
            lvl = c_lvl.split(' ')[-1]
        else:
            try: 
                c, lvl = c_lvl.split(' ')
            except ValueError as ve:
                c = SequenceMatcher(None, c_lvl, justClass).find_longest_match()
                lvl = c_lvl.split(' ')[-1]
        totalLvl += int(lvl)
        if int(lvl) > i:
            dClass = c
            i = int(lvl)
    return dClass, totalLvl

def data_clean(df):
    # Clean the ingested dataframe

    realTargets = pd.DataFrame({'target': {0: 'Artificer',  1: 'Barbarian',  2: 'Bard',  4: 'Blood',  5: 'Cleric',  7: 'Druid',  8: 'Fighter',  10: 'Monk',
                                                11: 'Mystic',  12: 'Paladin',  13: 'Ranger',  14: 'Rogue',  15: 'Sorcerer',  16: 'Warlock',  17: 'Wizard'},
                                    'counts': {0: 218,  1: 943,  2: 736,  4: 24,  5: 995,  7: 728,  8: 1441,  10: 732,  11: 13,  12: 874,  13: 792,  14: 1292,
                                            15: 664,  16: 677,  17: 760}})
    df['target'], df['level'] = zip(*df[['class', 'justClass']].apply(lambda x: dominantClass(x['class'], x['justClass']), axis = 1))
    features = ['HP', 'AC', 'Str', 'Dex', 'Con', 'Int', 'Wis', 'Cha', 'level']
    target = 'target'
    colList = features + [target]
    dfDropNA = df[colList].dropna(how = 'any')
    cleanDf = dfDropNA.merge(realTargets['target'], how = 'inner', on = target)
    return cleanDf

def dump_pickle(obj, filename):
    with open(filename, 'wb') as f_out:
        return pickle.dump(obj, f_out)

def preprocess_data(**kwargs):
    # kwargs -> {''rs_no': int, 'data_path': str}

    import pandas as pd
    from difflib import SequenceMatcher
    import pickle
    from sklearn.model_selection import train_test_split


    df = data_clean(pd.read_csv("https://raw.githubusercontent.com/oganm/dnddata/master/data-raw/dnd_chars_all.tsv", sep = '\t'))

    features = ['HP', 'AC', 'Str', 'Dex', 'Con', 'Int', 'Wis', 'Cha', 'level']
    target = 'target'
    # For reproducibility, The rs_no is the random state variable so that you can split the data into train, val, test datasets.
    X_trainVal, X_test, y_trainVal, y_test = train_test_split(df.loc[:, features], df.loc[:, target], test_size = 0.3, random_state = kwargs['rs_no'])
    X_train, X_val, y_train, y_val = train_test_split(X_trainVal, y_trainVal, test_size = 0.25, random_state = kwargs['rs_no'] + 3)

    dump_pickle((X_train, y_train), os.path.join(kwargs['data_path'], 'train.pkl'))
    dump_pickle((X_val, y_val), os.path.join(kwargs['data_path'], 'val.pkl'))
    dump_pickle((X_test, y_test), os.path.join(kwargs['data_path'], 'test.pkl'))

def load_pickle(filename):
    with open(filename, 'rb') as f_in:
        return pickle.load(f_in)
    
def hyperOptExperiment(**kwargs):
    # kwargs -> {'data_path': str, 'num_trials': int}

    import pandas as pd
    import pickle
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
    from hyperopt.pyll import scope
    import mlflow

    mlflow.set_experiment('HPO-DnD-classification')

    X_train, y_train = load_pickle(os.path.join(kwargs['data_path'], 'train.pkl'))
    X_val, y_val = load_pickle(os.path.join(kwargs['data_path'], 'val.pkl'))

    def objective(params):
        with mlflow.start_run():
            mlflow.set_tag('model', 'RandomForestClassifier')
            mlflow.log_params(params)
            rf = RandomForestClassifier(**params)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_val)
            accScore = accuracy_score(y_val, y_pred)
            mlflow.log_metric('accuracy_score', accScore)

        return {'loss': 1 - accScore, 'status': STATUS_OK}

    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 1)),
        'n_estimators': scope.int(hp.quniform('n_estimators', 10, 50, 1)),
        'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 4, 1)),
        'random_state': 42
    }

    rstate = np.random.default_rng(42)  # for reproducible results
    fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=kwargs['num_trials'],
        trials=Trials(),
        rstate=rstate
    )

def model_train(data_path, rf_params, params):

    
    X_train, y_train = load_pickle(os.path.join(data_path, 'train.pkl'))
    X_val, y_val = load_pickle(os.path.join(data_path, 'val.pkl'))
    X_test, y_test = load_pickle(os.path.join(data_path, 'test.pkl'))

    with mlflow.start_run():
        for param in rf_params:
            params[param] = int(params[param])
        rf = RandomForestClassifier(**params)
        rf.fit(X_train, y_train)

        # Evaluate model
        val_acc = accuracy_score(y_val, rf.predict(X_val))
        mlflow.log_metric("val_acc", val_acc)
        test_acc = accuracy_score(y_test, rf.predict(X_test))
        mlflow.log_metric("test_acc", test_acc)

def register_model(**kwargs):
    # kwargs -> {'data_path': str}

    import pandas as pd
    import pickle
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    
    import mlflow
    from mlflow.entities import ViewType
    from mlflow.tracking import MlflowClient

    mlflow.set_experiment('DnD-classification')
    mlflow.sklearn.autolog(log_datasets = False)
    
    client = MlflowClient(tracking_uri = "sqlite:///mlflow.db")
    experiment = client.get_experiment_by_name('HPO-DnD-classification')
    runs = client.search_runs(
        experiment_ids = experiment.experiment_id,
        run_view_type = ViewType.ACTIVE_ONLY,
        max_results = 5,
        order_by = ['metrics.accuracy_score DESC'])
    
    rf_Params = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state']
    for run in runs:
        model_train(kwargs['data_path'], rf_Params, run.data.params)

    experiment = client.get_experiment_by_name('DnD-classification')
    best_run = client.search_runs(
        experiment_ids = experiment.experiment_id,
        run_view_type = ViewType.ACTIVE_ONLY,
        max_results = 5,
        order_by = ['metrics.test_acc DESC'])[0]
    
    # Register best model
    best_model_uri = f'runs:/{best_run.info.run_id}/model'
    mlflow.register_model(model_uri = best_model_uri, name = 'best-DnD-RFModel')



if __name__ == '__main__':
        
    # prep_kwargs = {'rs_no': 13,
    #                'data_path': '/workspaces/DnDClassClassification/train_outputs/' or <os.getcwd() + '/processedData/'>}
    # register_kwargs = {'data_path': '/workspaces/DnDClassClassification/train_outputs/'}

    mlflow_set_tracking()

    # Parameter to change training data
    changeTrainData = True
    if changeTrainData:
        preprocess_data(rs_no = 7, 
                        data_path = '/workspaces/DnDClassClassification/train_outputs/' )

    # Change this parameter to find optimal hyper parameters or not (as in use ones found before).
    findHypers = True
    if findHypers:
        hyperOptExperiment(data_path = '/workspaces/DnDClassClassification/train_outputs/' , num_trials = 10)


    register_model(data_path = '/workspaces/DnDClassClassification/train_outputs/' )
    
