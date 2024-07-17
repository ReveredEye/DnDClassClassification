import os
import app_predict
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient


def test_model_load():
    os.chdir('./flask_app')
    model = app_predict.get_model_from_registry(1)
    assert type(model) == mlflow.pyfunc.PyFuncModel


def test_predict():
    os.chdir('./flask_app')

    # This character is Cleric 12
    character1 = {
        'HP': 99, 'AC': 20, 'Str': 12, 'Dex': 14, 'Con': 16, 'Int': 8, 
        'Wis': 20, 'Cha': 8, 'level': 12
    }
    # This character is Fighter 12
    character2 = {
        'HP': 112, 'AC': 19, 'Str': 20, 'Dex': 14, 'Con': 16, 'Int': 8,
        'Wis': 10, 'Cha': 8, 'level': 12
    }
    avail_classes = ['Artificer',  'Barbarian', 'Bard', 'Blood', 'Cleric', 'Druid', 'Fighter', 'Monk',
          'Mystic',  'Paladin',  'Ranger', 'Rogue', 'Sorcerer', 'Warlock', 'Wizard']
    predictions = app_predict.predict([character1, character2])
    assert sum([x in avail_classes for x in predictions]) == 2
