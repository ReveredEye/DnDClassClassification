import os
import numpy as np
import pandas as pd
from shortuuid import ShortUUID
from uuid import uuid4
import random
# from model_builder import data_clean, preprocess_data, load_pickle
import model_builder

def test_data_clean():
    df_example = pd.DataFrame(data = [(np.nan, ShortUUID().random(length = 7), uuid4().hex, ShortUUID().random(length = 7), 'Hill Dwarf',
                                'Guild Member - Justice', '2022-08-23T20:02:11Z', 'Sorcerer 13|Cleric 1', 'Sorcerer|Cleric', 'Clockwork Soul|Order Domain', 14, 
                                'Fey Touched|War Caster|Metamagic Adept', 146, 10,  9, 11, 20, 14, 14, 20, np.nan, 'Arcana|Religion|Intimidation', 'Crossbow, light|Dagger', 
                                'Alarm*1|Protection from Evil and Good*1|Command*1|Heroism*1|Aid*2|Lesser Restoration*2|Dispel Magic*3|Protection from Energy*3|Freedom of Movement*4|Summon Construct*4|Greater Restoration*5|Wall of Force*5',
                                'Cha', 'metamagic/Twinned Spell*Subtle Spell*Quickened Spell*Extended Spell', 'Canada', 'CA', np.nan, np.nan, np.nan, 'Dwarf', 
                                'Alarm*1|Protection from Evil and Good*1|Command*1|Heroism*1|Aid*2|Lesser Restoration*2|Dispel Magic*3|Protection from Energy*3|Freedom of Movement*4|*|Greater Restoration*5|Wall of Force*5', 
                                'Crossbow, Light|Dagger', 'thirsty_davinci'),
                                (ShortUUID().random(length = 7), ShortUUID().random(length = 7), uuid4().hex, ShortUUID().random(length = 7), 'Hill Dwarf', 
                                 'Guild Member - Justice', '2022-08-23T19:43:25Z', 'Sorcerer 13|Cleric 1', 'Sorcerer|Cleric', 'Clockwork Soul|Order Domain', 14, np.nan, 133, 10, 
                                 9, 11, 18, 14, 14, 20, np.nan, 'Arcana|Religion|Intimidation', 'Crossbow, light|Dagger', 
                                 'Alarm*1|Protection from Evil and Good*1|Command*1|Heroism*1|Aid*2|Lesser Restoration*2|Dispel Magic*3|Protection from Energy*3|Freedom of Movement*4|Summon Construct*4|Greater Restoration*5|Wall of Force*5',
                                 'Cha', 'metamagic/Twinned Spell*Subtle Spell*Quickened Spell', 'Canada', 'CA', np.nan, np.nan, np.nan, 'Dwarf', 
                                 'Alarm*1|Protection from Evil and Good*1|Command*1|Heroism*1|Aid*2|Lesser Restoration*2|Dispel Magic*3|Protection from Energy*3|Freedom of Movement*4|*|Greater Restoration*5|Wall of Force*5', 
                                 'Crossbow, Light|Dagger', 'thirsty_davinci'),
                                (ShortUUID().random(length = 7), ShortUUID().random(length = 7), uuid4().hex, ShortUUID().random(length = 7), 'Human', 'Noble', '2022-08-22T14:57:09Z', 
                                 'Fighter 13', 'Fighter', 'Eldritch Knight', 13, 'Heavy Armor Master', 140, 21, 20, 12, 19, 14, 11, 10, 'CG', 'History|Insight|Perception|Persuasion', 
                                 'Longsword +1|Dagger|Crossbow, light|Mace Of Disruption ', 
                                 'Prestidigitation*0|Mage Hand*0|Blade Ward*0|Light*0|Alarm*1|Burning Hands*1|Unseen Servant*1|Chromatic Orb*1|Magic Missile*1|Identify*1|Detect Magic*1|Absorb Elements*1|Armor of Agathys*1|Shatter*2|Misty Step*2|Enlarge/Reduce*2|Catnap*3',
                                 'Int', 'fighting style/Defense', 'United States', 'US', 'CG', np.nan, np.nan, 'Human', 
                                 'Prestidigitation*0|Mage Hand*0|Blade Ward*0|Light*0|Alarm*1|Burning Hands*1|Unseen Servant*1|Chromatic Orb*1|Magic Missile*1|Identify*1|Detect Magic*1|Absorb Elements*1|Armor of Agathys*1|Shatter*2|Misty Step*2|Enlarge/Reduce*2|Catnap*3', 
                                 'Longsword|Dagger|Crossbow, Light|Mace', 'cool_bhabha'),
                                 (ShortUUID().random(length = 7), ShortUUID().random(length = 7), uuid4().hex, ShortUUID().random(length = 7), 'Githyanki', 'Charlatan', '2022-06-15T13:57:19Z',
                                  None, None, None, np.nan, None, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 'CG', 'History|Insight|Perception|Persuasion',
                                  None, None, 'Str', None, 'Madagascar', 'MG', None, np.nan, np.nan, 'Githyanki', None,'Longsword|Dagger|Crossbow, Light|Crossbow', 'xxErrorxx123')], 
                            columns = ['ip', 'finger', 'hash', 'name', 'race', 'background', 'date', 'class',
                                        'justClass', 'subclass', 'level', 'feats', 'HP', 'AC', 'Str', 'Dex',
                                        'Con', 'Int', 'Wis', 'Cha', 'alignment', 'skills', 'weapons', 'spells',
                                        'castingStat', 'choices', 'country', 'countryCode',
                                        'processedAlignment', 'good', 'lawful', 'processedRace',
                                        'processedSpells', 'processedWeapons', 'alias'])
    expected_df = pd.DataFrame(data = [(146, 10,  9, 11, 20, 14, 14, 20, 14, 'Sorcerer'),
                                    (133, 10,  9, 11, 18, 14, 14, 20, 14, 'Sorcerer'),
                                    (140, 21, 20, 12, 19, 14, 11, 10, 13, 'Fighter')],
                                columns = ['HP', 'AC', 'Str', 'Dex', 'Con', 'Int', 'Wis', 'Cha', 'level', 'target'])
    df_out = model_builder.data_clean(df_example)
    assert df_out.compare(expected_df).empty

def test_preprocess_data():
    data_path = os.getcwd() + '/flask_app/processedData/'
    model_builder.preprocess_data(rs_no = random.randint(1,100), data_path = data_path)
    test_exists = os.path.isfile(data_path + 'test.pkl')
    train_exists = os.path.isfile(data_path + 'train.pkl')
    val_exists = os.path.isfile(data_path + 'val.pkl')
    assert test_exists and train_exists and val_exists

def test_load_pickle():
    data_path = os.getcwd() + '/flask_app/processedData/'
    X_train, y_train = model_builder.load_pickle(os.path.join(data_path, 'train.pkl'))
    X_val, y_val = model_builder.load_pickle(os.path.join(data_path, 'val.pkl'))
    X_test, y_test = model_builder.load_pickle(os.path.join(data_path, 'test.pkl'))
    expectedCols = ['HP', 'AC', 'Str', 'Dex', 'Con', 'Int', 'Wis', 'Cha', 'level']
    x_train_match = sum(X_train.columns == expectedCols) == 9
    x_val_match = sum(X_val.columns == expectedCols) == 9
    x_test_match = sum(X_test.columns == expectedCols) == 9
    y_train_match = y_train.name == 'target'
    y_val_match = y_val.name == 'target'
    y_test_match = y_test.name == 'target'

    assert x_train_match and x_val_match and x_test_match and y_train_match and y_val_match and y_test_match


    