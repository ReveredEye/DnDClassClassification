import json
import requests

# This character is Sorcerer 6/ Warlock 2/ Fighter 4
character1 = {
    'HP': 90, 
    'AC': 17, 
    'Str': 8, 
    'Dex': 16, 
    'Con': 14, 
    'Int': 8, 
    'Wis': 10, 
    'Cha': 23, 
    'level': 12
}
# This character is Wizard 11/ Cleric 1
character2 = {
    'HP': 75, 
    'AC': 22, 
    'Str': 12, 
    'Dex': 14, 
    'Con': 15, 
    'Int': 18, 
    'Wis': 8, 
    'Cha': 10, 
    'level': 12
}

url = 'http://localhost:9696/predict'
response = requests.post(url, json = character1)
print(response.json())