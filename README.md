# DnDClassClassification

This project aims to build an end-to-end ML pipeline classifying character classes in Dungeons &amp; Dragons.
As multiclassing exists, the task is simplified to identify the class where the character has the most levels in.

The raw data is borrowed from this link: <https://github.com/oganm/dnddata/tree/master/data-raw>.
You can visit the link to get more details on the data.
Although there are 2 data files, the one I will be using is just the `dnd_chars_all.tsv` as it contains all submissions.
The features I will pass into the model will be 'HP', 'AC', 'Str', 'Dex', 'Con', 'Int', 'Wis', 'Cha', 'level' to predict the predominant
class which will be labelled as 'target'. Note that 'level' column is derived by summing all values within 'class' column from the raw
initial dataset, this is made to avoid model being confused by higher 'HP' values from larger levels.
Data exploration is done in the eda notebook within `EDA` folder.

The orchestration tool used is Airflow and to use it run `docker compose up` inside the directory `./airflow_docker` (MAKE SURE YOU ARE
IN THE DIRECTORY TO RUN THIS as the docker-compose file is there). The username and password will both be `airflow`. To open mlflow run
the command `mlflow ui --backend-store-uri sqlite:///mlflow.db`
