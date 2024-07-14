# DnDClassClassification

This project aims to build an end-to-end ML pipeline classifying character classes in Dungeons &amp; Dragons.
As multiclassing exists, the task is simplified to identify the class where the character has the most levels in.


The raw data is borrowed from this link: <https://github.com/oganm/dnddata/tree/master/data-raw>.
You can visit the link to get more details on the data.
Although there are 2 data files, the one I will be using is just the `dnd_chars_all.tsv` as it contains all submissions.
The features I will pass into the model will be 'HP', 'AC', 'Str', 'Dex', 'Con', 'Int', 'Wis', 'Cha', 'level' to predict the predominant
class which will be labelled as 'target'. Note that 'level' column is derived by summing all values within 'class' column from the raw
initial dataset, this is made to avoid model being confused by higher 'HP' values from larger levels.
To initialise environment, just use the base directory (most outer) and
run `pipenv shell` or create virtual environment from `requirements.txt`.


The orchestration tool used is Airflow. Go in the directory `./airflow_docker` and make sure you have created the folders 
`mkdir -p ./config ./plugins` (https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html). 
You may have to initialize the database using `docker compose up airflow-init`.
Run `docker compose up` inside the directory `./airflow_docker` (MAKE SURE YOU ARE IN THE DIRECTORY TO RUN THIS as the docker-compose file is there). 
The username and password will both be `airflow`. The name of the pipeline is `dnd_classification_RFModel`.


In my runs, airflow for some reason gets stuck if zombie job if you just click manually for the pipeline to run. A way around this is to
run them individually in the shell. Use `docker ps -a` to find the `CONTAINER ID` corresponding to the image `airflow_docker-airflow-webserver`,
and after running docker compose up get into its shell using `docker exec -it <container id> /bin/bash`. Once inside the airflow container in docker,
run these commands in order (or alternatively just run `local_model.py` within `./local_run`);

    1 - `airflow tasks test dnd_classification_RFModel preprocess_data`
    2 - `airflow tasks test dnd_classification_RFModel hyperopt_experiment`
    3 - `airflow tasks test dnd_classification_RFModel register_model`

To open mlflow run the command `mlflow ui --backend-store-uri sqlite:///mlflow.db`. Within there, the model to be deployed is `best-DnD-RFModel`.
