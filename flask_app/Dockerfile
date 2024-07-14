FROM python:3.10-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY ["requirements.txt", "./"]

RUN pipenv install --system --deploy -r requirements.txt

COPY ["app_predict.py", "mlflow.db", "./"]

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "app_predict:app" ]
