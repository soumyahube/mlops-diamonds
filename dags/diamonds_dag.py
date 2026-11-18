"""DAG de test : vérifie que tout fonctionne (env, données, MLflow), puis entraîne.

check_env -> check_data -> check_mlflow -> train -> register_best

Déclenchement manuel (schedule=None) depuis l'interface Airflow.
"""
import os
from datetime import datetime, timedelta

import pandas as pd
import yaml
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

PROJECT_DIR = "/opt/airflow"
PARAMS_PATH = f"{PROJECT_DIR}/params.yaml"


def check_env():
    """Les identifiants DagsHub arrivent bien dans le conteneur ?"""
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        raise RuntimeError("MLFLOW_TRACKING_URI manquante (vérifie docker-compose.yaml)")
    if "mlflow:5000" not in uri:
        raise RuntimeError(f"URI inattendue : {uri} (attendu : http://mlflow:5000)")
    print("OK - MLFLOW_TRACKING_URI =", uri)


def check_data():
    """Le CSV d'entraînement est bien monté dans le conteneur ?"""
    with open(PARAMS_PATH) as f:
        train_path = yaml.safe_load(f)["train"]["train_path"]
    full_path = os.path.join(PROJECT_DIR, train_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"{full_path} introuvable (as-tu fait `dvc pull` sur ta machine ?)")
    df = pd.read_csv(full_path)
    print(f"OK - {full_path} : {len(df)} lignes, colonnes = {list(df.columns)}")


def check_mlflow():
    """Le conteneur Airflow arrive-t-il à joindre le serveur MLflow ?"""
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    exps = MlflowClient().search_experiments()
    print(f"OK - MLflow joignable, expériences : {[e.name for e in exps]}")


def load_runs() -> dict:
    """Lit la section `experiments` de params.yaml -> {nom: overrides}.

    Relu à chaque analyse du DAG par Airflow (environ toutes les 30 s) :
    modifier params.yaml suffit à changer les tâches, sans rebuild.
    """
    with open(PARAMS_PATH) as f:
        cfg = yaml.safe_load(f)
    runs = cfg.get("experiments") or []
    if not runs:
        return {"default": {}}          # un seul run avec la section `train:`
    result = {}
    for run in runs:
        run = dict(run)
        name = run.pop("name")
        result[name] = run
    return result


CONFIGS = load_runs()


def train_config(overrides: dict):
    """Lance src.train avec les paramètres de params.yaml, modifiés par `overrides`."""
    os.chdir(PROJECT_DIR)          # params.yaml et data/ sont en chemins relatifs
    from src.train import main

    main(overrides=overrides)


default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="diamonds_pipeline",
    description="Test de la chaîne Airflow + DVC data + MLflow (DagsHub)",
    default_args=default_args,
    start_date=datetime(2026, 11, 1),
    schedule=None,
    catchup=False,
    tags=["diamonds", "mlops"],
) as dag:

    t_env = PythonOperator(task_id="check_env", python_callable=check_env)
    t_data = PythonOperator(task_id="check_data", python_callable=check_data)
    t_mlflow = PythonOperator(task_id="check_mlflow", python_callable=check_mlflow)

    # Une tâche par configuration : chacune crée un run dans MLflow.
    # Exécutées l'une après l'autre (SQLite n'aime pas l'écriture en parallèle).
    train_tasks = [
        PythonOperator(
            task_id=f"train_{name}",
            python_callable=train_config,
            op_kwargs={"overrides": overrides},
        )
        for name, overrides in CONFIGS.items()
    ]

    t_register = BashOperator(
        task_id="register_best",
        bash_command=f"cd {PROJECT_DIR} && python -m src.register_best",
    )

    t_env >> t_data >> t_mlflow >> train_tasks[0]
    for prev, nxt in zip(train_tasks, train_tasks[1:]):
        prev >> nxt
    train_tasks[-1] >> t_register