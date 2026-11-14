"""Étape « train » : split -> pipeline sklearn -> entraînement -> run MLflow.

Usage : python -m src.train
"""
import logging
import os

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.evaluate import (
    evaluate_model,
    plot_feature_importance,
    plot_pred_vs_actual,
)
from src.save_model import save_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")

# Ordres du pire au meilleur
CUT_ORDER = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLOR_ORDER = ["J", "I", "H", "G", "F", "E", "D"]
CLARITY_ORDER = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

CATEGORICAL = ["cut", "color", "clarity"]
NUMERICAL = ["carat", "volume"]
FEATURE_NAMES = CATEGORICAL + NUMERICAL   # ordre après ColumnTransformer
TARGET = "price"


def split_data(df: pd.DataFrame, test_size: float, random_state: int):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def build_pipeline(params: dict):
    encoder = ColumnTransformer(
        [("ordinal",
          OrdinalEncoder(categories=[CUT_ORDER, COLOR_ORDER, CLARITY_ORDER]),
          CATEGORICAL)],
        remainder="passthrough",   # carat et volume passent tels quels
    )

    name = params["model"]
    if name == "linear":
        regressor = LinearRegression()
    elif name == "random_forest":
        regressor = RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=params["random_state"],
            n_jobs=-1,
        )
    elif name == "gradient_boosting":
        regressor = GradientBoostingRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            random_state=params["random_state"],
        )
    else:
        raise ValueError(f"Modèle inconnu : {name}")

    pipe = Pipeline([("prep", encoder), ("model", regressor)])

    if params["log_target"]:
        # entraîne sur log(price), prédit en prix réels grâce à expm1
        return TransformedTargetRegressor(
            regressor=pipe, func=np.log1p, inverse_func=np.expm1
        )
    return pipe


def train_model(X_train, y_train, params: dict):
    model = build_pipeline(params)
    model.fit(X_train, y_train)
    return model


def main(params_path: str = "params.yaml") -> None:
    load_dotenv()
    with open(params_path) as f:
        all_params = yaml.safe_load(f)
    params = all_params["train"]

    df = pd.read_csv(params["train_path"])
    logger.info("Données chargées : %d lignes", len(df))
    X_train, X_test, y_train, y_test = split_data(
        df, params["test_size"], params["random_state"]
    )

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("diamonds-price")

    run_name = f"{params['model']}_log{params['log_target']}_depth{params['max_depth']}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_param("n_rows", len(df))

        model = train_model(X_train, y_train, params)
        metrics = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics(metrics)

        # graphiques
        mlflow.log_artifact(plot_pred_vs_actual(model, X_test, y_test))
        importance_path = plot_feature_importance(model, FEATURE_NAMES)
        if importance_path:
            mlflow.log_artifact(importance_path)

            mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        save_model(model, metrics)

        logger.info("Run terminé : %s", metrics)


if __name__ == "__main__":
    main()