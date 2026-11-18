"""Enregistre dans le Model Registry le meilleur run de l'expérience.

Usage :
    python -m src.register_best                 # meilleur MAE (par défaut)
    python -m src.register_best --metric rmse   # ou rmse, mape
    python -m src.register_best --metric r2 --higher-is-better
"""
import argparse
import logging
import os

import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

EXPERIMENT = "diamonds-price"
MODEL_NAME = "diamonds-price-model"
ALIAS = "champion"


def main(metric: str, higher_is_better: bool) -> None:
    load_dotenv()
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = MlflowClient()

    exp = client.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        raise SystemExit(f"Expérience introuvable : {EXPERIMENT}")

    order = "DESC" if higher_is_better else "ASC"
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=[f"metrics.{metric} {order}"],
        max_results=1,
    )
    if not runs:
        raise SystemExit("Aucun run terminé trouvé.")

    best = runs[0]
    best_value = best.data.metrics[metric]
    logger.info("Meilleur run : %s (%s = %s)", best.info.run_name, metric, best_value)

    # Si un champion existe déjà, on ne remplace que s'il est battu
    try:
        champ = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
        champ_value = client.get_run(champ.run_id).data.metrics[metric]
        better = best_value > champ_value if higher_is_better else best_value < champ_value
        if not better:
            logger.info("Champion actuel (v%s, %s = %s) conservé.",
                        champ.version, metric, champ_value)
            return
    except Exception:
        logger.info("Pas de champion existant.")

        # MLflow 3 : le modèle loggé a son propre id (models:/m-xxxx)
    model_outputs = getattr(best.outputs, "model_outputs", None) or []
    if model_outputs:
        model_uri = f"models:/{model_outputs[0].model_id}"
    else:
        logged = mlflow.search_logged_models(
            experiment_ids=[exp.experiment_id],
            filter_string=f"source_run_id = '{best.info.run_id}'",
            output_format="list",
        )
        if not logged:
            raise SystemExit("Aucun modèle loggé trouvé pour ce run.")
        model_uri = f"models:/{logged[0].model_id}"

    mv = mlflow.register_model(model_uri, MODEL_NAME)
    client.set_registered_model_alias(MODEL_NAME, ALIAS, mv.version)
    logger.info("Modèle enregistré : %s v%s, alias '%s'.", MODEL_NAME, mv.version, ALIAS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", default="mae")
    parser.add_argument("--higher-is-better", action="store_true")
    args = parser.parse_args()
    main(args.metric, args.higher_is_better)