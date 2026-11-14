"""Sauvegarde du modèle et des métriques sur disque."""
import json
import logging
import os

import joblib

logger = logging.getLogger(__name__)


def save_model(model, metrics: dict, model_dir: str = "models") -> str:
    """Écrit model.pkl et metrics.json dans model_dir. Retourne le chemin du modèle."""
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "model.pkl")
    joblib.dump(model, model_path)

    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Modèle sauvegardé : %s", model_path)
    logger.info("Métriques sauvegardées : %s", metrics_path)
    return model_path