"""Évaluation d'un modèle de régression : métriques et graphiques."""
import logging
import os

import matplotlib

matplotlib.use("Agg")  # pas de fenêtre, utile en pipeline automatisé
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


def evaluate_model(model, X_test, y_test) -> dict:
    """Retourne {"rmse", "mae", "r2", "mape_pct"} en prix réels (dollars)."""
    pred = model.predict(X_test)
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "mae": float(mean_absolute_error(y_test, pred)),
        "r2": float(r2_score(y_test, pred)),
        "mape_pct": float(np.mean(np.abs((y_test - pred) / y_test)) * 100),
    }
    logger.info("Métriques : %s", metrics)
    return metrics


def plot_pred_vs_actual(model, X_test, y_test, out_dir: str = "models") -> str:
    """Graphique prix prédit vs prix réel. Retourne le chemin du fichier."""
    os.makedirs(out_dir, exist_ok=True)
    pred = model.predict(X_test)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, pred, s=4, alpha=0.3)
    lim = [0, max(float(np.max(y_test)), float(np.max(pred)))]
    plt.plot(lim, lim, "r--", label="prédiction parfaite")
    plt.xlabel("Prix réel")
    plt.ylabel("Prix prédit")
    plt.title("Prédit vs réel")
    plt.legend()

    path = os.path.join(out_dir, "pred_vs_actual.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    return path


def plot_feature_importance(model, feature_names, out_dir: str = "models"):
    """Importance des variables (Random Forest / Gradient Boosting seulement).

    Retourne le chemin du fichier, ou None pour la régression linéaire.
    """
    # si le modèle est enveloppé (TransformedTargetRegressor), on récupère le pipeline
    pipe = getattr(model, "regressor_", model)
    regressor = pipe.named_steps["model"]

    if not hasattr(regressor, "feature_importances_"):
        return None

    os.makedirs(out_dir, exist_ok=True)
    importances = regressor.feature_importances_
    order = np.argsort(importances)

    plt.figure(figsize=(6, 4))
    plt.barh(np.array(feature_names)[order], importances[order])
    plt.xlabel("Importance")
    plt.title("Importance des variables")

    path = os.path.join(out_dir, "feature_importance.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    return path