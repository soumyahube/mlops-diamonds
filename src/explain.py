"""Explicabilité du modèle avec SHAP."""
import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

logger = logging.getLogger(__name__)


def plot_shap_summary(model, X_test, feature_names, out_dir: str = "models",
                      n_samples: int = 1000, random_state: int = 42):
    """Graphique SHAP (importance globale). Retourne le chemin, ou None si non supporté."""
    # même logique que plot_feature_importance : on déballe TransformedTargetRegressor
    pipe = getattr(model, "regressor_", model)
    prep = pipe.named_steps["prep"]
    regressor = pipe.named_steps["model"]

    # TreeExplainer = modèles à arbres seulement (pas la régression linéaire)
    if not hasattr(regressor, "feature_importances_"):
        logger.info("SHAP ignoré : modèle non basé sur des arbres")
        return None

    # échantillon pour garder un temps de calcul raisonnable
    sample = X_test.sample(n=min(n_samples, len(X_test)), random_state=random_state)
    X_transformed = prep.transform(sample)   # SHAP doit voir les données encodées

    explainer = shap.TreeExplainer(regressor)
    shap_values = explainer.shap_values(X_transformed)

    os.makedirs(out_dir, exist_ok=True)
    shap.summary_plot(shap_values, X_transformed,
                      feature_names=feature_names, show=False)
    path = os.path.join(out_dir, "shap_summary.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Graphique SHAP sauvegardé : %s", path)
    return path