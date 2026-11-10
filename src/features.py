import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

# Contrat avec la partie entraînement : ces colonnes, et seulement elles
OUTPUT_COLUMNS = ["carat", "cut", "color", "clarity", "volume", "price"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crée la variable volume et garde les colonnes utiles au modèle."""
    df = df.copy()
    df["volume"] = df["x"] * df["y"] * df["z"]
    logger.info("Feature créée : volume = x * y * z")
    return df[OUTPUT_COLUMNS]


def save_processed(df: pd.DataFrame, path: str) -> None:
    """Sauvegarde le dataset prêt pour l'entraînement."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Dataset sauvegardé : %s (%d lignes)", path, len(df))