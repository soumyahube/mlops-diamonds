import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Valeurs autorisées pour les variables catégorielles
VALID_CATEGORIES = {
    "cut": ["Fair", "Good", "Very Good", "Premium", "Ideal"],
    "color": ["J", "I", "H", "G", "F", "E", "D"],
    "clarity": ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"],
}

DIMENSION_COLUMNS = ["x", "y", "z"]


def clean(df: pd.DataFrame, max_dimension_mm: float = 20.0) -> pd.DataFrame:
    """Nettoie le dataset avec des règles générales (valables pour tout nouveau fichier)."""
    n_start = len(df)
    df = df.copy()

    # 1. Valeurs manquantes
    df = df.dropna()
    logger.info("Valeurs manquantes : %d lignes supprimées", n_start - len(df))

    # 2. Catégories inconnues (l'encodage ordinal échouerait plus tard)
    n_before = len(df)
    for col, valid in VALID_CATEGORIES.items():
        df = df[df[col].isin(valid)]
    logger.info("Catégories inconnues : %d lignes supprimées", n_before - len(df))

    # 3. Dimensions physiquement impossibles (x, y ou z <= 0)
    n_before = len(df)
    df = df[(df[DIMENSION_COLUMNS] > 0).all(axis=1)]
    logger.info("Dimensions <= 0 : %d lignes supprimées", n_before - len(df))

    # 4. Erreurs de décimale : une dimension trop grande est divisée par 10 ;
    #    si le résultat est encore trop grand, la ligne est supprimée.
    n_corrected = 0
    for col in DIMENSION_COLUMNS:
        suspect = df[col] > max_dimension_mm
        n_corrected += int(suspect.sum())
        df.loc[suspect, col] = df.loc[suspect, col] / 10
    n_before = len(df)
    df = df[(df[DIMENSION_COLUMNS] <= max_dimension_mm).all(axis=1)]
    logger.info("Erreurs de décimale corrigées : %d valeurs (%d lignes supprimées ensuite)",
                n_corrected, n_before - len(df))

    logger.info("Nettoyage terminé : %d -> %d lignes", n_start, len(df))
    return df.reset_index(drop=True)