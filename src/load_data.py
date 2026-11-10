import logging

import pandas as pd

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = [
    "carat", "cut", "color", "clarity",
    "depth", "table", "price", "x", "y", "z",
]


def load_data(path: str) -> pd.DataFrame:
    """Lit le CSV et vérifie que les colonnes attendues sont présentes."""
    df = pd.read_csv(path)

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans {path} : {missing}")

    logger.info("Données chargées depuis %s : %d lignes, %d colonnes",
                path, df.shape[0], df.shape[1])
    return df[EXPECTED_COLUMNS].copy()