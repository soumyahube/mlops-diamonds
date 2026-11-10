"""Vérification manuelle de la partie préparation des données.

Usage (depuis la racine du projet) :
    python -m tests.check_prep [chemin_du_csv]
"""
import logging
import sys

from src.clean import clean
from src.features import OUTPUT_COLUMNS, build_features, save_processed
from src.load_data import load_data

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/diamonds.csv"

df = load_data(path)
df = clean(df)
df = build_features(df)

assert list(df.columns) == OUTPUT_COLUMNS, "Colonnes de sortie inattendues"
assert df.isna().sum().sum() == 0, "Il reste des valeurs manquantes"
assert (df["volume"] > 0).all(), "Volume <= 0 détecté"

save_processed(df, "data/processed/diamonds_features.csv")
print(df.head())
print("\nTout est OK :", df.shape)