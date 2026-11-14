# src/split_features_holdout.py  (UNE SEULE FOIS, hors dvc.yaml)
import os

import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES = "data/processed/diamonds_features.csv"
HOLDOUT = "data/new/diamonds_holdout.csv"

if os.path.exists(HOLDOUT):
    raise SystemExit("Le holdout existe déjà : ne relance pas ce script.")

df = pd.read_csv(FEATURES)
main_df, holdout_df = train_test_split(df, test_size=0.10, random_state=42)

os.makedirs(os.path.dirname(HOLDOUT), exist_ok=True)
holdout_df.to_csv(HOLDOUT, index=False)   # 10 % : intouchable
main_df.to_csv(FEATURES, index=False)     # 90 % : même nom, même chemin

print(len(main_df), len(holdout_df))