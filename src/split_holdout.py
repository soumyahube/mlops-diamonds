# src/split_holdout.py  (UNE SEULE FOIS, hors dvc.yaml)
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/processed/diamonds_features.csv")
train_df, holdout_df = train_test_split(df, test_size=0.10, random_state=42)

train_df.to_csv("data/processed/diamonds_train.csv", index=False)
holdout_df.to_csv("data/new/diamonds_holdout.csv", index=False)
print(len(train_df), len(holdout_df))