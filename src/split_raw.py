import os

import pandas as pd
from sklearn.model_selection import train_test_split

RAW = "data/raw/diamonds.csv"
NEW = "data/new/diamonds_new.csv"

if os.path.exists(NEW):
    raise SystemExit("Le split existe déjà : ne relance pas ce script.")

df = pd.read_csv(RAW)
main_df, new_df = train_test_split(df, test_size=0.10, random_state=42)

os.makedirs(os.path.dirname(NEW), exist_ok=True)
main_df.to_csv(RAW, index=False)
new_df.to_csv(NEW, index=False)
print(len(main_df), len(new_df))