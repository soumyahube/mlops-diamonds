import os, mlflow, pandas as pd
from dotenv import load_dotenv
load_dotenv()
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
model = mlflow.sklearn.load_model("models:/diamonds-price-model@champion")
df = pd.read_csv("data/processed/diamonds_features.csv").drop(columns=["price"]).head(5)
print(model.predict(df))