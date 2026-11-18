import os
from dotenv import load_dotenv
import mlflow
from mlflow.tracking import MlflowClient

load_dotenv()
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
c = MlflowClient()
exp = c.get_experiment_by_name("diamonds-price")
for r in c.search_runs([exp.experiment_id]):
    print(r.info.run_name, r.data.params.get("n_rows"),
          round(r.data.metrics.get("mae", -1), 1), r.info.run_id[:8])