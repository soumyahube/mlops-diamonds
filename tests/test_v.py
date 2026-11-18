import os
from dotenv import load_dotenv
import mlflow
from mlflow.tracking import MlflowClient

load_dotenv()
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
c = MlflowClient()
for v in c.search_model_versions("name='diamonds-price-model'"):
    run = c.get_run(v.run_id)
    print("v" + v.version, v.aliases, run.info.run_name, run.info.run_id[:8],
          round(run.data.metrics.get("mae", -1), 1))