import os
from datetime import datetime, timezone
from dotenv import load_dotenv
import mlflow
from mlflow.tracking import MlflowClient

load_dotenv()
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
c = MlflowClient()
exp = c.get_experiment_by_name("diamonds-price")

limite = datetime(2026, 11, 18, tzinfo=timezone.utc).timestamp() * 1000
a_supprimer = [r for r in c.search_runs([exp.experiment_id])
               if r.info.start_time < limite]

for r in a_supprimer:
    print("à supprimer :", r.info.run_name, r.info.run_id[:8])

if input("Supprimer ces runs ? (oui/non) ") == "oui":
    for r in a_supprimer:
        c.delete_run(r.info.run_id)
        print("supprimé", r.info.run_name)