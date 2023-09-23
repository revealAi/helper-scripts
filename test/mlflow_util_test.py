from pprint import pprint

import mlflow
from mlflow import MlflowClient

from dags.mlflow_util import fetch_logged_data


run_id = ''
params, metrics, tags, artifacts = fetch_logged_data(run_id)
pprint(params)
pprint(metrics)
pprint(tags)
pprint(artifacts)


