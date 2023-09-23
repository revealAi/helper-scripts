from pprint import pprint

from mlflow import MlflowClient

MLFLOW_TRACKING_URI = 'http://localhost:5000'

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


def fetch_logged_data(run_id):
    data = client.get_run(run_id).data
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in client.list_artifacts(run_id, "model")]
    return data.params, data.metrics, tags, artifacts


def get_all_experiments():
    all_experiments = client.search_experiments()
    return all_experiments


def get_registered_models():
    all_registered_models = client.search_registered_models()
    return all_registered_models


def get_run(run_id):
    return client.get_run(run_id=run_id)


def get_runs_list(experiment_id):
    return client.get_run(run_id=experiment_id)



# https://github.com/mlflow/mlflow/issues/613
