
import mlflow
from mlflow import MlflowClient

MLFLOW_TRACKING_URI = 'http://localhost:5000'

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

def fetch_logged_data(run_id):
    data = client.get_run(run_id).data
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in client.list_artifacts(run_id, "model")]
    return data.params, data.metrics, tags, artifacts

def download_artifakts(run_id):
    artifacts = [f.path for f in client.list_artifacts(run_id, "model")]
    return artifacts


def get_all_experiments():
    all_experiments = client.search_experiments()
    return all_experiments


def get_registered_models():
    all_registered_models = client.search_registered_models()
    return all_registered_models


def get_run(run_id):
    return client.get_run(run_id=run_id)


def get_runs_list(experiment_id):
    all_runs= client.search_runs(experiment_ids=[experiment_id])
    return  all_runs


def log_classification_repot(report):
    if isinstance(report, dict):
        for class_or_avg, metrics_dict in report.items():
            if isinstance(metrics_dict, dict):
                for metric, value in metrics_dict.items():
                    mlflow.log_metric('evaluate_' + class_or_avg + '_' + metric, value)
            else:
                mlflow.log_metric('evaluate_' + class_or_avg, metrics_dict)


def register_model():
    return  None
    """
    mlflow.sklearn.log_model(
        sk_model=self.model,
        artifact_path="sklearn-model",
        signature=signature,
        registered_model_name="Wael",
    )
    """ # https://github.com/mlflow/mlflow/issues/613
