from pprint import pprint

import mlflow

from dags.mlflow_util import fetch_logged_data,get_all_experiments,get_runs_list

print('get all experiments:')
mlflow.set_tracking_uri('http://localhost:5000')
pprint(get_all_experiments())

print('get all runs for an experiment:')
print(get_runs_list(experiment_id=6))
print('*'*40)

run_id = '8a0ffedea9ef480aa58d83bfa65686d6'
params, metrics, tags, artifacts = fetch_logged_data(run_id)
pprint(f'parameters: {params}')
pprint(f'metrics: {metrics}')
pprint(f'tags: {tags}')
pprint(f'artifacts: {artifacts}')


