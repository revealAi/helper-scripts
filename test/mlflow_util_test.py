from pprint import pprint

import mlflow

from dags.mlflow_util import fetch_logged_data, get_all_experiments, get_runs_list, delete_run, get_deleted_experiments, \
    register_model, get_registered_models, download_artifakts

print("""######## download artifacts ########""")
download_artifakts(run_id='399928e498bc45cb8e26d12a687af837',dst_path='D:\\helper-scripts\\test\\download')

print("""######## get all registered models ########""")

all_registerd = get_registered_models()
print(all_registerd)
print(""" ######## register a model #######""")

register_model('399928e498bc45cb8e26d12a687af837')

print(""" ######## get all experiments #######""")
mlflow.set_tracking_uri('http://localhost:5000')
pprint(get_all_experiments())

print(""" ######## get all runs for an experiment ######""")
print(get_runs_list(experiment_id=7))
print('*' * 40)

print(""" ######## get run: params, metrics, tags, artifacts #########""")
run_id = 'f4282f2e14214390b228ff1a1ace442d'
params, metrics, tags, artifacts = fetch_logged_data(run_id)
pprint(f'parameters: {params}')
pprint(f'metrics: {metrics}')
pprint(f'tags: {tags}')
pprint(f'artifacts: {artifacts}')
print('*' * 40)

delete_run(run_id='f4282f2e14214390b228ff1a1ace442d')

print(""" ######## delete an experiment ########""")
# pprint(delete_experiment(experiment_id=7))
# pprint(restore_experiment(experiment_id=7))


print(""" ######## get all deleted experiments ########""")
pprint(get_deleted_experiments())

