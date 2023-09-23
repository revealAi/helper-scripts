import json
import os
import yaml
import pickle


def create_folder(path, folder_name=None):
    """
    @param path:
    @param folder_name:
    @return:
    """
    try:
        if folder_name is not None:
            dir_path = os.path.join(path, folder_name)
        else:
            dir_path = path
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
        return True
    except Exception as e:  # pylint: disable=bare-except
        raise e
        return False


def export_yaml(data_dict, file_path):

    with open(file_path, "w") as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)


def load_yaml(path):
    """
    @param path:
    @return:
    """
    with open(path) as file:
        config = yaml.full_load(file)
        return config


def load_pickle(file_path):
    """
    @param file_path:
    @return pickle:
    @return:
    """
    return pickle.load(open(file_path, "rb"))


def export_pickle(file_path, model):
    """
    @param file_path:
    @param model:
    @return:
    """
    with open(file_path, "wb") as fin:
        pickle.dump(model, fin)

def export_json(file_path, model):
    """
    @param file_path:
    @param model:
    @return:
    """
    with open(file_path, "wb") as fin:
        json.dump(model, fin)
