import collections
import os
from os import path as osp

from collections import Counter
from enum import Enum
from label_studio_sdk import Client as LabelingClient
from sklearn import model_selection

from dags.common.utils.data_util import export_yaml

# Labeling server configuration
LABELING_GATEWAY = {
    "SCHEME": os.getenv("LABELING_SCHEME", "http"),
    "HOST": os.getenv("LABELING_HOST", "labeling"),
    "PORT": os.getenv("LABELING_PORT", 80),
    "DEFAULT_TIMEOUT": os.getenv("LABELING_DEFAULT_TIMEOUT", 120),
    "API_KEY": os.getenv("LABELING_API_KEY", ""),
}

AIRFLOW_DATASET_BASEDIR = "/opt/airflow/workspace/datasets"


class ProjectType(Enum):
    TEXT_CLASSIFICATION = 1
    NER_UNSTRUCTURED = 2
    NER_STRUCTURED = 3
    UNDEFINED = 4


def get_tasks_type(tasks):
    for anno in tasks:
        if "choices" in anno["annotations"][0]["result"][0]["value"]:
            return ProjectType.TEXT_CLASSIFICATION
        if "x" in anno["annotations"][0]["result"][0]["value"]:
            return ProjectType.NER_STRUCTURED
        if "labels" in anno["annotations"][0]["result"][0]["value"]:
            return ProjectType.NER_UNSTRUCTURED
        return ProjectType.UNDEFINED


def save_properties(project_id, labels):
    dataset_labels = []
    counter = Counter(labels)
    for element in counter:
        dataset_labels.append({"name": element, "count": counter[element]})

    dataset_labels = sorted(dataset_labels, key=lambda k: k["count"], reverse=True)
    # save to properties file
    properties_path = osp.join(
        osp.join(AIRFLOW_DATASET_BASEDIR, str(project_id)),
        "properties.yaml",
    )
    labels = {label["name"]: label["count"] for label in dataset_labels}
    if not osp.exists(properties_path):
        os.makedirs(osp.dirname(properties_path), exist_ok=True)
    export_yaml(labels, properties_path)


class LabelingGateway:
    def __init__(self, scheme=None, host=None, port=None):
        self.LABELING_GATEWAY = "{}://{}:{}".format(
            scheme or LABELING_GATEWAY["SCHEME"],
            'localhost',
            '1988',
        )

        self.client = LabelingClient(
            url=self.LABELING_GATEWAY, api_key='f7f2492f771c1093e410b6785c46a30181e8044f'
        )

    def get_labeling_projects(self):
        projects = []
        results = self.client.list_projects()
        for item in results:
            project_parameters = item.get_params()
            tasks = item.export_tasks()
            project_type = get_tasks_type(tasks)
            projects.append(
                {
                    "id": project_parameters["id"],
                    "total_annotations_number": project_parameters[
                        "total_annotations_number"
                    ],
                    "title": project_parameters["title"],
                    "created_at": project_parameters["created_at"],
                    "description": project_parameters["description"],
                    "type": project_type,
                }
            )

        return projects

    def update_datasets_properities(self):
        results = self.client.list_projects()
        for project in results:
            tasks = project.export_tasks()
            project_type = get_tasks_type(tasks)
            labels = []
            if project_type == ProjectType.TEXT_CLASSIFICATION:
                for annotations in tasks:
                    if len(annotations["annotations"][0]["result"]) > 0:
                        labels.append(
                            annotations["annotations"][0]["result"][0]["value"][
                                "choices"
                            ][0]
                        )
                save_properties(project.id, labels)
            if project_type == ProjectType.NER_STRUCTURED:
                for anno in tasks:
                    for annotation in anno["annotations"]:
                        for resut in annotation["result"]:
                            if "labels" in resut["value"]:
                                labels.extend(resut["value"]["labels"])
                save_properties(project.id, labels)
            if project_type == ProjectType.NER_UNSTRUCTURED:
                for anno in tasks:
                    for resut in anno["annotations"][0]["result"]:
                        labels.extend(resut["value"]["labels"])
                save_properties(project.id, labels)

    def get_TXC_dataset_from_project(self, project_id):
        results = self.client.get_project(id=project_id)
        tasks = results.export_tasks()
        text, labels = [], []
        for anno in tasks:
            labels.append(anno["annotations"][0]["result"][0]["value"]["choices"][0])
            text.append(anno["data"]["text"])
        return text, labels

    def get_NER_unstrucutred_dataset_from_project_CONNL(
            self, project_id, export_type="CONLL2003"
    ):
        response = self.client.make_request(
            method="GET",
            url=f"/api/projects/{project_id}/export?exportType={export_type}",
        )
        examples = response.text
        return examples

    def get_NER_unstrucutred_dataset_from_project(
            self, project_id, export_type="CONLL2003"
    ):
        results = self.client.get_project(id=project_id)
        tasks = results.export_tasks(export_type=export_type)
        text, labels = [], []
        for anno in tasks:
            for resut in anno["annotations"][0]["result"]:
                if "labels" not in resut["value"].keys():
                    continue
                labels.append(resut["value"])
                text.append(anno["data"]["ner"])
        return text, labels

    def get_NER_strucutred_dataset_from_project(self, project_id):
        results = self.client.get_project(id=project_id)
        tasks = results.export_tasks()

        documents = []
        candidates = []
        for anno in tasks:
            image_data = anno["data"]
            text, labels = [], []
            labels_temp = []
            for resut in anno["annotations"][0]["result"]:
                if "labels" not in resut["value"].keys():
                    candidates.append(resut["value"])
                    continue
                labels_temp.append(resut["value"])

            for label in labels_temp:
                x = label["x"]
                y = label["y"]
                for candidate in candidates:
                    if (
                            "text" in candidate.keys()
                            and x == candidate["x"]
                            and y == candidate["y"]
                    ):
                        text.append(candidate["text"])
                        labels.append(label)
            documents.append({"doc": image_data, "labels": labels, "text": text})

        return documents


def get_TXC_cardinality(labels):
    labels_extended = []
    for label in labels:
        labels_extended.extend(label)

    counter = Counter(labels_extended)
    return dict(counter)


def get_NER_unstrucutred_cardinality(labels):
    labels_extended = []
    for label in labels:
        labels_extended.extend(label["labels"])

    counter = Counter(labels_extended)
    return dict(counter)


def get_NER_strucutred_cardinality(documents):
    labels_extended = []
    for doc in documents:
        for label in doc["labels"]:
            labels_extended.extend(label["labels"])

    counter = Counter(labels_extended)
    return dict(counter)



def load_tcl_dataset_label_studio(project_id, categories, encoding="utf-8"):
    """
    @param directory:
    @param encoding:
    @return:

    """
    try:
        client = LabelingGateway()
        text_unfiltered, labels_unfiltered = client.get_TXC_dataset_from_project(
            project_id
        )

        text, labels = [], []
        for i, label_unfiltered in enumerate(labels_unfiltered):
            if label_unfiltered in categories:
                text.append(text_unfiltered[i])
                labels.append(label_unfiltered)

        target_names = list(set(labels))
        return text, labels, target_names

    except Exception as error:
        raise error

def get_data_cardinality(labels):
    try:
        counter = collections.Counter(labels)
        return dict(counter)
    except Exception as error:
        raise error

def train_test_split(features, labels, train_size=0.8, random_state=1988):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        features,
        labels,
        train_size=train_size,
        shuffle=True,
        random_state=random_state,
        stratify=labels,
    )
    return X_train, X_test, y_train, y_test
