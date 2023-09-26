# Copyright (C) 2021 RevealAI
#
# SPDX-License-Identifier: MIT

import glob

from sklearn import preprocessing
import sklearn.model_selection
from collections import Counter

from common.labeling_client import LabelingGateway


def train_test_split(features, labels, split=0.8, random_state=1988):
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        features, labels, train_size=split, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def get_training_data_tensor_labeling(
    project_id, categories, split=0.8, encoding="utf-8"
):
    client = LabelingGateway()
    text_unfiltered, labels_unfiltered = client.get_TXC_dataset_from_project(project_id)
    text, labels = [], []
    for i, label_unfiltered in enumerate(labels_unfiltered):
        if label_unfiltered in categories:
            text.append(text_unfiltered[i])
            labels.append(label_unfiltered)

    target_names = list(set(labels))

    lb = preprocessing.LabelBinarizer()
    lb.fit(target_names)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        text, labels, split=split
    )
    train_labels = get_sparse_labeling(lb, train_labels)
    val_labels = get_sparse_labeling(lb, val_labels)

    return train_texts, val_texts, train_labels, val_labels, target_names,labels


def get_dataset_cardinality_from_directory(source):
    files = glob.glob(source + "/**/*.txt")
    print(len(files), " files were found")
    dict_class = {}

    for f in files:
        folder = f.split("/")[len(f.split("/")) - 2]
        if folder not in dict_class.keys():
            dict_class[folder] = 1
        else:
            dict_class[folder] = dict_class[folder] + 1
    return dict_class


def get_data_cardinality(labels):
    try:
        counter = Counter(labels)
        return dict(counter)
    except Exception as error:
        raise error


# Create training and validation datasets
def get_sparse(labels, lb, data):
    map_ids = [labels[i] for i in data]
    return lb.transform(map_ids)


def get_sparse_labeling(lb, data):
    map_ids = data
    return lb.transform(map_ids)
