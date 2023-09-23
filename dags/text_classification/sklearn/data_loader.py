# Copyright (C) 2022 RevealAI
#
# SPDX-License-Identifier: MIT

import collections

from sklearn import model_selection

from dags.labeling_client import LabelingGateway


def load_files_labeling(project_id, categories, encoding="utf-8"):
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
