# Copyright (C) 2021 RevealAI
#
# SPDX-License-Identifier: MIT

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def create_model(conf):
    """
    :param conf:
    :return:
    """
    model_type = conf['model']['type']
    del conf['model']['type']
    del conf['model']['split']

    if model_type == 'LogisticRegression':
        return get_LogisticRegression_model(conf['model'])
    if model_type == 'RandomForestClassifier':
        return get_RandomForestClassifier_model(conf['model'])
    if model_type == 'SVM':
        return get_SVM_model(conf['model'])

    raise NameError("not implemented")


def get_LogisticRegression_model(conf):
    conf['max_iter'] = int(conf['max_iter'])
    model = LogisticRegression(**conf)
    return model


def get_SVM_model(conf):
    conf['max_iter'] = int(conf['max_iter'])
    conf['degree'] = int(conf['degree'])
    model = SVC(**conf)
    return model


def get_RandomForestClassifier_model(conf):
    conf['n_estimators'] = int(conf['n_estimators'])
    conf['max_depth'] = int(conf['max_depth'])
    conf['min_samples_split'] = int(conf['min_samples_split'])
    model = RandomForestClassifier(**conf)
    return model
