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
    if model_type == 'LogisticRegression':
        return get_LogisticRegression_model(conf['model'])
    if model_type == 'CNN_3F_3D':
        return get_CNN_3F_3D_model(conf['model'])
    if model_type == 'RandomForestClassifier':
        return get_RandomForestClassifier_model(conf['model'])
    if model_type == 'SVM':
        return get_SVM_model(conf['model'])

    raise NameError("not implemented")

def get_LogisticRegression_model(conf):
    penalty = conf['penalty']
    solver = conf['solver']
    max_iter = int(conf['max_iter'])
    model = LogisticRegression(penalty=penalty, solver=solver, max_iter=max_iter)
    return model

def get_CNN_3F_3D_model(conf):
    kernel = conf['kernel']
    decision_function_shape = conf['decision_function_shape']
    max_iter = int(conf['max_iter'])
    degree = int(conf['degree'])

    model = SVC(kernel=kernel, decision_function_shape=decision_function_shape, max_iter=max_iter, degree=degree)
    return model

def get_SVM_model(conf):
    kernel = conf['kernel']
    decision_function_shape = conf['decision_function_shape']
    max_iter = int(conf['max_iter'])
    degree = int(conf['degree'])

    model = SVC(kernel=kernel, decision_function_shape=decision_function_shape, max_iter=max_iter, degree=degree)
    return model

def get_RandomForestClassifier_model(conf):
    n_estimators = int(conf['n_estimators'])
    criterion = conf['criterion']
    max_depth = int(conf['max_depth'])
    min_samples_split = int(conf['min_samples_split'])
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion,
                                   min_samples_split=min_samples_split)
    return model
