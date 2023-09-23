import inspect

import sklearn
from sklearn.linear_model import RidgeClassifierCV, LinearRegression
from sklearn.utils import all_estimators


def get_supported_classifiers():
    estimators = all_estimators(type_filter='classifier')
    for estimator in estimators:
        print(estimator)


def gt_model_parameters(classifier):
    return classifier._get_param_names()


def gt_model_constraints(classifier):
    return classifier._parameter_constraints


get_supported_classifiers()

# lr = LinearRegression(**params)
