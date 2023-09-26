import inspect

import sklearn
from sklearn.linear_model import RidgeClassifierCV, LinearRegression
from sklearn.utils import all_estimators

from sklearn import metrics

def classification_report(y_test, y_pred, target_names=None, output_dict=True):
    metric_report = metrics.classification_report(y_test, y_pred, target_names=target_names,
                                                  output_dict=output_dict)
    return metric_report
"""
estimators = all_estimators(type_filter='classifier')
for name, RegressorClass in estimators:
    try:
        print('-', name)
        print(RegressorClass._get_param_names())
        a=RegressorClass._parameter_constraints
        print(a)

        print('*' * 50)
    except Exception as e:
        print(e)

"""