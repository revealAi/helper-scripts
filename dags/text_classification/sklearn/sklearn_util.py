import inspect

import sklearn
from sklearn.linear_model import RidgeClassifierCV, LinearRegression
from sklearn.utils import all_estimators


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

params={}
for param in LinearRegression._parameter_constraints:
    params['d']=''


lr = LinearRegression(**params)