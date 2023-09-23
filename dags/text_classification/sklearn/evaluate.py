# Copyright (C) 2022 RevealAI
#
# SPDX-License-Identifier: MIT

from sklearn import metrics

def classification_report(y_test, y_pred, target_names, output_dict=True):
    metric_report = metrics.classification_report(y_test, y_pred, target_names=target_names,
                                                  output_dict=output_dict)
    return metric_report
