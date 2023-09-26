# Copyright (C) 2021 RevealAI
#
# SPDX-License-Identifier: MIT

import logging
import os

from ..utils import data_util as du


class Textflow_Trainer:
    model = None
    config = None
    text = None
    labels = None
    target_names = None
    cardinality = None

    def __init__(self):
        self.model = None

    def export(self):
        pass

    def train(self):
        pass

    def export_predictions(self, y_test, y_pred):
        pred = {}
        pred["real"] = y_test
        pred["prediction"] = y_pred
        # not completed
