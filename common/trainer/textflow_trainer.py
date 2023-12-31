# Copyright (C) 2021 RevealAI
#
# SPDX-License-Identifier: MIT

import logging
import os

from ..utils import data_util as du
import mlflow
from io import StringIO

class Textflow_Trainer:
    model = None
    config = None
    text = None
    labels = None
    target_names = None
    cardinality = None

    mlflow.set_tracking_uri('http://localhost:5000')
    log_stream = StringIO()

    def __init__(self):
        self.model = None
        logging.basicConfig(stream=self.log_stream, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            level=logging.INFO)

    def export(self):
        pass

    def train(self):
        pass

    def export_predictions(self, y_test, y_pred):
        pred = {}
        pred["real"] = y_test
        pred["prediction"] = y_pred
        # not completed
