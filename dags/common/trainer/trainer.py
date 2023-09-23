# Copyright (C) 2021 RevealAI
#
# SPDX-License-Identifier: MIT

import logging
import os

from ..utils import data_util as du


class Trainer:
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

    def create_logger(self):
        self.config["export_path"] = self.config["export_path"]
        self.config["logging"] = self.config["logging"]
        self.config["logging_dir"] = self.config["logging_dir"]

        du.create_folder(self.config["export_path"])
        logging.basicConfig(
            format="%(asctime)s %(levelname)-20s %(message)s",
            filename=self.config["logging"],
            level=logging.INFO,
        )

    def create_workspace(self):
        export_path = self.config["export_path"]
        config_path = os.path.join(export_path, "config.yaml")
        properties_path = os.path.join(export_path, "properties.yaml")

        du.export_yaml(self.config, config_path)

        properties = self.cardinality
        # properties["cardinality"] = self.cardinality
        du.export_yaml(properties, properties_path)

    def train(self):
        pass

    def export_predictions(self, y_test, y_pred):
        pred = {}
        pred["real"] = y_test
        pred["prediction"] = y_pred
        # not completed
