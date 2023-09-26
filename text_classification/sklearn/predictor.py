# Copyright (C) 2021 RevealAI
#
# SPDX-License-Identifier: MIT

import os

from common import Predictor
from common.utils.data_util import load_pickle


class SklearnPredictor(Predictor):
    vectorizer = None

    def __init__(self):
        self.model = None

    def load_model(self, model_package_path):
        self.model = load_pickle(os.path.join(model_package_path, 'model.pkl'))
        self.labels = self.model.classes_
        self.vectorizer = load_pickle(os.path.join(model_package_path, 'vectorizer.pkl'))

    def predict_single(self, text):
        texts = self.vectorizer.transform([text])
        return self.model.predict(texts)

    def predict_list(self, list_text):
        texts = self.vectorizer.transform(list_text)
        return self.model.predict(texts)

    def predict_prob_single(self, text):
        texts = self.vectorizer.transform([text])

        prediction_scores = self.model.predict_proba(texts)
        predictions = dict(zip(self.labels, prediction_scores[0]))
        return predictions

    def predict_prob_list(self, list_text):
        texts = self.vectorizer.transform(list_text)
        compact_results = []
        prediction_list = self.model.predict_proba(texts)

        for prediction in prediction_list:
            dict_predictions = dict(zip(self.labels, prediction))
            compact_results.append(dict_predictions)

        return compact_results
