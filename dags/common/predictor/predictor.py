# Copyright (C) 2022 RevealAI
#
# SPDX-License-Identifier: MIT

class Predictor:
    model = None
    labels= None

    def load_model(self,model_package_path):
        pass

    def predict_single(self, text):
        pass

    def predict_list(self, list_text):
        pass
