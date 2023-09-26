# Copyright (C) 2021 RevealAI
#
# SPDX-License-Identifier: MIT

import os

import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer

# common functions
from common.utils.data_util import load_pickle, load_yaml
from common.predictor.predictor import Predictor

class TransformerPredictor(Predictor):
    tokenizer = None
    max_length= 512

    def tokenize(self,sentences, tokenizer, max_length):

        print("******** max string length:", max_length)
        input_ids, input_masks, input_segments = [], [], []
        for sentence in sentences:
            inputs = tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=max_length,
                                           pad_to_max_length=True,
                                           return_attention_mask=True, return_token_type_ids=True)
            input_ids.append(inputs['input_ids'])
            input_masks.append(inputs['attention_mask'])
            input_segments.append(inputs['token_type_ids'])

        return np.asarray(input_ids, dtype='int32'), np.asarray(input_masks, dtype='int32'), np.asarray(input_segments,
                                                                                                        dtype='int32')

    def __init__(self):
        self.model = None

    def load_model(self, model_package_path):
        self.model = tf.saved_model.load( os.path.join(model_package_path,'trained_model'))
        self.labels = load_pickle(os.path.join(model_package_path,"labels.pkl"))
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_package_path,'trained_model'))

        configs=load_yaml(os.path.join(model_package_path,"config.yaml"))
        self.max_length = configs['model']['max_length']

    def predict_single(self, text):
        input_ids,input_masks,input_segments =self.tokenize([text], self.tokenizer,self.max_length)
        outputs = self.model([input_ids, input_masks, input_segments], True, None)
        final_results = tf.nn.softmax(outputs, axis=1).numpy()
        print(final_results)
        return final_results

    def predict_list(self, list_text):
        input_ids, input_masks, input_segments = self.tokenize(list_text, self.tokenizer, self.max_length)
        outputs = self.model([input_ids, input_masks, input_segments], True, None)
        final_results = tf.nn.softmax(outputs, axis=1).numpy()
        print(final_results)
        return final_results

    def predict_prob_single(self, text):
        input_ids, input_masks, input_segments = self.tokenize([text], self.tokenizer, self.max_length)
        outputs = self.model([input_ids, input_masks, input_segments], True, None)
        prediction_scores = tf.nn.softmax(outputs, axis=1).numpy()

        predictions = dict(zip(self.labels, prediction_scores[0]))
        return predictions

    def predict_prob_list(self, list_text):
        input_ids, input_masks, input_segments = self.tokenize(list_text, self.tokenizer, self.max_length)
        outputs = self.model([input_ids, input_masks, input_segments], True, None)
        prediction_scores = tf.nn.softmax(outputs, axis=1).numpy()
        compact_results=[]
        for prediction in prediction_scores:
                dict_predictions = dict(zip(self.labels, prediction))
                compact_results.append(dict_predictions)

        return compact_results
