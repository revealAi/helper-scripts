import os
import pickle

import numpy as np
import tensorflow as tf
import yaml
from transformers import AutoTokenizer


class ClassifiactionResult:
    def __init__(self, document_type=None, probability=None):
        self.document_type = document_type
        self.probability = probability

    def __str__(self):
        return f"{self.document_type}: {self.probability}"


class TextClassificationPredictor:
    tokenizer = None
    max_length = 512

    def __init__(self, model_package_path):

        self.model = tf.saved_model.load(os.path.join(model_package_path, 'trained_model'))
        self.labels = self.load_pickle(os.path.join(model_package_path, "labels.pkl"))
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_package_path, 'trained_model'))
        self.configs = self.load_yaml(os.path.join(model_package_path, "config.yaml"))
        self.max_length = self.configs['model']['max_length']

    def infer_single(self, text):
        classifications = []
        try:
            input_ids, input_masks, input_segments = self.tokenize([text], self.tokenizer, self.max_length)
            outputs = self.model([input_ids, input_masks, input_segments], True, None)
            prediction_scores = tf.nn.softmax(outputs, axis=1).numpy()
            predictions = dict(zip(self.labels, prediction_scores[0]))
            for prediction in predictions:
                cl_results = ClassifiactionResult()
                cl_results.document_type = prediction
                cl_results.probability = predictions[prediction]
                classifications.append(cl_results)
        except:
            return []  # no results in case of exception
        return classifications

    def infer_texts(self, list_text):
        classifications = []
        for text in list_text:
            classifications.append(self.infer_single(text))
        return classifications

    def load_pickle(self, file_path):
        """
        @param file_path:
        @return pickle:
        @return:
        """
        return pickle.load(open(file_path, "rb"))

    def load_yaml(self, path):
        """
        @param path:
        @return:
        """
        with open(path) as file:
            config = yaml.full_load(file)
            return config

    def tokenize(self, sentences, tokenizer, max_length):

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

""""""
model_package_path = 'D:\\AutoNLP-Reveal\\autonlp-ls\\workspace\\projects\\1\\trained_models\\17'
tkl = TextClassificationPredictor(model_package_path=model_package_path)

text = '"Ich wäre schon längst in einem Hochsicherheitsgefängnis", sagt der Late-Night-Talker zur Causa. Washington – John Oliver, scharfzüngiger und treffsicherer Late-Night-Talker in den USA, springt Jan Böhmermann bei. Er sei sehr froh, dass Amerika anders als Deutschland kein Gesetz habe, das einen für ein Gedicht hinter Gitter bringe, sagte Oliver. Ich wäre schon längst in einem Hochsicherheitsgefängnis. Erdoğan hat eine unglaublich dünne Haut, sagte Oliver. Er ist schuld, er macht es einem viel zu leicht, sich über ihn lustig zu machen. Oliver zitierte Erdoğans Vergleiche von Israel mit Hitler-Deutschland sowie frauenfeindliche Aussagen des Präsidenten. An die Adresse Erdoğans fügte er hinzu: Wenn Du so ängstlich darauf bedacht bist, nicht verspottet zu werden, versuch doch mal, die freie Rede weder in Deinem Land noch in anderen zu unterdrücken und Dich generell so zu verhalten, dass nicht jeder sehen will, wie man Dir in den Hintern tritt. Böhmermann hatte Ende März in seiner Fernsehshow Neo Magazin Royale ein drastisches Gedicht auf den türkischen Präsidenten verlesen und damit eine größere Affäre ausgelöst.'
res=tkl.infer_single(text)

for item in res:
    print(item.__str__())

