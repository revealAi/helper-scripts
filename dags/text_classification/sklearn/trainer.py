# Copyright (C) 2021 RevealAI
#
# SPDX-License-Identifier: MIT

import logging
import os
from datetime import datetime
import json

import logging
from io import StringIO

import nltk
from mlflow.models import infer_signature
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.model_selection import cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from nltk.stem.snowball import SnowballStemmer

from dags.common.trainer.trainer import Trainer
from dags.common.utils.data_util import export_yaml, export_pickle
import mlflow.sklearn


from .data_loader import (
    get_data_cardinality,
    train_test_split,
    load_dataset_label_studio,
)
from .model_loader import create_model
from .evaluate import classification_report
import mlflow

from ...mlflow_util import log_classification_repot

class SklearnTrainer(Trainer):
    vectorizer = None
    report = None
    mlflow.set_tracking_uri('http://localhost:5000')
    mlflow.sklearn.autolog()
    log_stream = StringIO()

    def __init__(self, config, from_label_Studio=True):
        self.model = None
        self.vectorizer = None
        self.report = None
        self.config = config
        self.from_label_Studio = from_label_Studio

        logging.basicConfig(stream=self.log_stream, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            level=logging.INFO)

    def preprocess_text(self, text):
        text = text.lower()
        text = nltk.re.sub(r"\d+", "", text)
        return text

    def fit_vectorizer(self, text):
        """
        @param text:
        @return:
        """
        analyzer = self.config["vectorizer"]["analyzer"]
        min_df = self.config["vectorizer"]["min_df"]
        max_df = self.config["vectorizer"]["max_df"]
        binary = self.config["vectorizer"]["binary"]
        ngram_range = eval(self.config["vectorizer"]["ngram_range"])
        max_features = None
        if (
            "max_features" in self.config["vectorizer"]
            and self.config["vectorizer"]["max_features"] != "None"
        ):
            max_features = self.config["vectorizer"]["max_features"]

        text = map(self.preprocess_text, text)

        if self.config["vectorizer"]["stemming"]:
            stemmer = SnowballStemmer("german")
            text = map(stemmer.stem, text)

        german_stop_words = None

        if self.config["vectorizer"]["use_stopwords"]:
            lang = self.config["vectorizer"]["stop_words"]
            german_stop_words = stopwords.words(lang)

        vectorizer = None

        if self.config["vectorizer"]["type"] == "TFIDF":
            vectorizer = TfidfVectorizer(
                min_df=min_df,
                max_df=max_df,
                analyzer=analyzer,
                stop_words=german_stop_words,
                max_features=max_features,
                ngram_range=ngram_range,
                binary=binary,
                preprocessor=self.preprocess_text,
            )

        if self.config["vectorizer"]["type"] == "COUNT":
            vectorizer = CountVectorizer(
                min_df=min_df,
                max_df=max_df,
                analyzer=analyzer,
                stop_words=german_stop_words,
                max_features=max_features,
                ngram_range=ngram_range,
                binary=binary,
                preprocessor=self.preprocess_text,
            )

        features = vectorizer.fit_transform(text).toarray()
        return vectorizer, features

    def train(self):
        mlflow.set_experiment('1988')
        with mlflow.start_run(run_name='testo') as run:
            try:
                logging.info("Loading the data from directory ")
                text, labels, target_names = load_dataset_label_studio(
                    self.config["dataset"], self.config["categories"]
                )
                logging.info(f'Training pipeline: {json.dumps(self.config)}')

                self.cardinality = get_data_cardinality(labels)
                logging.info(f'Get data cardinality: {json.dumps(self.cardinality)}')

                logging.info('Generate tfidf vectorizer')
                self.vectorizer, features = self.fit_vectorizer(text)

                logging.info('Split dataset into training and evlaution subsets')
                split = self.config['model']['split']
                X_train, X_test, y_train, y_test = train_test_split(features, labels, split)

                logging.info('Start model training')
                mlflow.log_text(json.dumps(self.config), 'model/training_config.json')
                mlflow.log_text(json.dumps(self.cardinality), 'model/dataset_cardinality.json')


                logging.info('Calibrate the trained model with CalibratedClassifierCV')
                basic_model = create_model(self.config)
                fitted_model = basic_model.fit(X_train, y_train)# self.train_with_split(basic_model, X_train, y_train)

                self.model = CalibratedClassifierCV(estimator=fitted_model, cv='prefit')
                self.model.fit( X_train, y_train)

                logging.info('Generate and export multiclass classification report')
                y_pred = self.model.predict(X_test)
                self.report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
                logging.info(json.dumps(self.report ))
                log_classification_repot(self.report)

                logging.info('Finished Text Classification: ' + str(datetime.now()))
                mlflow.log_text(self.log_stream.getvalue(), 'logger.log')
                local_path=os.path.join(os.path.dirname(__file__),'model_artifacts','infer.py')

                mlflow.log_artifact(local_path=local_path,artifact_path="model/")

            except Exception as error:
                logging.error(str(error))
                mlflow.log_text(self.log_stream.getvalue(), 'logger.log')
                print('getcwd:      ', os.getcwd())
                raise error

    def train_with_split(self, model, X_train, y_train):
        model.fit(X_train, y_train)
        return model
