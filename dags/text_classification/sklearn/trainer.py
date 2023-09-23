# Copyright (C) 2021 RevealAI
#
# SPDX-License-Identifier: MIT

import logging
import os
from datetime import datetime
import json

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
    load_files_labeling,
)
from .model_loader import create_model
from .evaluate import classification_report
import mlflow

class SklearnTrainer(Trainer):
    vectorizer = None
    report = None
    mlflow.set_tracking_uri('http://localhost:5000')

    def __init__(self, config, from_label_Studio=True):
        self.model = None
        self.vectorizer = None
        self.report = None
        self.config = config
        self.from_label_Studio = from_label_Studio

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
        try:
            self.create_logger()
            # 1
            logging.info("-------------- Loading the data from directory ")
            text, labels, target_names = load_files_labeling(
                self.config["dataset"], self.config["categories"]
            )
            logging.info(f'-------------- Training pipeline:')
            logging.info(json.dumps(self.config, indent=50))
            print(json.dumps(self.config, indent=50))
            
            # 2
            
            self.cardinality = get_data_cardinality(labels)
            logging.info(f'-------------- Get data cardinality:')
            print(f'-------------- Get data cardinality:')
            logging.info(json.dumps(self.cardinality, indent=50))
            print(json.dumps(self.cardinality, indent=50))
            
            
            # 2-1 create workspace
            self.create_workspace()

            # 3
            logging.info('-------------- Generate tfidf vectorizer')
            print('-------------- Generate tfidf vectorizer')
            self.vectorizer, features = self.fit_vectorizer(text)

            # 4
            logging.info('-------------- get train test split')
            print('-------------- get train test split')
            split = self.config['model']['split']
            X_train, X_test, y_train, y_test = train_test_split(features, labels, split)

            # 5
            logging.info('-------------- train model')
            print('-------------- train model')
            # enable autologging
            mlflow.set_experiment('TKL_2')
            mlflow.sklearn.autolog()
            print(mlflow.get_tracking_uri())
            with mlflow.start_run(run_name = 'testo') as run:

                basic_model = create_model(self.config)
                fitted_model = self.train_with_split(basic_model, X_train, y_train)
                self.model = CalibratedClassifierCV(base_estimator=fitted_model, cv='prefit')
                self.model.fit( X_train, y_train)
                y_pred = self.model.predict(X_test)
                signature = infer_signature(X_test, y_pred)

                # 6 evaluate model
                logging.info('-------------- Generate and export multiclass classification report')
                print('-------------- Generate and export multiclass classification report')
                self.report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
                #mlflow.log_param("alpha", self.report)
                # fetch logged data
                # 7 export model
                # save model information
                self.export()
                # Log the sklearn model and register as version 1
                mlflow.sklearn.log_model(
                    sk_model=self.model,
                    artifact_path="sklearn-model",
                    signature=signature,
                    registered_model_name="Wael",
                )





            logging.info('Finished Text Classification: ' + str(datetime.now()))
            logging.info(json.dumps(self.report, indent=50))
            print('Finished Text Classification: ' + str(datetime.now()))
            print(json.dumps(self.report, indent=50))
            logging.info('..............................................................')

        except Exception as error:
            logging.error(str(error))
            raise error

    def train_with_split(self, model, X_train, y_train):
        """
        @param model:
        @param X_train:
        @param y_train:
        @return:
        """
        logging.debug(
            "||||||||||||||||| start training with model " + model.__class__.__name__
        )
        model.fit(X_train, y_train)
        logging.debug(
            "||||||||||||||||| finish training model  " + model.__class__.__name__
        )
        return model

    def train_with_crossvalidation(self, model, features, labels, CV):
        """
        @param model:
        @param features:
        @param labels:
        @param CV:
        @return:
        """
        accuracies = cross_val_score(model, features, labels, scoring="accuracy", cv=CV)
        return accuracies

    def export(self):
        export_path = self.config["export_path"]
        model_path = os.path.join(export_path, "model.pkl")
        vectorizer_path = os.path.join(export_path, "vectorizer.pkl")
        metrics_path = os.path.join(export_path, "metrics.yaml")
        export_pickle(vectorizer_path, self.vectorizer)
        export_pickle(model_path, self.model)
        export_yaml(self.report, metrics_path)
