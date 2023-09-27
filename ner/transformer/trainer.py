# Copyright (C) 2021 RevealAI
#
# SPDX-License-Identifier: MIT

import os
import random
from datetime import datetime
import logging

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification

from common.trainer.textflow_trainer import Textflow_Trainer
import spacy
from spacy.util import minibatch, compounding
from spacy.training.example import Example
from spacy.scorer import Scorer
from io import StringIO
from common.mlflow_util import log_classification_repot
import json
import mlflow.spacy

from common.labeling_client import LabelingGateway
from .data_util import (
    get_spacy_trainingdata_cardinality,
    create_train_test_split, get_conll_trainingdata_cardinality
)


class TransformerNerTrainer(Textflow_Trainer):
    report = None

    def __init__(self, config, from_label_Studio=True):
        self.pretrained_model = None
        self.report = None
        self.config = config
        self.from_label_Studio = from_label_Studio

    def train(self):

        mlflow.set_experiment(self.config["textflow_project_id"])
        with mlflow.start_run(run_name=self.config["run_name"]) as run:
            try:
                mlflow.set_tag('model_flavor', 'transformer')
                logging.info(f"config:{str(self.config)}")

                self.pretrained_model = self.config["model"]["pretrained_model"]
                dataset_path = self.config["dataset"]
                split = self.config["model"]["split"]

                client = LabelingGateway()
                dataset = client.get_NER_unstrucutred_dataset_from_project_CONNL(dataset_path)

                dataset,self.cardinality = get_conll_trainingdata_cardinality( dataset, self.config["categories"] )

                logging.info("Data Cardinality:" + str(self.cardinality))
                mlflow.log_text(json.dumps(self.config), 'model/training_config.json')
                mlflow.log_text(json.dumps(self.cardinality), 'model/dataset_cardinality.json')


                train, test = create_train_test_split(dataset, split=split)
                logging.info(f"Dataset Split( train:{len(train)},test:{len(test)}) {str(datetime.now())}" )

                self.train_ner_model(training_data= train,validation_data=test)

                logging.info(f"Finished spacy NER model training: {str(datetime.now())}")
                local_path = os.path.join(os.path.dirname(__file__), 'model_artifacts', 'infer.py')
                mlflow.log_artifact(local_path=local_path, artifact_path="model")

            except Exception as error:
                logging.info(str(error))
                mlflow.log_text(self.log_stream.getvalue(), 'logger.log')
                raise error

    def train_ner_model(
        self,
        training_data,
        validation_data=None,
    ):
        self.pretrained_model = self.config["model"]["pretrained_model"]
        batch_size = self.config["model"]["batch_size"]
        dataset_path = self.config["dataset"]
        split = self.config["model"]["split"]
        epochs = self.config["model"]["epochs"]
        max_length = self.config["model"]["max_length"]
        target_names=self.config['categories'].split('@')
        labels_map, id2Label = self.label_map(target_names)

        model = AutoModelForTokenClassification.from_pretrained(self.pretrained_model, num_labels=len(target_names))
        model.config.id2label = id2Label
        model.config.label2id = labels_map
        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)

        logging.info("start transformer ner model training")

        # TRAINING THE MODEL
        logging.info("start training")
        mlflow.log_text(self.log_stream.getvalue(), 'logger.log')

        mlflow.log_text(self.log_stream.getvalue(), 'logger.log')

        # 13 Export the model and the tolenizer
        logging.info('export model and metrics')
        components = {"model": model, "tokenizer": tokenizer}
        mlflow.transformers.log_model(transformers_model=components, artifact_path="model")

        logging.info("save model to output folder")
        mlflow.log_text(self.log_stream.getvalue(), 'logger.log')
        if validation_data is not None:
            logging.info("validate the model on validation data")
            self.report = {}
            logging.info(json.dumps(self.report))
            log_classification_repot(self.report)

        logging.info("finished training workflow")
        mlflow.log_text(self.log_stream.getvalue(), 'logger.log')

    def evaluate(self, ner_model, examples):
        """
        :param ner_model: the trained spacy model
        :param examples:
        :return: performance scores
        """
        scorer = Scorer()
        example = []
        for input_, annot in examples:
            pred = ner_model(input_)
            temp = Example.from_dict(pred, annot)
            example.append(temp)
        scores = scorer.score(example)
        logging.info(f" Model Evaluation Summary: {scores}")
        return scores
