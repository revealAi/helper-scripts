# Copyright (C) 2021 RevealAI
#
# SPDX-License-Identifier: MIT

import os
import random
from datetime import datetime
import logging
from common.trainer.textflow_trainer import Textflow_Trainer
import spacy
from spacy.util import minibatch, compounding
from spacy.training.example import Example
from spacy.scorer import Scorer
from io import StringIO
from common.utils.data_util import export_yaml
from common.mlflow_util import log_classification_repot
import json
import mlflow.spacy

from common.labeling_client import LabelingGateway
from .data_util import (
    get_spacy_trainingdata_cardinality,
    create_train_test_split,
    convert_conll_to_spacy_labeling,
)


class SpacyTrainer(Textflow_Trainer):
    report = None

    mlflow.set_tracking_uri('http://localhost:5000')
    log_stream = StringIO()


    def __init__(self, config, from_label_Studio=True):
        self.pretrained_model = None
        self.report = None
        self.config = config
        self.from_label_Studio = from_label_Studio

        logging.basicConfig(stream=self.log_stream, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            level=logging.INFO)

    def train(self):

        mlflow.set_experiment(self.config["textflow_project_id"])
        with mlflow.start_run(run_name=self.config["run_name"]) as run:
            try:
                mlflow.set_tag('model_flavor', 'spacy')
                logging.info(f"config:{str(self.config)}")

                self.pretrained_model = self.config["model"]["pretrained_model"]
                iterations = self.config["model"]["iterations"]
                dataset_path = self.config["dataset"]
                split = self.config["model"]["split"]
                drop = self.config["model"]["drop"]

                client = LabelingGateway()
                examples = client.get_NER_unstrucutred_dataset_from_project_CONNL(dataset_path)
                dataset = convert_conll_to_spacy_labeling( examples, self.pretrained_model )

                entities_cardinality = {}
                entities_cardinality = get_spacy_trainingdata_cardinality( dataset, entities_cardinality )

                self.cardinality = {
                    key: value
                    for key, value in entities_cardinality.items()
                    if key in self.config["categories"]
                }
                logging.info("Data Cardinality:" + str(self.cardinality))
                mlflow.log_text(json.dumps(self.config), 'model/training_config.json')
                mlflow.log_text(json.dumps(self.cardinality), 'model/dataset_cardinality.json')

                # set categories list
                if "categories" in self.config.keys():
                    selected_categories = self.config["categories"]
                    for item in dataset:
                        current_entites = item[1]["entities"]
                        filtered_entities = []
                        for ent in current_entites:
                            if ent[2] in selected_categories:
                                filtered_entities.append(ent)
                        item[1]["entities"] = filtered_entities

                train, test = create_train_test_split(dataset, split=split)
                logging.info(f"Dataset Split( train:{len(train)},test:{len(test)}) {str(datetime.now())}" )

                self.train_ner_model(self.pretrained_model,
                    train, iterations=iterations,
                    validation_data=test, drop=drop )

                logging.info(f"Finished spacy NER model training: {str(datetime.now())}")
                local_path = os.path.join(os.path.dirname(__file__), 'model_artifacts', 'infer.py')
                mlflow.log_artifact(local_path=local_path, artifact_path="model")

            except Exception as error:
                logging.info(str(error))
                mlflow.log_text(self.log_stream.getvalue(), 'logger.log')
                raise error

    def train_ner_model(
        self,
        pretrained_model,
        training_data,
        iterations=10,
        validation_data=None,
        drop=0.2
    ):
        """

        :param pretrained_model: en_core_web_sm for English, de_core_news_sm for German
        :param training_data: List of training examples
            Example, ("Walmart is a leading e-commerce company", {"entities": [(0, 7, "ORG"),(14, 20, "MISC")]})
        :param iterations: nu,ber of iteration over the full list
        :param output_folder: folder path to save th trained model
        :return:
        """
        logging.info("start spacy ner model training")

        # Load pre-existing spacy model
        logging.info("load pretrained model")
        nlp = spacy.load(pretrained_model)

        # Getting the pipeline component
        ner = nlp.get_pipe("ner")

        # Adding labels to the `ner`
        logging.info("add entities to training pipeline")
        for _, annotations in training_data:
            for ent in annotations.get("entities"):
                ner.add_label(ent[2])

        # Disable pipeline components you dont need to change
        pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
        unaffected_pipes = [
            pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions
        ]

        # TRAINING THE MODEL
        logging.info("start training")
        mlflow.log_text(self.log_stream.getvalue(), 'logger.log')

        with nlp.disable_pipes(*unaffected_pipes):
            for iteration in range(iterations):
                # shuffling examples  before every iteration
                random.shuffle(training_data)
                losses = {}
                # batch up the examples using spaCy's minibatch
                batches = minibatch(training_data, size=compounding(4.0, 32.0, 1.001))
                for batch in batches:
                    for text, annotations in batch:
                        # create Example
                        doc = nlp.make_doc(text)
                        example = Example.from_dict(doc, annotations)
                        # Update the model
                        nlp.update([example], losses=losses, drop=drop)
                print("Losses", losses)
                logging.info(f"Training loss {losses}")
                mlflow.log_text(self.log_stream.getvalue(), 'logger.log')

        mlflow.spacy.log_model(spacy_model=nlp, artifact_path='model')
        logging.info("save model to output folder")

        mlflow.log_text(self.log_stream.getvalue(), 'logger.log')
        if validation_data is not None:
            logging.info("validate the model on validation data")
            scores = self.evaluate(nlp, validation_data)
            self.report = scores["ents_per_type"]
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
