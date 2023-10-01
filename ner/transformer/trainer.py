# Copyright (C) 2021 RevealAI
#
# SPDX-License-Identifier: MIT

import json
import logging
import os
from datetime import datetime

import mlflow.transformers
import numpy as np
from datasets import metric, load_metric
from transformers import AutoTokenizer, AutoModelForTokenClassification, \
    TrainingArguments, Trainer, DataCollatorForTokenClassification, TrainerCallback

from common.labeling_client import LabelingGateway
from common.mlflow_util import log_classification_repot
from common.trainer.textflow_trainer import Textflow_Trainer
from .data_util import (
    create_train_test_split, get_conll_trainingdata_cardinality
)


class PrinterCallbackMLFlow(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        logging.info(f"{logs}")


class TransformerNerTrainer(Textflow_Trainer):
    report = None

    def __init__(self, config, from_label_Studio=True):
        self.pretrained_model = None
        self.report = None
        self.config = config
        self.from_label_Studio = from_label_Studio
        self.tokenizer = None
        self.label_all_tokens = True
        self.label_list = []
        self.metric = load_metric("seqeval")
        logging.basicConfig(stream=self.log_stream, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            level=logging.INFO)

    def train(self):

        mlflow.set_experiment(self.config["textflow_project_id"])
        with mlflow.start_run(run_name=self.config["run_name"]) as run:
            try:
                mlflow.set_tag('model_flavor', 'transformer')

                logging.info(f"config:{str(self.config)}")
                mlflow.log_text(self.log_stream.getvalue(), 'logger.log')

                self.pretrained_model = self.config["model"]["pretrained_model"]
                dataset_path = self.config["dataset"]
                split = self.config["model"]["split"]
                batch_size = self.config["model"]["batch_size"]
                epochs = self.config["model"]["epochs"]

                client = LabelingGateway()
                dataset = client.get_NER_unstrucutred_dataset_from_project_CONNL(dataset_path)

                train_dataset, test_dataset, self.label_list, tag2id, id2tag, self.cardinality = get_conll_trainingdata_cardinality(
                    dataset, split, self.config["categories"])


                logging.info("Data Cardinality:" + str(self.cardinality))
                mlflow.log_text(json.dumps(self.config), 'model/training_config.json')
                mlflow.log_text(json.dumps(self.cardinality), 'model/dataset_cardinality.json')

                #train, test = create_train_test_split(dataset, split=split)
                #logging.info(f"Dataset Split( train:{len(train)},test:{len(test)}) {str(datetime.now())}")
                mlflow.log_text(self.log_stream.getvalue(), 'logger.log')

                self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)
                train_tokenized_datasets = train_dataset.map(self.tokenize_and_align_labels, batched=True)
                test_tokenized_datasets = test_dataset.map(self.tokenize_and_align_labels, batched=True)

                model = AutoModelForTokenClassification.from_pretrained(self.pretrained_model,
                                                                        num_labels=len(self.label_list))
                model.id2label = id2tag
                model.label2id = tag2id

                args = TrainingArguments(
                    output_dir="./results",
                    evaluation_strategy="epoch",
                    learning_rate=2e-5,
                    per_device_train_batch_size=batch_size,
                    per_device_eval_batch_size=batch_size,
                    num_train_epochs=epochs,
                    weight_decay=0.01
                )

                data_collator = DataCollatorForTokenClassification(self.tokenizer)

                trainer = Trainer(
                    model,
                    args,
                    train_dataset=train_tokenized_datasets,
                    eval_dataset=test_tokenized_datasets,
                    data_collator=data_collator,
                    tokenizer=self.tokenizer,
                    compute_metrics=self.compute_metrics,
                    callbacks=[PrinterCallbackMLFlow]
                )

                trainer.train()

                mlflow.log_text(self.log_stream.getvalue(), 'logger.log')

                trainer.evaluate()

                predictions, labels, _ = trainer.predict(test_tokenized_datasets)
                predictions = np.argmax(predictions, axis=2)

                # Remove ignored index (special tokens)
                true_predictions = [
                    [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                    for prediction, label in zip(predictions, labels)
                ]
                true_labels = [
                    [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
                    for prediction, label in zip(predictions, labels)
                ]

                self.report = self.metric.compute(predictions=true_predictions, references=true_labels)
                print(self.report)
                logging.info('export model and metrics')
                components = {"model": model, "tokenizer": self.tokenizer}
                mlflow.transformers.log_model(transformers_model=components, artifact_path="model")

                mlflow.log_text(self.log_stream.getvalue(), 'logger.log')
                log_classification_repot(self.report)
                logging.info(self.report)

                logging.info(f"Finished transformer NER model training: {str(datetime.now())}")
                local_path = os.path.join(os.path.dirname(__file__), 'model_artifacts', 'infer.py')
                mlflow.log_artifact(local_path=local_path, artifact_path="model")
                mlflow.log_text(self.log_stream.getvalue(), 'logger.log')

            except Exception as error:
                logging.info(str(error))
                mlflow.log_text(self.log_stream.getvalue(), 'logger.log')
                raise error

    def tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label[word_idx] if self.label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def compute_metrics(self, p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = self.metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
