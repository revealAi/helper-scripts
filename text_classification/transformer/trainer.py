# Copyright (C) 2021 RevealAI
#
# SPDX-License-Identifier: MIT
import json
import logging
from datetime import datetime
from io import StringIO
from tempfile import mkdtemp

import mlflow.transformers
import tensorflow as tf
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, TrainingArguments, AutoModelForSequenceClassification, Trainer

from common.labeling_client import load_tcl_dataset_label_studio, train_test_split, get_data_cardinality
from common.mlflow_util import log_classification_repot
from common.trainer.textflow_trainer import Textflow_Trainer

import numpy as np
import evaluate

from text_classification.sklearn.sklearn_util import classification_report

metric = evaluate.load("accuracy")

class TransformerTextflowTrainer(Textflow_Trainer):
    #bert-base-german-cased
    #bert-base-german-dbmdz-cased
    #bert-base-german-dbmdz-uncased
    #distilbert-base-german-cased
    report = None
    mlflow.set_tracking_uri('http://localhost:5000')

    log_stream = StringIO()

    def __init__(self, config, from_label_Studio=True):
        self.model = None
        self.tokenizer = None
        self.report = None
        self.config = config
        self.from_label_Studio = from_label_Studio
        logging.basicConfig(stream=self.log_stream, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            level=logging.INFO)
        self.output_folder_tmp = mkdtemp()

    def log_metadata(self,labels_all):
        # 3
        self.cardinality = get_data_cardinality(labels_all)
        self.cardinality = {key: value for key, value in self.cardinality.items() if
                            key in self.config["categories"]}

        logging.info(f'Get data cardinality: {json.dumps(self.cardinality)}')
        mlflow.log_text(json.dumps(self.config), 'model/training_config.json')
        mlflow.log_text(json.dumps(self.cardinality), 'model/dataset_cardinality.json')

        mlflow.log_text(self.log_stream.getvalue(), 'logger.log')

    def log_gpu_inf(self):
        logging.info(f'Training Pipeline:{self.config}')
        GPUs = tf.config.list_physical_devices("GPU")
        CPUs = tf.config.list_physical_devices("CPU")
        logging.info(f"Num GPUs:{len(GPUs)}, Num CPUs:{len(CPUs)}")

    def label_map(self, lables):
        map={}
        label_set=list(set(lables))
        id2Label={}

        for i in range(len(label_set)):
            map[label_set[i]]=i
            id2Label[i]=label_set[i]

        mlflow.log_text(json.dumps({'labels': map}), 'model/labels_map.json')

        return map,id2Label

    def train(self):
        try:
            mlflow.set_experiment(self.config["textflow_project_id"])
            with mlflow.start_run(run_name=self.config["run_name"]) as run:
                self.log_gpu_inf()

                logging.info("Loading the data from Label Studio ")
                epochs = self.config["model"]["epochs"]
                batch_size = self.config["model"]["batch_size"]
                split = self.config["model"]["split"]
                max_length = self.config["model"]["max_length"]
                pretrained_model = self.config["model"]["distil_bert"]

                text, labels, target_names = load_tcl_dataset_label_studio(self.config["dataset"], self.config["categories"])
                labels_map,id2Label=self.label_map(labels)

                logging.info('Split dataset into training and evlaution subsets')
                train_texts, val_texts, train_labels, val_labels = train_test_split(text, labels, split)
                self.log_metadata(labels)
                train_labels= [labels_map[x] for x in train_labels]
                val_labels = [labels_map[x] for x in val_labels]

                model=AutoModelForSequenceClassification.from_pretrained(pretrained_model, num_labels=len(target_names))
                model.config.id2label=id2Label
                model.config.label2id=labels_map
                tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

                train_inputs = tokenizer(train_texts, padding=True,max_length=max_length, truncation=True, return_tensors="pt")
                train_inputs["label"] = torch.tensor(train_labels)
                input_dataset = Dataset.from_dict(train_inputs)

                validation_input=tokenizer(val_texts, padding=True, truncation=True,max_length=max_length, return_tensors="pt")
                validation_input["labels"] = torch.tensor(val_labels)
                validation_dataset = Dataset.from_dict(validation_input)

                logging.info('Start model training')
                mlflow.log_text(self.log_stream.getvalue(), 'logger.log')

                training_args = TrainingArguments(
                    output_dir=self.output_folder_tmp,# output directory
                    evaluation_strategy="epoch",
                    num_train_epochs=epochs,  # total number of training epochs
                    per_device_train_batch_size=batch_size,  # batch size per device during training
                    warmup_steps=50,  # number of warmup steps for learning rate scheduler
                    weight_decay=0.01,  # strength of weight decay
                )

                # Create the Trainer and train
                trainer = Trainer(
                    model=model,  # the instantiated 🤗 Transformers model to be trained
                    args=training_args,  # training arguments, defined above
                    train_dataset=input_dataset,  # training dataset
                    eval_dataset=validation_dataset,
                    compute_metrics=self.compute_metrics,
                )

                # Train the model
                trainer.train()
                logging.info('Model training is completed, start evaluation')

                # 13 Export the model and the tolenizer
                logging.info('export model and metrics')
                components = {"model": model, "tokenizer": tokenizer}
                mlflow.transformers.log_model(transformers_model=components, artifact_path="model")


                inputs = tokenizer(val_texts, padding=True, truncation=True, max_length=max_length,
                                             return_tensors="pt")

                #eval_dataloader = DataLoader(inputs, batch_size=8)
                predicted_categories=[]
                #for batch in eval_dataloader:
                    #batch = {k: v.to('cpu') for k, v in batch.items()}
                with torch.no_grad():
                    outputs = model(**inputs)

                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)

                predicted_categories= [id2Label[x.item()] for x in predictions]
                val_labels= [id2Label[x] for x in val_labels]
                print(f'predicted_categories:{predicted_categories}')
                print(f'val_labels:{val_labels}')

                self.report = classification_report(val_labels,predicted_categories,  output_dict=True)
                logging.info(json.dumps(self.report))
                log_classification_repot(self.report)

                logging.info(f"Finished Text Classification: {str(datetime.now())}")
                mlflow.log_text(self.log_stream.getvalue(), 'logger.log')

        except Exception as error:
            logging.error(str(error))
            mlflow.log_text(self.log_stream.getvalue(), 'logger.log')
            raise error

    def compute_metrics(self,eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    def tokenize_function(self,examples):
        return self.tokenizer(examples["text"], padding="max_length", truncation=True)


