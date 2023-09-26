# Copyright (C) 2021 RevealAI
#
# SPDX-License-Identifier: MIT

import os
from collections import Counter
import logging

import spacy
from sklearn.model_selection import train_test_split


def convert_conll_to_spacy(conll_file, ner_model_name="de_core_news_sm"):
    """
    :param data_file:
    When	WRB	O
    Sebastian	NNP	B-PERSON
    Thrun	NNP	I-PERSON
    started	VBD	O
    working	VBG	O
    :return: spacy training data format
    """

    logging.info("||||||||||||||||| convert conll format to spacy training data")

    with open(conll_file, encoding="utf-8", errors="ignore") as f:
        example = f.read()
    docs = spacy.training.converters.conll_ner_to_docs(example, model=ner_model_name)
    training_data = convert_spacy_doc_to_training_data(docs)

    return training_data


def convert_conll_to_spacy_labeling(example, ner_model_name="de_core_news_sm"):
    """
    :param data_file:
    When	WRB	O
    Sebastian	NNP	B-PERSON
    Thrun	NNP	I-PERSON
    started	VBD	O
    working	VBG	O
    :return: spacy training data format
    """

    logging.info("||||||||||||||||| convert conll format to spacy training data")

    example = example.replace("-X- _ ", "")
    # example=example.split('\\')
    docs = spacy.training.converters.conll_ner_to_docs(example, model=ner_model_name)
    training_data = convert_spacy_doc_to_training_data(docs)

    return training_data


def get_spacy_trainingdata_cardinality(training_data, cardinality):
    """
    :param spacy docs list
    :return: Counter({'PRODUCT': 11, 'ORG': 7, 'GPE': 1})
    """
    entities = []
    for doc in training_data:
        for entity in doc[1]["entities"]:
            entities.append(entity[2])
    entities_counter = dict(Counter(entities))
    return {
        k: cardinality.get(k, 0) + entities_counter.get(k, 0)
        for k in set(cardinality) | set(entities_counter)
    }


def create_train_test_split(training_data, split=0.8, random_state=1):
    """
    :param training_data:
    :param split:
    :param random_state:
    :return:
    """
    logging.info("||||||||||||||||| create train test split")
    train, test = train_test_split(
        training_data, train_size=split, random_state=random_state
    )
    return train, test


def convert_spacy_doc_to_training_data(docs):
    """

    :param docs:
    :return:
    """
    logging.info("||||||||||||||||| convert spacy do to training data")
    training_data = []
    for doc in docs:
        # ("Walmart is a leading e-commerce company", {"entities": [(0, 7, "ORG")]})
        entities = []
        for entity in doc.ents:
            entities.append((entity.start_char, entity.end_char, entity.label_))
        training_data.append((doc.text, {"entities": entities}))
    return training_data


def get_immediate_subfiles(path):
    return [f.name for f in os.scandir(path) if f.is_file()]
