# Copyright (C) 2021 RevealAI
#
# SPDX-License-Identifier: MIT

import os
import re
from collections import Counter
import logging

import spacy
from sklearn.model_selection import train_test_split
import datasets
from datasets import Dataset


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


def get_conll_trainingdata_cardinality(training_data, categories):
    """
    :param spacy docs list
    :return: Counter({'PRODUCT': 11, 'ORG': 7, 'GPE': 1})
    """

    list_labels=[]
    raw_docs = re.split(r'\n\t?\n', training_data)
    token_docs = []
    tag_docs = []
    ids = []
    id = 0
    for doc in raw_docs:
        tokens = []
        tags = []
        ids.append(id)
        id = id + 1
        for line in doc.split('\n'):
            if len(line)< 5:
                continue
            splitted = line.split()
            token, tag = splitted[0], splitted[-1]
            tokens.append(token)

            if tag[2:] not in categories:
                tag='O'
            tags.append(tag)
            list_labels.append(tag)

        token_docs.append(tokens)
        tag_docs.append(tags)

    unique_tags = set(tag for doc in tag_docs for tag in doc)
    tag2id = {tag: id for id, tag in enumerate(unique_tags)}
    id2tag = {id: tag for tag, id in tag2id.items()}
    ner_tags_ids = []

    for doc in tag_docs:
        doc_tags_ids = []
        for tag in doc:
            doc_tags_ids.append(tag2id[tag])
        ner_tags_ids.append(doc_tags_ids)

    dataset_dict = {'id': ids,
                    'tokens': token_docs,
                    'ner_tags': ner_tags_ids
                    }

    dataset_ = Dataset.from_dict(dataset_dict)

    entities_cardinality = Counter(list_labels)
    cardinality = {key: value
                   for key, value in entities_cardinality.items()
                   }

    return dataset_, list(unique_tags), tag2id, id2tag, cardinality


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
