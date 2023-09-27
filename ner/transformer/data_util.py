# Copyright (C) 2021 RevealAI
#
# SPDX-License-Identifier: MIT

import os
from collections import Counter
import logging

import spacy
from sklearn.model_selection import train_test_split



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
    dataset=[]

    externded_categories=[]
    for cat in categories.split('@'):
        externded_categories.append('B-'+cat)
        externded_categories.append('I-' + cat)

    sentences= training_data.split('\n\n')
    for sentence in sentences:
        sentence_tokens=[]
        sentence_tokens_labels=[]

        for row in sentence.split('\n'):
            if len(row)> 5:
                splitted=row.split()
                label=splitted[-1]
                list_labels.append(label)
                sentence_tokens.append(splitted[0])
                if label in externded_categories:
                    sentence_tokens_labels.append(label)
                else:
                    sentence_tokens_labels.append('O')
        dataset.append({
            'text_tokens':sentence_tokens,
             'labels':sentence_tokens_labels
        })


    entities_cardinality=Counter(list_labels)
    cardinality = { key: value
            for key, value in entities_cardinality.items()
            if key in externded_categories }

    return  dataset, cardinality



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
