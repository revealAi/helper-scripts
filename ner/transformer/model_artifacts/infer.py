import pickle

import spacy


class NerResult:
    def __init__(self, entity=None,text=None, probability=None):
        self.entity = entity
        self.text = text
        self.probability = probability

class TransformerNerPredictor:

    def __init__(self, model_path):

        self.ner_model = spacy.load(model_path)

    def infer_single(self, text):
        entities = []
        try:
            doc = self.ner_model(text)
            for ent in doc.ents:
                #print(ent)
                entities.append(NerResult(ent.label_,ent.text))
        except:
            return []  # no results in case of exception
        return entities