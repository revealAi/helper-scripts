from ner.spacy.trainer import SpacyTrainer
from text_classification.sklearn.trainer import SklearnTextflowTrainer

pipeline = {'framework': 'sklearn', 'dataset': '2','run_name':'NER_SPACY' , 'textflow_project_id':'spacy_1992',  'categories': 'LOC@ORG',
            'metadata': 'Text Classification',
            'model' : {'type': 'spacy', 'split': 0.8, 'iterations': 10,
                        'drop': 0.2,'pretrained_model': 'de_core_news_sm'
                      }
}


sp_trainer = SpacyTrainer(pipeline)
sp_trainer.train()


