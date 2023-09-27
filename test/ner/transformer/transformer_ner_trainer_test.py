from ner.transformer.trainer import TransformerNerTrainer

pipeline = {'framework': 'Transformer Ner', 'dataset': '2','run_name':'Transformer_Ner' , 'textflow_project_id':'transformer_1992',  'categories': 'LOC@ORG',
            'metadata': 'Text Classification',
            'model' : {'type': 'spacy', 'split': 0.8, 'batch_size': 16,'epochs':1,
               'max_length': 512,'pretrained_model': 'distilbert-base-german-cased'
                      }
}


sp_trainer = TransformerNerTrainer(pipeline)
sp_trainer.train()


