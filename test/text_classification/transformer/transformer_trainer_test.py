from text_classification.transformer.trainer import TransformerTextflowTrainer

pipeline = {'framework': 'transformer', 'dataset': '1','run_name':'NER_tkl' , 'textflow_project_id':'trans_1992',  'categories': 'Web@Etat@Wissenschaft',
            'metadata': 'Text Classification',
           'model' : {
               'type': 'Transformer', 'split': 0.8, 'batch_size': 16,'epochs':1,
               'max_length': 512,'distil_bert': 'distilbert-base-german-cased'#'bert-base-german-cased'
                      }
            }

tr_trainer = TransformerTextflowTrainer(pipeline)
tr_trainer.train()


