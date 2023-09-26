from text_classification.transformer.trainer import TransformerTextflowTrainer

pipeline = {'framework': 'sklearn', 'dataset': '1','run_name':'NER_tkl' , 'textflow_project_id':'trans_1992',  'categories': 'Web@Etat@Wissenschaft',
            'metadata': 'Text Classification',
           'model' : {
               'type': 'SVM', 'split': 0.8, 'batch_size': 32,'epochs':1,
               'max_length': 512,'distil_bert': 'bert-base-german-cased'
                      }
            }

tr_trainer = TransformerTextflowTrainer(pipeline)
tr_trainer.train()


