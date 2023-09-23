from dags.text_classification.sklearn.trainer import SklearnTrainer

pipeline = {'framework': 'sklearn', 'dataset': '1','run_name':'NER_tkl' , 'textflow_project_id':'1992',  'categories': 'Inland@Wissenschaft',
            'metadata': 'Text Classification',
            'vectorizer' : {'use_stopwords': True, 'lowercase': True, 'stemming': False, 'binary': False,
                            'type': 'TFIDF',
                            'analyzer': 'word',
                            'min_df': 0.005, 'max_df': 0.999, 'stop_words': 'german', 'ngram_range': '(1,1)'},
                           'model' : {'type': 'SVM', 'split': 0.8, 'kernel': 'rbf',
                                      'decision_function_shape': 'ovr','max_iter': -1,'degree': 3
                                      }
}

sk_trainer = SklearnTrainer(pipeline)
sk_trainer.train()


