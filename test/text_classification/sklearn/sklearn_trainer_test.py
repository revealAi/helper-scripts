from dags.text_classification.sklearn.trainer import SklearnTrainer

pipeline = {'framework': 'sklearn', 'dataset': '1', 'categories': 'Inland@Wissenschaft',
            'export_path': 'D:\\mlflow\\WA',
            'logging': 'D:\\mlflow\\WA\\logger.log',
            'logging_dir': 'D:\\mlflow\\WA\\model_log',
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


