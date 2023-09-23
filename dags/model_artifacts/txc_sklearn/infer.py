import pickle


class ClassifiactionResult:
    def __init__(self, document_type=None, probability=None):
        self.document_type = document_type
        self.probability = probability


class TextClassificationPredictor:

    def __init__(self, model_path, vectorizer_path):

        with open(model_path, "rb") as model_path_file:
                self.model = pickle.load(model_path_file)
        with open(vectorizer_path, "rb") as vectorizer_path_file:
            self.vectorizer = pickle.load(vectorizer_path_file)

    def infer_single(self, text):

        classifications = []
        try:

            texts = self.vectorizer.transform([text])
            predictions = self.model.predict_proba(texts)

            classes=self.model.classes_
            for prediction in predictions:
                prediction_list=list(zip(classes,prediction))

                prediction_list_sorted= sorted(prediction_list, key=lambda pred: pred[1],reverse=True)
                pred_score_max= prediction_list_sorted[0]

                cl_results = ClassifiactionResult()
                cl_results.document_type=pred_score_max[0]
                cl_results.probability = pred_score_max[1]

                classifications.append(cl_results)

        except:
            return []  # no results in case of exception

        return classifications

    def infer_texts(self, texts_list):

        classifications = []
        try:

            texts = self.vectorizer.transform(texts_list)
            predictions = self.model.predict_proba(texts)

            classes = self.model.classes_
            for prediction in predictions:
                prediction_list = list(zip(classes, prediction))

                prediction_list_sorted = sorted(prediction_list, key=lambda pred: pred[1], reverse=True)
                pred_score_max = prediction_list_sorted[0]

                cl_results = ClassifiactionResult()
                cl_results.document_type = pred_score_max[0]
                cl_results.probability = pred_score_max[1]

                classifications.append(cl_results)

        except:
            return []  # no results in case of exception

        return classifications
