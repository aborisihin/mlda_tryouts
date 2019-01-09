""" model_predictor module.
Contains model predictor class
"""

import os

from sklearn.externals import joblib

from estimator.text_processor import LemmatizeTextTransformer

__all__ = ['ModelPredictor']


class ModelPredictor():

    def __init__(self, model_dir):
        self.transformer = LemmatizeTextTransformer()
        print('load vectorizer...')
        self.vectorizer = joblib.load(os.path.join(model_dir, 'vectorizer.pkl'))
        print('load classifier...')
        self.estimator = joblib.load(os.path.join(model_dir, 'classifier.pkl'))
        print('done')

    def predict(self, text):
        try:
            transformed_text = self.transformer.fit_transform([text])
            vectorized_text = self.vectorizer.transform(transformed_text)
            prediction = self.estimator.predict(vectorized_text)[0]
            proba = self.estimator.predict_proba(vectorized_text)[0].max()
        except Exception as e:
            print(str(e))
            return -1

        if proba < 0.6:
            return 2
        elif proba < 0.75:
            return 1 if (prediction == 1) else 3
        else:
            return 0 if (prediction == 1) else 4
