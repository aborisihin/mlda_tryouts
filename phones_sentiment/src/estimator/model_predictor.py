""" model_predictor module.
Contains model predictor class
"""

from sklearn.externals import joblib

from estimator.text_processor import LemmatizeTextTransformer

__all__ = ['ModelPredictor']


class ModelPredictor():
    """ Model prediction maker.
    Класс реализует вычисление предсказания ранее созданной модели.

    Args:
    vectorizer_path (str): Путь к сохраненному объекту векторайзера
    classifier_path (str): Путь к сохраненному объекту классификатора

    Attributes:
    transformer (obj): Объект предобработчика текста отзыва
    vectorizer (obj): Объект векторайзера текста отзыва
    classifier (obj): Объект классификатора
    """

    def __init__(self, vectorizer_path, classifier_path):
        self.transformer = LemmatizeTextTransformer()
        print('load vectorizer...')
        self.vectorizer = joblib.load(vectorizer_path)
        print('load classifier...')
        self.classifier = joblib.load(classifier_path)
        print('done')

    def predict(self, text):
        """Make prediction
        Вычисление предсказанной метки для текста отзыва

        Returns:
            -1 - для случая невозможности вычислить предсказание
            0 - уверенная негативная метка
            1 - возможная негативная метка
            2 - нейтральная метка
            3 - возможная позитивная метка
            4 - уверенная позитивная метка
        """
        try:
            transformed_text = self.transformer.fit_transform([text])
            vectorized_text = self.vectorizer.transform(transformed_text)
            prediction = self.classifier.predict(vectorized_text)[0]
            proba = self.classifier.predict_proba(vectorized_text)[0].max()
        except Exception as e:
            print(str(e))
            return -1

        if proba < 0.6:
            return 2
        elif proba < 0.75:
            return 1 if (prediction == 1) else 3
        else:
            return 0 if (prediction == 1) else 4
