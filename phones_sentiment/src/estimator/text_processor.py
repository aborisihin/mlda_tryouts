""" text_processor module.
Contains reviews text processing class
"""

import numpy as np

from sklearn.base import TransformerMixin

from pymystem3 import Mystem
from string import punctuation

__all__ = ['LemmatizeTextTransformer', 'TargetMarkupTransformer']


class LemmatizeTextTransformer(TransformerMixin):
    """ Lemmatize text transformer.
    Класс реализует стандартное поведение sklearn transformer.
    Выполняет предобработку текста отзыва: удаление символов пунктуации, приведение в нижний регистр,
    лемматизация слов текста (сохраняя порядок слов).
    """

    def __init__(self):
        pass

    def transform(self, X):
        mystem = Mystem()

        processed_X = []
        for text in X:
            # punctuation
            punctuation_translator = str.maketrans(punctuation, ' ' * len(punctuation))
            processed_text = text.translate(punctuation_translator)
            # lemmas extraction
            processed_text = ' '.join([mystem.lemmatize(token.lower())[0] for token in processed_text.split()])
            processed_X.append(processed_text)

        return processed_X

    def fit(self, X, y=None):
        return self


class TargetMarkupTransformer(TransformerMixin):
    """ Target markup transformer.
    Класс реализует стандартное поведение sklearn transformer.
    Выполняет разметку датасета по целевой переменной. Отзывы с оценкой 3 удаляются из выборки (как
    нейтральные), затем отзывам с оценкой 1 или 2 присваивается негативная метка, а с оценкой 
    4 или 5 - позитивная метка.
    """

    def __init__(self):
        pass

    def transform(self, X):
        X_copy = X.copy()
        X_copy['rank'] = X_copy[X_copy['rank'] != 3]
        X_copy['target'] = (X_copy['rank'] < 3).astype(np.int)
        return X_copy

    def fit(self, X, y=None):
        return self
