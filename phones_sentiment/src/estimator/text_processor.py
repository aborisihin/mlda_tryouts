""" text_processor module.
Contains reviews text processing class
"""

import numpy as np

from sklearn.base import TransformerMixin

from pymystem3 import Mystem
from string import punctuation

__all__ = ['LemmatizeTextTransformer', 'TargetMarkupTransformer']


class LemmatizeTextTransformer(TransformerMixin):

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

    def __init__(self):
        pass

    def transform(self, X):
        X_copy = X.copy()
        #X_copy['rank'] = X_copy[X_copy['rank'] != 3]
        X_copy['target'] = (X_copy['rank'] <= 3).astype(np.int)
        return X_copy

    def fit(self, X, y=None):
        return self
