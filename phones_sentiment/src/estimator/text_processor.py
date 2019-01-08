""" text_processor module.
Contains reviews text processing class
"""

from sklearn.base import TransformerMixin

from pymystem3 import Mystem
from string import punctuation


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
