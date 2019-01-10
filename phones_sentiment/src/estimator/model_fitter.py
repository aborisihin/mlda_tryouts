""" model_fitter module.
Contains model parameters fitter class
"""

import os

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib

from hyperopt import tpe, fmin, space_eval

from estimator.text_processor import TargetMarkupTransformer, LemmatizeTextTransformer

__all__ = ['HyperoptModelFitter']


class HyperoptModelFitter():
    """ Model fitting through hyperopt.
    Класс реализует подбор параметров векторизации текста и параметров модели,
    используя средства библиотеки hyperopt.

    Args:
    model_dir: Путь к директории модели

    Attributes:
    model_dir (str): Путь к каталогу модели
    """

    def __init__(self, data_path, param_space, max_evals):
        self.data_path = data_path
        self.param_space = param_space
        self.max_evals = max_evals
        self.best_params = None
        self.vectorizer = None
        self.classifier = None

    def prepare_data(self):
        data = pd.read_csv(self.data_path)
        data = TargetMarkupTransformer().fit_transform(data)
        self.X = LemmatizeTextTransformer().fit_transform(data['text'])
        self.y = data['target']

    def fit(self):
        hyperopt_best_params = fmin(self.hyperopt_target_func, self.param_space,
                                    algo=tpe.suggest, max_evals=self.max_evals)
        self.best_params = space_eval(self.param_space, hyperopt_best_params)
        self.vectorizer = TfidfVectorizer(**self.best_params['tfidf'])
        self.vectorizer.fit(self.X)
        self.classifier = self.best_params['classifier']['object'](**self.best_params['classifier']['params'])
        self.classifier.fit(self.vectorizer.transform(self.X), self.y)

    def hyperopt_target_func(self, args):
        vectorizer = TfidfVectorizer(**args['tfidf'])
        classifier = args['classifier']['object'](**args['classifier']['params'])
        return -1.0 * cross_val_score(classifier,
                                      vectorizer.fit_transform(self.X), self.y,
                                      scoring='accuracy', cv=3, n_jobs=-1).mean()

    def save_model(self, model_dir):
        if self.vectorizer:
            joblib.dump(self.vectorizer, os.path.join(model_dir, 'vectorizer.pkl'))
        if self.classifier:
            joblib.dump(self.classifier, os.path.join(model_dir, 'classifier.pkl'))
