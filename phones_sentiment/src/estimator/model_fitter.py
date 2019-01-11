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
    data_path (str): Путь к файлу с данными
    param_space (dict): Словарь пространства поиска параметров
    max_evals (int): Число итераций hyperopt

    Attributes:
    data_path (str): Путь к файлу с данными
    param_space (dict): Словарь пространства поиска параметров
    max_evals (int): Число итераций hyperopt
    best_params (dict): Подобранное лучшее сочетание параметров модели
    vectorizer (obj): Объект векторизатора из подобранных параметров
    classifier (obj): Объект классификатора из подобранных параметров
    X (np.array): Обучающая выборка, нормализованные тексты отзывов
    y (np.array): Обучающая выборка, целевая переменная
    """

    def __init__(self, data_path, param_space, max_evals):
        self.data_path = data_path
        self.param_space = param_space
        self.max_evals = max_evals
        self.best_params = None
        self.vectorizer = None
        self.classifier = None

    def prepare_data(self):
        """Data preparing
        Запуск процесса загрузки и подготовки данных. Производится разметка данных по целевой 
        переменной и лемматизация текста отзывов.
        """
        data = pd.read_csv(self.data_path)
        data = TargetMarkupTransformer().fit_transform(data)
        self.X = LemmatizeTextTransformer().fit_transform(data['text'])
        self.y = data['target']

    def fit(self):
        """Start model fitting via hyperopt
        Запуск процесса подборов параметров модели через hyperopt
        """
        hyperopt_best_params = fmin(self.hyperopt_target_func, self.param_space,
                                    algo=tpe.suggest, max_evals=self.max_evals)
        self.best_params = space_eval(self.param_space, hyperopt_best_params)
        self.vectorizer = TfidfVectorizer(**self.best_params['tfidf'])
        self.vectorizer.fit(self.X)
        self.classifier = self.best_params['classifier']['object'](**self.best_params['classifier']['params'])
        self.classifier.fit(self.vectorizer.transform(self.X), self.y)

    def hyperopt_target_func(self, args):
        """hyperopt target function
        Целевая функция для ее минимизации через hyperopt. 
        Выбранная метрика качества - обратное значение accuracy, посчитанное через 
        кросс-валидацию по 3 фолдам.

        Returns:
            Значение минимизируемой функции
        """
        vectorizer = TfidfVectorizer(**args['tfidf'])
        classifier = args['classifier']['object'](**args['classifier']['params'])
        return -1.0 * cross_val_score(classifier,
                                      vectorizer.fit_transform(self.X), self.y,
                                      scoring='accuracy', cv=3, n_jobs=-1).mean()

    def save_model(self, model_dir):
        """Model saving
        Сохранение подобранных компонентов модели в файлы
        """
        if self.vectorizer:
            joblib.dump(self.vectorizer, os.path.join(model_dir, 'vectorizer.pkl'))
        if self.classifier:
            joblib.dump(self.classifier, os.path.join(model_dir, 'classifier.pkl'))
