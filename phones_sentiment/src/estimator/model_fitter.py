""" model_fitter module.
Contains model parameters fitter class
"""

import os

from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib

from hyperopt import tpe, fmin, space_eval

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

	def fit(self):
		pass

	def get_best_params(self):
		return self.best_params

	def save_model(self, model_dir):
		# os.path.join(model_dir, 'vectorizer.pkl')
		# os.path.join(model_dir, 'classifier.pkl')
		