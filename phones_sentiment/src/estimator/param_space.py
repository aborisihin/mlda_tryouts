""" param_space module.
Contains model parameters space for hyperopt
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC

from hyperopt import hp

__all__ = ['parameters_space']


parameters_space = {
	'tfidf': 
	{
		'ngram_range': hp.choice('tfidf_ngram_range', [[1, 2], [1, 3], [2, 3]]),
	},
	'estimator': hp.choice('estimator', [
		{
			'object': LogisticRegression,
			'params': 
			{
				'C': hp.uniform('lr_C', 0.0, 50.0),
				'class_weight': hp.choice('lr_class_weight', ['balanced', None]),
				'solver': hp.choice('lr_solver', ['liblinear', 'lbfgs']),
				'n_jobs': -1
			}
		},
		{
			'object': SGDClassifier,
			'params' : 
			{
				'loss': hp.choice('sgd_loss', ['hinge', 'log']),
				'alpha': hp.uniform('sgd_alpha', 0.0, 50.0),
				'class_weight': hp.choice('sgd_class_weight', ['balanced', None]),
				'max_iter': 1000,
				'n_jobs': -1
			}
		},
		{
			'object': LinearSVC,
			'params': 
			{
				'loss': hp.choice('svc_loss', ['hinge', 'squared_hinge']),
				'C': hp.uniform('svc_C', 0.0, 50.0),
				'class_weight': hp.choice('svc_class_weight', ['balanced', None]),
			}
		}
	])
}