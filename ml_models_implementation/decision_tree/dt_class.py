""" dt_class module.
Contains DecisionTree class.
"""

import numpy as np

from sklearn.base import BaseEstimator

__all__ = ['DecisionTree']


"""
Types, methods and constants
----------------------------
"""


def entropy(y):
    p = [(1.0 * y[y == k].shape[0]) / y.shape[0] for k in np.unique(y)]
    return -np.dot(p, np.log2(p))


def gini(y):
    p = [(1.0 * y[y == k].shape[0]) / y.shape[0] for k in np.unique(y)]
    return 1.0 - np.dot(p, p)


def variance(y):
    return np.var(y) if len(y) > 0 else 0.0


def mad_median(y):
    return np.mean(np.abs(y - np.median(y))) if len(y) > 0 else 0.0


criterion_dict = {'entropy': entropy, 'gini': gini,
                  'variance': variance, 'mad_median': mad_median}

criterion_types_dict = {'entropy': 'classification', 'gini': 'classification',
                        'variance': 'regression', 'mad_median': 'regression'}


def leaf_value_classification(y):
    return np.argmax(np.bincount(y))


def leaf_value_regression(y):
    return np.mean(y)


leaf_value_dict = {'classification': leaf_value_classification,
                   'regression': leaf_value_regression}


def leaf_labels_ratio_classification(y, n_classes):
    return np.bincount(y, minlength=n_classes) / len(y)


def leaf_labels_ratio_regression(y, n_classes):
    return None


leaf_labels_ratio_dict = {'classification': leaf_labels_ratio_classification,
                          'regression': leaf_labels_ratio_regression}


class TreeNode():
    def __init__(self, feature_idx=None, threshold=None,
                 leaf_value=None, leaf_labels_ratio=None,
                 left_child=None, right_child=None):

        self.feature_idx = feature_idx
        self.threshold = threshold
        self.leaf_value = leaf_value
        self.leaf_labels_ratio = leaf_labels_ratio
        self.left_child = left_child
        self.right_child = right_child


"""
DecisionTree class
------------------
"""


class DecisionTree(BaseEstimator):
    """ DecisionTree classifier/regressor
    Класс реализует модель дерева решений, поддерживающего задачи классификации и регрессии.

    Parameters
    ----------
    tags : [string]
        Список допустимых для классификации тегов (классов). Не входящие в этот список теги
        игнорируются.

    Attributes
    ----------
    vocab_ : dict {string: int}
        Mapping слов-признаков в численные индексы. Слова добавляются в словарь в процессе обучения,
        индексы назначаются инкрементально.
        Т.к. обучение модели ведется по онлайн-схеме, то всего признакового пространства мы не знаем.
        В этом случае пользоваться bag-of-words или, например, CountVectorizer из sklearn, не
        целесообразно (словарь придется пересчитывать при каждом появлении нового слова).
    """
    def __init__(self, max_depth=np.inf, min_samples_split=2,
                 criterion='gini', debug=False):

        params = {'max_depth': max_depth,
                  'min_samples_split': min_samples_split,
                  'criterion': criterion,
                  'debug': debug}

        self.set_params(**params)

    def set_params(self, **params):

        super(DecisionTree, self).set_params(**params)

        self._criterion = criterion_dict[self.criterion]
        self._leaf_value = leaf_value_dict[criterion_types_dict[self.criterion]]
        self._leaf_labels_ratio = leaf_labels_ratio_dict[criterion_types_dict[self.criterion]]

        if self.debug:
            print('DecisionTree class set params:')
            print(params)

        return self

    def _functional(self, X, y, feature_idx, threshold):

        division_mask = (X[:, feature_idx] < threshold)
        X_size = X.shape[0]
        l_size = np.sum(division_mask.astype(int))
        r_size = X_size - l_size

        if X_size != 0:
            return (self._criterion(y) -
                    (l_size / X_size) * self._criterion(y[division_mask]) -
                    (r_size / X_size) * self._criterion(y[~division_mask]))
        else:
            return 0.0

    def _build_tree(self, X, y, depth=1):

        if np.unique(y).shape[0] == 1:

            if self.debug:
                l_ratio = self._leaf_labels_ratio(y, self._n_classes)
                print('create leaf (depth={} n_objects={}): value = {} labels_ratio = {}'.format(
                    depth, X.shape[0], round(self._leaf_value(y), 2),
                    np.round(l_ratio, decimals=2) if l_ratio is not None else None))

            return TreeNode(leaf_value=self._leaf_value(y),
                            leaf_labels_ratio=self._leaf_labels_ratio(y, self._n_classes))

        best_functional = 0.0
        best_feature_idx = None
        best_threshold = None

        n_samples, n_features = X.shape

        if (depth < self.max_depth) and (n_samples >= self.min_samples_split):

            for feature_idx in range(n_features):

                if np.unique(X[:, feature_idx]).shape[0] == 1:
                    continue

                thresholds = np.unique(X[:, feature_idx])
                thresholds = thresholds[1:]

                functionals = [self._functional(X, y, feature_idx, thr) for thr in thresholds]

                if np.max(functionals) > best_functional:
                    best_functional = np.max(functionals)
                    best_feature_idx = feature_idx
                    best_threshold = thresholds[np.argmax(functionals)]

        if best_feature_idx is not None:

            if self.debug:
                print('node (depth={} n_objects={}) division: feature_idx = {} threshold = {}'.format(
                    depth, X.shape[0], best_feature_idx, round(best_threshold, 2)))

            best_left_mask = X[:, best_feature_idx] < best_threshold

            return TreeNode(feature_idx=best_feature_idx, threshold=best_threshold,
                            left_child=self._build_tree(X[best_left_mask, :], y[best_left_mask], depth=depth + 1),
                            right_child=self._build_tree(X[~best_left_mask, :], y[~best_left_mask], depth=depth + 1))

        else:

            if self.debug:
                l_ratio = self._leaf_labels_ratio(y, self._n_classes)
                print('create leaf (depth={} n_objects={}): value = {} labels_ratio = {}'.format(
                    depth, X.shape[0], round(self._leaf_value(y), 2),
                    np.round(l_ratio, decimals=2) if l_ratio is not None else None))

            return TreeNode(leaf_value=self._leaf_value(y),
                            leaf_labels_ratio=self._leaf_labels_ratio(y, self._n_classes))

    def fit(self, X, y):

        if criterion_types_dict[self.criterion] == 'classification':
            self._n_classes = np.amax(y) + 1
        else:
            self._n_classes = None

        self._root = self._build_tree(X, y)

        return self

    def _get_object_leaf(self, obj):

        node = self._root

        while node.leaf_value is None:

            if obj[node.feature_idx] < node.threshold:
                node = node.left_child
            else:
                node = node.right_child

        return node

    def _predict_object(self, obj):
        return self._get_object_leaf(obj).leaf_value

    def predict(self, X):
        return np.array([self._predict_object(obj) for obj in X])

    def _predict_object_proba(self, obj):
        return self._get_object_leaf(obj).leaf_labels_ratio

    def predict_proba(self, X):
        return np.array([self._predict_object_proba(obj) for obj in X])
