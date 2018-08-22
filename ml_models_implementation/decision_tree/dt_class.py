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

""" Критерии разбиения
"""


# Энтропия
def _entropy(y):
    p = [(1.0 * y[y == k].shape[0]) / y.shape[0] for k in np.unique(y)]
    return -np.dot(p, np.log2(p))


# Неопределенность Джини
def _gini(y):
    p = [(1.0 * y[y == k].shape[0]) / y.shape[0] for k in np.unique(y)]
    return 1.0 - np.dot(p, p)


# Дисперсия
def _variance(y):
    return np.var(y) if len(y) > 0 else 0.0


# Среднее отклонение от медианы
def _mad_median(y):
    return np.mean(np.abs(y - np.median(y))) if len(y) > 0 else 0.0


# Словарь критериев разбиения
_criterion_dict = {'entropy': _entropy, 'gini': _gini,
                   'variance': _variance, 'mad_median': _mad_median}


# Словарь соответствия критерия характеру задачи
_criterion_types_dict = {'entropy': 'classification', 'gini': 'classification',
                         'variance': 'regression', 'mad_median': 'regression'}


""" Функции вычисления ответов в узлах
"""


# Ответ для задачи классификации
def _node_value_classification(y):
    return np.argmax(np.bincount(y))


# Ответ для задачи регрессии
def _node_value_regression(y):
    return np.mean(y)


# Словарь функций вычисления оветов в узле
_node_value_dict = {'classification': _node_value_classification,
                    'regression': _node_value_regression}


""" Функции вычисления вероятностных ответов в узлах
"""


# Ответ для задачи классификации
def _node_labels_ratio_classification(y, n_classes):
    return np.bincount(y, minlength=n_classes) / len(y)


# Ответ для задачи регрессии (не определен)
def _node_labels_ratio_regression(y, n_classes):
    return None


# Словарь функций вычисления вероятностных оветов в узле
_node_labels_ratio_dict = {'classification': _node_labels_ratio_classification,
                           'regression': _node_labels_ratio_regression}


""" Класс узла дерева
"""


class _TreeNode():
    """ Tree node class
    Класс узла дерева.

    Parameters
    ----------
    feature_idx : int, default: None
        Индекс признака разбиения в узле.

    threshold : float, default: None
        Значение порога разбиения в узле.

    node_value : float, default: None
        Ответ в узле.

    node_labels_ratio : float, default: None
        Вероятностный ответ в узле (распределение классов в задаче классификации).

    left_child : _TreeNode, default: None
        Левый дочерний узел.

    right_child : _TreeNode, default: None
        Правый дочерний узел.
    """
    def __init__(self, feature_idx=None, threshold=None,
                 node_value=None, node_labels_ratio=None,
                 left_child=None, right_child=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.node_value = node_value
        self.node_labels_ratio = node_labels_ratio
        self.left_child = left_child
        self.right_child = right_child


"""
DecisionTree class
------------------
"""


class DecisionTree(BaseEstimator):
    """ DecisionTree classifier/regressor class.
    Класс реализует модель дерева решений, поддерживающего задачи классификации и регрессии.

    Parameters
    ----------
    max_depth : int or None, default: None
        Максимальная глубина построения дерева.
        Если не задано, максимальная глубина не органичена.

    min_samples_split : int, default: 2
        Минимальное количество примеров обучающей выборки в узле, при котором происходит 
        ее разбиение.

    criterion : string, default: 'gini'
        Тип критерия разбиения. Поддерживаемые значения:
        - для задачи классификации: 
            'gini' (неопределенность Джини)
            'entropy' (энтропия Шеннона)
        - для задачи регрессии:
            'variance' (дисперсия вокруг среднего)
            'mad_median' (среднее отклонение от медианы).

    verbose : bool, default: False
        Флаг вывода сообщений в консоль.

    Attributes
    ----------
    _criterion_func : function
        Функция подсчета критерия разбиения.

    _node_value : function
        Функция вычисления ответа в узле.

    _node_labels_ratio : function
        Функция вычисления вероятностного ответа в узле.

    _n_classes : int or None
        Количество классов в задаче классификации (не определен для задачи регрессии).

    _n_samples : int
        Размер обучающей выборки

    _n_features : int
        Размерность признакового пространства

    _root : _TreeNode
        Корневой узел построенного дерева.
    """
    def __init__(self, max_depth=None, min_samples_split=2,
                 criterion='gini', verbose=False):

        params = {'max_depth': max_depth or np.inf,
                  'min_samples_split': min_samples_split,
                  'criterion': criterion,
                  'verbose': verbose}

        self.set_params(**params)

    def set_params(self, **params):
        """Set parameters for model.
        Метод установки параметров модели. Переопределенный метод родительского класса
        sklearn.base.BaseEstimator.

        Returns
        -------
        self : object
            Возвращает объект класса.
        """
        super(DecisionTree, self).set_params(**params)

        self._criterion_func = _criterion_dict[self.criterion]
        self._node_value = _node_value_dict[_criterion_types_dict[self.criterion]]
        self._node_labels_ratio = _node_labels_ratio_dict[_criterion_types_dict[self.criterion]]

        self._n_classes = None
        self._n_samples = None
        self._n_features = None

        if self.verbose:
            print('DecisionTree class set params:')
            print(params)

        return self

    def _functional(self, X, y, feature_idx, threshold):
        """Get functional value.
        Вычисление значения функционала в узле.

        Parameters
        ----------
        X : array-like, shape = [n_sub_samples, n_features]
            Массив примеров обучающей выборки, находящихся в узле.

        y : array-like, shape = [n_sub_samples] or [n_sub_samples, n_classes]
            Вектор целевых значений для примеров из узла.

        feature_idx : int
            Индекс признака, по которому проводится разбиение в узле.

        threshold : float
            Порог разбиения выборки по признаку.

        Returns
        -------
        float
            Возвращает вычисленное значение функционала в узле.
        """
        division_mask = (X[:, feature_idx] <= threshold)
        X_size = X.shape[0]
        l_size = np.sum(division_mask.astype(int))
        r_size = X_size - l_size

        if X_size != 0:
            return (self._criterion_func(y) -
                    (l_size / X_size) * self._criterion_func(y[division_mask]) -
                    (r_size / X_size) * self._criterion_func(y[~division_mask]))
        else:
            return 0.0

    def _build_tree(self, X, y, depth=1):
        """Build a tree.
        Рекурсивный метод формирования узлов дерева решения.

        Parameters
        ----------
        X : array-like, shape = [n_sub_samples, n_features]
            Массив примеров обучающей выборки, находящихся в узле.

        y : array-like, shape = [n_sub_samples] or [n_sub_samples, n_classes]
            Вектор целевых значений для примеров из узла.

        depth : int, default: 1
            Значение глубины узла, от которого строится текущее дерево.

        Returns
        -------
        _TreeNode
            Возвращает корневой узел построенного дерева.
        """

        # если в узле содержатся объекты одного класса, прерываем разбиение и 
        # возвращаем листовой объект
        if np.unique(y).shape[0] == 1:

            if self.verbose:
                message = 'create leaf (depth={} n_objects={}): value={}'
                message = message.format(depth, X.shape[0], round(self._node_value(y), 2))
                l_ratio = self._node_labels_ratio(y, self._n_classes)
                if l_ratio is not None:
                    message += ' labels_ratio={}'
                    message = message.format(np.round(l_ratio, decimals=2))
                print(message)

            return _TreeNode(node_value=self._node_value(y),
                             node_labels_ratio=self._node_labels_ratio(y, self._n_classes))

        # лучшие параметры разбиения
        best_functional = 0.0
        best_feature_idx = None
        best_threshold = None

        n_node_samples = X.shape[0]

        # проверка критерия останова
        if (depth < self.max_depth) and (n_node_samples >= self.min_samples_split):
            
            # посчитаем функционалы для каждого признака
            for feature_idx in range(self._n_features):

                X_feature_slice = X[:, feature_idx]
                nulls_mask = np.isnan(X_feature_slice)

                # если все непропущенные значения признака на выборке одинаковы, 
                # то признак не рассматриваем
                if np.unique(X_feature_slice[~nulls_mask]).shape[0] == 1:
                    continue

                # множество порогов разбиения как уникальные непропущенные значения признака
                thresholds = np.unique(X_feature_slice[~nulls_mask])
                thresholds = thresholds[:-1]

                # значения функционала для каждого порога
                functionals = [self._functional(X, y, feature_idx, thr) for thr in thresholds]

                # при необходимости обновим лучшие параметры разбиения
                if np.max(functionals) > best_functional:
                    best_functional = np.max(functionals)
                    best_feature_idx = feature_idx
                    best_threshold = thresholds[np.argmax(functionals)]

        # разбиение найдено, выполним рекурсивные вызовы
        if best_feature_idx is not None:

            if self.verbose:
                message = 'node (depth={} n_objects={}) division: feature_idx = {} threshold = {}'
                message = message.format(depth, X.shape[0], best_feature_idx, round(best_threshold, 2))
                print(message)

            best_left_mask = X[:, best_feature_idx] <= best_threshold

            return _TreeNode(feature_idx=best_feature_idx, threshold=best_threshold,
                             node_value=self._node_value(y),
                             node_labels_ratio=self._node_labels_ratio(y, self._n_classes),
                             left_child=self._build_tree(X[best_left_mask, :], y[best_left_mask], depth=depth + 1),
                             right_child=self._build_tree(X[~best_left_mask, :], y[~best_left_mask], depth=depth + 1))

        # разбиение не найдено, возвращаем листовой объект
        else:

            if self.verbose:
                message = 'create leaf (depth={} n_objects={}): value={}'
                message = message.format(depth, X.shape[0], round(self._node_value(y), 2))
                l_ratio = self._node_labels_ratio(y, self._n_classes)
                if l_ratio is not None:
                    message += ' labels_ratio={}'
                    message = message.format(np.round(l_ratio, decimals=2))
                print(message)

            return _TreeNode(node_value=self._node_value(y),
                             node_labels_ratio=self._node_labels_ratio(y, self._n_classes))

    def fit(self, X, y):
        """Fit a model.
        Метод обучения модели (построения дерева решений).

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Массив примеров обучающей выборки.

        y : array-like, shape = [n_samples] or [n_samples, n_classes]
            Вектор целевых значений для примеров обучающей выборки.

        Returns
        -------
        self : object
            Возвращает объект класса.
        """
        self._n_samples, self._n_features = X.shape

        if _criterion_types_dict[self.criterion] == 'classification':
            self._n_classes = np.amax(y) + 1
        else:
            self._n_classes = None

        self._root = self._build_tree(X, y)

        return self

    def _get_object_leaf(self, obj):
        """Get a leaf node for object.
        Метод получения листового узла для примера из тестовой выборки.

        Parameters
        ----------
        obj : array-like, shape = [n_features]
            Пример из тестовой выборки.

        Returns
        -------
        _TreeNode
            Возвращает листовой узел дерева.
        """
        node = self._root

        while node.left_child and node.right_child:
            if obj[node.feature_idx] <= node.threshold:
                node = node.left_child
            else:
                node = node.right_child

        return node

    def _predict_object(self, obj):
        """Get a prediction for object.
        Метод получения прогноза для примера из тестовой выборки.

        Parameters
        ----------
        obj : array-like, shape = [n_features]
            Пример из тестовой выборки.

        Returns
        -------
        int or float
            Возвращает предсказание для примера (индекс класса для задачи классификации
            или вещественное значение для задачи регрессии).
        """
        return self._get_object_leaf(obj).node_value

    def _predict_object_proba(self, obj):
        """Get a probability prediction for object.
        Метод получения вероятностного прогноза для примера из тестовой выборки.

        Parameters
        ----------
        obj : array-like, shape = [n_features]
            Пример из тестовой выборки.

        Returns
        -------
        array-like, shape = [n_classes]
            Возвращает вектор вероятностного предсказания для примера (вероятности 
            принадлежности объекта каждому классу).
        """
        return self._get_object_leaf(obj).node_labels_ratio

    def predict(self, X):
        """Get a prediction for list of objects.
        Метод получения вероятностного прогноза для тестовой выборки.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Массив примеров из тестовой выборки.

        Returns
        -------
        array-like, shape = [n_samples]
            Возвращает вектор предсказаний для выборки (индекс класса для задачи 
            классификации или вещественное значение для задачи регрессии).
        """
        return np.array([self._predict_object(obj) for obj in X])

    def predict_proba(self, X):
        """Get a probability prediction for list of objects.
        Метод получения вероятностного прогноза для тестовой выборки.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Массив примеров из тестовой выборки.

        Returns
        -------
        array-like, shape = [n_samples, n_classes]
            Возвращает массив вероятностных предсказаний для выборки (вероятности 
            принадлежности объектов каждому классу).
        """
        return np.array([self._predict_object_proba(obj) for obj in X])
