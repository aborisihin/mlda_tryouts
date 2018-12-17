""" auto_sklearn_solver module.
Contains AutoSklearnSolver class.
"""

import os
import shutil
import pickle
import warnings

from typing import Any, Dict, Callable

import pandas as pd

from autosklearn import metrics
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.regression import AutoSklearnRegressor

from utils.config import Config
from utils.data_reader import read_df
from utils.data_processor import process_dataframe
from utils.logger import time_logging, log

__all__ = ['AutoSklearnSolver']


class AutoSklearnSolver:
    """ Model implementing through auto-sklearn.
    https://github.com/automl/auto-sklearn
    Класс реализует работу модели через функциональность auto-sklearn.

    Args:
        model_dir: Путь к директории модели
        time_limit: Временной лимит на обучение модели (с)
        memory_limit: Лимит на объем используемой памяти (Мб)

    Attributes:
        model_dir (str): Путь к каталогу модели
        config (Config): Параметры модели
        model ([AutoSklearnClassifier, AutoSklearnRegressor]): Объект модели auto-sklearn
        per_run_time_limit (int): Временной лимит на обучение модели
        metrics_object (autosklearn.metrics): Объект метрики качества содели
        procesed_data_path (str): Путь сохранения обработанных данных
    """
    def __init__(self, model_dir: str, time_limit: int=0, memory_limit: int=0) -> None:
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        self.config = Config(model_dir, time_limit, memory_limit)
        self.model = None
        self.per_run_time_limit = min(360, time_limit // 2)

    @time_logging
    def fit(self, train_csv: str, mode: str, metrics_name: str, save_processed_data: bool) -> None:
        """Start model fitting
        Запуск процесса обучения модели

        Args:
            train_csv: Путь к обучающему датасету
            mode: Режим работы (классификация или регрессия)
            metrics_name: Имя объекта метрики качества в модуле autosklearn.metrics
            save_processed_data: Флаг сохранения датасета с обработанными данными
        """
        if not os.path.exists(train_csv):
            log('Data file {} is not exist!'.format(train_csv))
            return

        # получение объекта метрики
        try:
            self.metrics_object = getattr(metrics, metrics_name)
        except AttributeError as error:
            self.metrics_object = None
            log('Can\'t get the metrics object!')
            log('{}: {}'.format(type(error).__name__, error))
            return

        # подготовка каталога для сохранения данных
        if save_processed_data:
            self.procesed_data_path = os.path.join(self.model_dir, 'processed_data')
            os.makedirs(self.procesed_data_path, exist_ok=True)

        self.config['task'] = 'fit'
        self.config['mode'] = mode
        self.config['tmp_dir'] = self.config['model_dir'] + '/tmp'

        # удаление временной директории 
        # (auto-sklearn ругается перед началом работы, если этого не делать)
        shutil.rmtree(self.config['tmp_dir'], ignore_errors=True)

        # первичный анализ, чтение данных, разбитие на матрицы X и y
        df = read_df(train_csv, self.config)
        y = df['target']
        X = df.drop('target', axis=1)

        # обработка данных
        process_dataframe(X, self.config)

        if save_processed_data:
            log('Saving processed data')
            X.to_csv(os.path.join(self.procesed_data_path, 'X.csv'))
            y.to_csv(os.path.join(self.procesed_data_path, 'y.csv'))

        # параметры создаваемой auto-sklearn модели
        # (выключаем препроцессинг, т.к. он уже проведен)
        model_params = {'time_left_for_this_task': self.config.time_left(),
                        'per_run_time_limit': self.per_run_time_limit,
                        'ml_memory_limit': self.config['memory_limit'],
                        'tmp_folder': self.config['tmp_dir'],
                        'include_preprocessors': ['no_preprocessing'],
                        'delete_tmp_folder_after_terminate': True}

        # инициализация объекта модели
        self.model_init(model_params)

        # обучение модели
        self.model_fit(X, y, self.metrics_object)

        log('model_fitted: {}'.format(type(self.model)))
        log('autosklearn model contains:')
        log(self.model.show_models())

    @time_logging
    def model_init(self, model_params: Dict[str, Any]) -> None:
        """Model initialization
        Инициализация объекта модели в зависимости от типа задачи

        Args:
            model_params: Словарь параметров модели
        """
        if self.config['mode'] == 'classification':
            self.model = AutoSklearnClassifier(**model_params)
        elif self.config['mode'] == 'regression':
            self.model = AutoSklearnRegressor(**model_params)

    @time_logging
    def model_fit(self, X: pd.DataFrame, y: pd.Series, metrics: Callable) -> None:
        """Model fitting wrapper
        Обертка для вызова fit (для учета времени в логе)

        Args:
            X: Матрица признаков
            y: Вектор ответов
            metrics: Объект метрики качества
        """
        # подавляем вывод предупреждений в лог
        warnings.filterwarnings('ignore', category=FutureWarning)
        warnings.filterwarnings('ignore', category=RuntimeWarning)

        self.model.fit(X, y, metric=metrics)

        warnings.resetwarnings()

    @time_logging
    def predict(self, test_csv: str, prediction_csv: str, validation_csv: str, need_proba: bool) -> pd.DataFrame:
        """Start model prediction
        Запуск процесса предсказывания целевого признака на новых данных

        Args:
            test_csv: Путь к тестовому датасету
            prediction_csv: Путь для записи ответов модели
            validation_csv: Путь к датасету правильных ответов на тестовой выборке (для подсчета метрики)
            need_proba: Флаг необходимости выдавать вероятностные предсказания

        Returns:
            Датасет с ответами модели
        """
        if not os.path.exists(test_csv):
            log('Data file {} is not exist!'.format(test_csv))
            return

        self.config['task'] = 'predict'

        df = read_df(test_csv, self.config)
        process_dataframe(df, self.config)

        predictions_df = self.model_predict(df, prediction_csv, need_proba)

        if validation_csv != 'None':
            self.model_validate(predictions_df, validation_csv)

    @time_logging
    def model_predict(self, X: pd.DataFrame, prediction_csv: str, need_proba: bool) -> pd.DataFrame:
        """Model predict wrapper
        Обертка для вызова predict

        Args:
            X: Матрица признаков
            prediction_csv: Путь для записи ответов модели
            need_proba: Флаг необходимости выдавать вероятностные предсказания
        """
        if (self.config['mode'] == 'classification') and need_proba:
            predictions = self.model.predict_proba(X, n_jobs=-1)
            df_columns = ['target_0', 'target_1']
        else:
            predictions = self.model.predict(X, n_jobs=-1)
            df_columns = ['target']

        # подготовка каталога для записи ответов
        output_dir = '/'.join(prediction_csv.split('/')[:-1])
        os.makedirs(output_dir, exist_ok=True)

        # запись датафрейма с ответами
        predictions_df = pd.DataFrame(predictions, index=X.index, columns=df_columns)
        predictions_df.to_csv(prediction_csv)

        return predictions_df

    @time_logging
    def model_validate(self, predictions_df: pd.DataFrame, validation_csv: str) -> None:
        """Model validate
        Валидирование модели по известным правильным ответам

        Args:
            prediction_csv: Путь для записи ответов модели
            validation_csv: Путь к датасету правильных ответов на тестовой выборке
        """
        if self.metrics_object is None:
            log('Can\'t get the metrics object!')
            return

        if not os.path.exists(validation_csv):
            log('Validation file {} is not exist!'.format(validation_csv))
            return

        # чтение датасета с правильными ответами
        validation_df = pd.read_csv(validation_csv, encoding='utf-8', sep=',')
        
        # объединение правильных и предсказанных ответов для соответствия по индексам
        compare_df = pd.merge(validation_df, predictions_df, on="line_id")

        # подсчет score
        # в объединенном датасете будут следующий индексы столбцов:
        # 0: index, 1: true values, 2-...: predicted values
        score = self.metrics_object(compare_df.iloc[:, 1].values, compare_df.iloc[:, 2:].values)
        log('Metrics: {}'.format(self.metrics_object))
        log('Score: {}'.format(score))

        return score

    @time_logging
    def save(self) -> None:
        """Save model, parameters and metrics object
        Сохранение на диск модели, параметров и объекта метрики
        """
        self.config.save()
        with open(os.path.join(self.config['model_dir'], 'model.pkl'), 'wb') as f:
            pickle.dump(self.model, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(self.config['model_dir'], 'metrics_object.pkl'), 'wb') as f:
            pickle.dump(self.metrics_object, f, protocol=pickle.HIGHEST_PROTOCOL)

    @time_logging
    def load(self) -> None:
        """Load model, parameters and metrics object
        Загрузка с диска модели, параметров и объекта метрики
        """
        self.config.load()
        with open(os.path.join(self.config['model_dir'], 'model.pkl'), 'rb') as f:
            self.model = pickle.load(f)
        with open(os.path.join(self.config['model_dir'], 'metrics_object.pkl'), 'rb') as f:
            self.metrics_object = pickle.load(f)

    def __repr__(self) -> str:
        repr_string = 'AutoSklearnSolver\n'
        repr_string += '-----------------\n'
        repr_string += str(self.config)
        return repr_string
