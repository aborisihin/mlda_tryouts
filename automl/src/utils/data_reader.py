""" data_reader module.
Contains dataframe reading methods
"""

import pandas as pd

from utils.logger import time_logging, log
from utils.config import Config

__all__ = ['read_df']


@time_logging
def read_df(csv_path: str, config: Config) -> pd.DataFrame:
    """Reading the dataframe. Use this method to run all steps in reading process.
    Чтение исходного датафрейма. Заглавный метод модуля, используется для запуска
    всех этапов чтения.

    Args:
        csv_path: Путь к файлу с данными
        config: Параметры модели

    Returns:
        pandas.DataFrame: Объект прочитанного датасета
    """
    log('Read dataset: {}'.format(csv_path))

    if config.is_fit():
        preview_df(csv_path, config)

    df = pandas_read_df(csv_path, config)

    config['actual_dataset_size'] = round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
    log('Actual dataset size: {:0.2f} Mb'.format(config['actual_dataset_size']))

    config['nrows'] = df.shape[0]

    return df


@time_logging
def pandas_read_df(csv_path: str, config: Config) -> pd.DataFrame:
    """pandas.read_csv() wrapping method
    Метод-обертка для вызова pandas.read_csv() с заданными параметрами.

    Args:
        csv_path: Путь к файлу с данными
        config: Параметры модели

    Returns:
        pandas.DataFrame: Объект прочитанного датасета.
    """
    return pd.read_csv(csv_path, encoding='utf-8', low_memory=False, sep=',', index_col=None,
                       dtype=config['dtype'], parse_dates=config['parse_dates'])


@time_logging
def preview_df(csv_path: str, config: Config, nrows: int=100) -> None:
    """Dataframe preview method
    Метод предпросмотра датафрейма для получения его характеристик

    Args:
        csv_path: Путь к файлу с данными
        config: Параметры модели
        nrows: Количество строк для загрузки
    """
    with open(csv_path) as csv_file:
        total_rows = sum([1 for line in csv_file]) - 1
    log('Total rows in dataset: {}'.format(total_rows))

    df = pd.read_csv(csv_path, encoding='utf-8', low_memory=False, nrows=nrows)

    # подсчет примерного размера датасета
    mem_per_row = df.memory_usage(deep=True).sum() / nrows
    df_size = (total_rows * mem_per_row) / 1024 / 1024
    log('Approximate dataset size: {:0.2f} Mb'.format(df_size))

    # разбор признаков по типам
    config['parse_dates'] = []
    config['dtype'] = dict()

    type_counters = {'id': 0,
                'number': 0,
                'cat': 0,
                'string': 0,
                'datetime': 0}

    for feature in df:
        if feature.startswith("id_"):
            type_counters["id"] += 1
        elif feature.startswith("number_"):
            type_counters["number"] += 1
        elif feature.startswith("cat_"):
            type_counters["cat"] += 1
        elif feature.startswith("string_"):
            type_counters["string"] += 1
            config["dtype"][feature] = str
        elif feature.startswith("datetime_"):
            type_counters["datetime"] += 1
            config["dtype"][feature] = str
            #config["parse_dates"].append(feature)

    log("ID features: {}".format(type_counters["id"]))
    log("Number features: {}".format(type_counters["number"]))
    log("Categorical features: {}".format(type_counters["cat"]))
    log("String features: {}".format(type_counters["string"]))
    log("Datetime features: {}".format(type_counters["datetime"]))

    config["type_counters"] = type_counters
