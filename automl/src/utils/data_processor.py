""" data_processor module.
Contains dataframe processing methods
"""

import datetime
import itertools

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.logger import time_logging, log
from utils.config import Config

__all__ = ['process_dataframe']

# константы

# доля числа уникальных значений целочисленного признака, ниже которой признак считается категориальным
INT_NUNIQUE_RATE = 0.2
# доля числа уникальных значений строкового признака, ниже которой признак считается категориальным
STRING_NUNIQUE_RATE = 0.5
# среднее число слов в строковом признаке, свыше которого tfidf считается по словам (иначе - по символам)
AVG_WC_THRESHOLD = 3
# диапазон n-грам для слов
WORD_NGRAM_RANGE = (1, 2)
# диапазон n-грам для символов
CHAR_NGRAM_RANGE = (1, 5)
# максимальное число tfidf признаков для одного строкового
MAX_TFIDF_FEATURES = 100

# поля Config, в которых содержатся списки известных признаков; неизвестные из датасета удаляются
types_config_fields = ['datetime_features', 'cat_features', 'int_features', 'float_features',
                       'string_wordwise_features', 'string_charwise_features',
                       'string_tfidf_features', 'cat_dummy_features']


@time_logging
def process_dataframe(df: pd.DataFrame, config: Config) -> None:
    """Process the dataframe. Use this method to run all steps in process.
    Обработка исходного датафрейма. Заглавный метод модуля, используется для запуска
    всех этапов обработки.

    Args:
        df: Датафрейм для обработки
        config: Параметры модели
    """
    if config.is_fit():
        # инициализация пустыми списками
        init_fields = types_config_fields + ['constant_features', 'unknown_features']
        config.update({key: [] for key in init_fields})

    # запуск пайплайна
    features_processing_pipeline(df, config)

    log('Processed features count: {}'.format(df.shape[1]))


def features_processing_pipeline(df: pd.DataFrame, config: Config) -> None:
    """Dataframe processing pipeline.
    Пайплайн обработки датафрейма

    Args:
        df: Датафрейм для обработки
        config: Параметры модели
    """
    pipeline_steps = (set_index,
                      fillna,
                      process_datetime,
                      process_numeric,
                      process_string,
                      process_categorical,
                      drop_features,
                      scale,
                      optimize_memory)

    for step in pipeline_steps:
        step(df, config)


@time_logging
def set_index(df: pd.DataFrame, config: Config) -> None:
    """Set index column in dataframe.
    Установка индекс-столбца в датафрейме

    Args:
        df: Датафрейм для обработки
        config: Параметры модели
    """
    if 'line_id' in df:
        df.set_index(keys='line_id', drop=True, inplace=True)


@time_logging
def fillna(df: pd.DataFrame, config: Config) -> None:
    """Fill NA values in dataframe.
    Заполнение пропусков в датафрейме

    Args:
        df: Датафрейм для обработки
        config: Параметры модели
    """
    for feat in df:
        if feat.startswith('cat_'):
            df[feat].fillna(-1, inplace=True)
        elif feat.startswith('number_'):
            df[feat].fillna(-1, inplace=True)
        elif feat.startswith('string_'):
            df[feat].fillna('', inplace=True)
        elif feat.startswith('datetime_'):
            df[feat].fillna('1970-01-01', inplace=True)


@time_logging
def process_datetime(df: pd.DataFrame, config: Config) -> None:
    """Process datetime features.
    Обрабокта datetime признаков

    Args:
        df: Датафрейм для обработки
        config: Параметры модели
    """
    if config.is_fit():
        config['datetime_features'] = [feat for feat in df if feat.startswith('datetime_')]

    if len(config['datetime_features']) == 0:
        return

    # метод конвертации строки в datetime
    def parse_dt(val):
        if not isinstance(val, str):
            return None
        elif len(val) == len('2010-01-01'):
            return datetime.datetime.strptime(val, '%Y-%m-%d')
        elif len(val) == len('2010-01-01 10:10:10'):
            return datetime.datetime.strptime(val, '%Y-%m-%d %H:%M:%S')
        else:
            return None

    # трансформеры для генерации признаков (кортежи префикс-трансформер)
    transformers = [('cat_year_', lambda x: x.year),
                    ('cat_month_', lambda x: x.month),
                    ('cat_day_', lambda x: x.day),
                    ('cat_hour_', lambda x: x.hour),
                    ('cat_minute_', lambda x: x.minute),
                    ('cat_second_', lambda x: x.second),
                    ('cat_weekday_', lambda x: x.weekday()),
                    ('number_hour_of_week_', lambda x: x.hour + x.weekday() * 24),
                    ('number_minute_of_day_', lambda x: x.minute + x.hour * 60)]

    # генерация признаков
    for feat in config['datetime_features']:
        df[feat] = df[feat].apply(parse_dt)
        for (prefix, method) in transformers:
            df[prefix + feat] = df[feat].apply(method)
        df.drop([feat], axis=1, inplace=True)


@time_logging
def process_numeric(df: pd.DataFrame, config: Config) -> None:
    """Process numeric features.
    Обрабокта численных признаков

    Args:
        df: Датафрейм для обработки
        config: Параметры модели
    """
    if config.is_fit():
        int_dtypes = ['int', 'int32', 'int64']
        float_dtypes = ['float', 'float32', 'float64']
        for feat in [f for f in df if f.startswith('number_')]:
            if df.dtypes[feat] in int_dtypes:
                # проверка, можно ли отнести целочисленный признак к категориальному
                unique_rate = df[feat].nunique() / df.shape[0]
                if unique_rate < INT_NUNIQUE_RATE:
                    config['cat_features'].append(feat)
                else:
                    config['int_features'].append(feat)
            elif df.dtypes[feat] in float_dtypes:
                    config['float_features'].append(feat)


@time_logging
def process_string(df, config):
    """Process string features.
    Обрабокта строковых признаков

    Args:
        df: Датафрейм для обработки
        config: Параметры модели
    """
    if config.is_fit():
        config['string_features'] = []
        for feat in [f for f in df if f.startswith('string_')]:
            # проверка, можно ли отнести строковый признак к категориальному
            unique_rate = df[feat].nunique() / df.shape[0]
            if unique_rate < STRING_NUNIQUE_RATE:
                config['cat_features'].append(feat)
            else:
                config['string_features'].append(feat)

        # работа с оставшимися строковыми признаками
        if len(config['string_features']) > 0:
            def avg_word_counter(col):
                return col.apply(lambda val: len(val.split(' '))).mean()

            # вектор соредними значениями слов в строковых признаках
            avg_wc = df[config['string_features']].apply(avg_word_counter, axis=0)
            # разделение признаков на группы: tfidf по словам, tfidf по символам
            string_wordwise_features = list(df[config['string_features']].loc[:, avg_wc > AVG_WC_THRESHOLD].columns)
            string_charwise_features = list(df[config['string_features']].loc[:, avg_wc <= AVG_WC_THRESHOLD].columns)
            if len(string_wordwise_features) > 0:
                config['string_wordwise_features'] = string_wordwise_features
            if len(string_charwise_features) > 0:
                config['string_charwise_features'] = string_charwise_features

            # словарь TfidfVectorizer для каждого признака
            config['string_features_encoders'] = dict()

            for feat in config['string_wordwise_features'] + config['string_charwise_features']:
                if feat in config['string_wordwise_features']:
                    tfidf = TfidfVectorizer(analyzer='word', ngram_range=WORD_NGRAM_RANGE,
                                            max_features=MAX_TFIDF_FEATURES)
                elif feat in config['string_charwise_features']:
                    tfidf = TfidfVectorizer(analyzer='char', ngram_range=CHAR_NGRAM_RANGE,
                                            max_features=MAX_TFIDF_FEATURES)
                tfidf.fit(df[feat])
                config['string_features_encoders'][feat] = tfidf

    # кодирование каждого строкового признака своим TfidfVectorizer
    for feat in config['string_wordwise_features'] + config['string_charwise_features']:
        tfidf = config['string_features_encoders'][feat]
        tfidf_features = np.array(tfidf.transform(df[feat]).todense())
        tfidf_features_names = ['tfidf_{}_{}'.format(feat, idx) for idx in range(tfidf_features.shape[1])]

        config['string_tfidf_features'] += tfidf_features_names
        df[tfidf_features_names] = pd.DataFrame(tfidf_features, index=df.index)
        df.drop(feat, axis=1, inplace=True)


@time_logging
def process_categorical(df: pd.DataFrame, config: Config) -> None:
    """Process categorical features.
    Обрабокта категориальных признаков

    Args:
        df: Датафрейм для обработки
        config: Параметры модели
    """
    if config.is_fit():
        # дополняем список уже найденных категориальных признаков теми, которые заданы явно
        config['cat_features'] += [feat for feat in df if feat.startswith('cat_')]

        if len(config['cat_features']) > 0:
            # комбинация LabelEncoder+OneHotEncoder не проходит, т.к. LabelEncoder не поддерживает
            # новые значения признака; боль закончится после выхода стабильного sklearn-0.20
            # (появится ColumnTransformer для прохода сразу по всем кат. признакам)
            # пока пользуемся методом pd.get_dummies: выполняем кодирование и сохраняем
            # список категорий; на тестовой выборке будем выполнять reindex для поддержки
            # отсутствующих или новых категорий
            config['feat_dummy_columns'] = dict()
            for feat in config['cat_features']:
                feat_dummy_columns = pd.get_dummies(df[feat], prefix=feat).columns
                config['feat_dummy_columns'][feat] = feat_dummy_columns

    for feat in config['cat_features']:
        feat_dummies = pd.get_dummies(df[feat], prefix=feat)
        feat_dummies = feat_dummies.reindex(columns=config['feat_dummy_columns'][feat], fill_value=0)

        config['cat_dummy_features'] += list(config['feat_dummy_columns'][feat])
        df[config['feat_dummy_columns'][feat]] = feat_dummies
        df.drop(feat, axis=1, inplace=True)


@time_logging
def drop_features(df: pd.DataFrame, config: Config) -> None:
    """Drop unknown features.
    Удаление константных и неизвестных признаков (в известные включаются все те признаки,
    которые прошли предыдущие стадии обработки датафрейма).

    Args:
        df: Датафрейм для обработки
        config: Параметры модели
    """
    if config.is_fit():
        # константные признаки
        config['constant_features'] = [feat for feat in df if df[feat].nunique() == 1]

        # список списков известных признаков
        l_used_features = [config[tcf] for tcf in types_config_fields]
        # получение одного общего списка
        used_features = list(itertools.chain.from_iterable(l_used_features))
        # оставшиеся (неизвестные) признаки
        config['unknown_features'] = list(set(df.columns).difference(set(used_features)))
        log('Unknown features: {}'.format(config['unknown_features']))

        config['dropped_features'] = config['constant_features'] + config['unknown_features']

    df.drop([feat for feat in config['dropped_features'] if feat in df], axis=1, inplace=True)


@time_logging
def scale(df: pd.DataFrame, config: Config) -> None:
    """Scale features.
    Масштабирование признаков.

    Args:
        df: Датафрейм для обработки
        config: Параметры модели
    """
    if config.is_fit():
        scaler = StandardScaler(copy=False)
        scaler.fit(df)
        config['scaler'] = scaler

    scaler = config['scaler']
    df = scaler.transform(df)


@time_logging
def optimize_memory(df: pd.DataFrame, config: Config) -> None:
    """Optimize dataframe memory usage.
    Оптимизирование используемой памяти. Источник:
    https://www.dataquest.io/blog/pandas-big-data/
    Используется только downcast численных признаков.

    Args:
        df: Датафрейм для обработки
        config: Параметры модели
    """
    if config.is_predict():
        return

    int_features = []
    float_features = []
    for feat in df:
        feat_type = df.dtypes[feat]
        if feat_type in ['int', 'int32', 'int64']:
            int_features.append(feat)
        elif feat_type in ['float', 'float32', 'float64']:
            float_features.append(feat)

    config['processed_dataset_size'] = round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
    log('Processed dataframe size: {:0.2f} Mb'.format(config['processed_dataset_size']))

    if len(int_features) > 0:
        df[int_features] = df[int_features].apply(pd.to_numeric, downcast='integer')

    if len(float_features) > 0:
        df[float_features] = df[float_features].apply(pd.to_numeric, downcast='float')

    config['reduced_dataset_size'] = round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
    log('Reduced dataframe size: {:0.2f} Mb'.format(config['reduced_dataset_size']))
