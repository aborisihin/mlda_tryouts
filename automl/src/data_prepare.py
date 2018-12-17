""" Data preparation script.
"""

import os
import argparse
import json

import pandas as pd

from utils.logger import time_logging, log


@time_logging
def data_prepare():
    """Prepare data for AutoML.
    Подготовка данных для работы системы AutoML.
    Описание файла настроек можно найти в README.md проекта.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.settings):
        log("Settings file {} is not exist!".format(args.settings))
        return

    with open(args.settings, 'r') as settings_file:
        settings = json.load(settings_file)

    # проверка наличия датасета и подготовка выходной директории
    if not os.path.exists(settings['input_path']):
        log("Data file {} is not exist!".format(settings['input_path']))
        return

    output_dir = '/'.join(settings['output_path'].split('/')[:-1])
    os.makedirs(output_dir, exist_ok=True)

    # чтение данных
    data = pd.read_csv(settings['input_path'], **settings['reader_params'])

    # переименование значений признаков
    for feat, feat_map in settings['data_markup_params']['feature_values_transformers'].items():
        data[feat] = data[feat].map(feat_map)

    # переименование целевого признака
    data.rename({settings['data_markup_params']['target_column']: 'target'}, axis=1, inplace=True)

    # добавление префиксов по типам данных
    features_rename_mapping = {}
    for prefix, features in settings['data_markup_params']['feature_names_mapping'].items():
        features_rename_mapping.update({feat: '{}_{}'.format(prefix, feat) for feat in features})

    data.rename(features_rename_mapping, axis=1, inplace=True)

    # сохранение результатов
    data.to_csv(settings['output_path'], **settings['writer_params'])
    log('Data prepared: {}'.format(settings['output_path']))

if __name__ == '__main__':
    data_prepare()
