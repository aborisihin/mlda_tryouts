""" Model fitting script.
"""

import argparse

from estimator.model_fitter import HyperoptModelFitter
from estimator.param_space import parameters_space


def main():
    # описание файла настроек можно найти в README.md проекта.
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyperopt_max_evals', type=int, required=True)
    args = parser.parse_args()

    # объект model_fitter
    fitter = HyperoptModelFitter(data_path='./../scrapped_data/yandex_mobile_reviews.csv',
        param_space=parameters_space,
        max_evals=args.hyperopt_max_evals)

    # подбор параметров и сохранение объектов пайплайна
    print('start parameters search...')
    fitter.fit()

    print('best params:')
    print(fitter.get_best_params())

    fitter.save_model('./../model')
    print('model saved!')


if __name__ == '__main__':
    main()
