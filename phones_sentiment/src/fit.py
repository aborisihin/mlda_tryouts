""" Model fitting script.
"""

import os
import argparse

from estimator.model_fitter import HyperoptModelFitter
from estimator.param_space import parameters_space

data_path = './scrapped_data/yandex_mobile_reviews.csv'


def main():
    if not os.path.isfile(data_path):
        print('Error: dataset file doesn\'t exist! Run `make scrapy-run` first.')
        return

    # описание файла настроек можно найти в README.md проекта.
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyperopt_max_evals', type=int, required=True)
    args = parser.parse_args()

    fitter = HyperoptModelFitter(data_path=data_path,
                                 param_space=parameters_space,
                                 max_evals=args.hyperopt_max_evals)
    print('start data preparing...')
    fitter.prepare_data()
    print('start parameters search...')
    fitter.fit()
    print('best params:')
    print(fitter.best_params)
    fitter.save_model('./model')
    print('model saved!')


if __name__ == '__main__':
    main()
