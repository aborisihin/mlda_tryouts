""" Prediction script.
"""

import argparse

from solvers.auto_sklearn_solver import AutoSklearnSolver


def main():
    # описание файла настроек можно найти в README.md проекта.
    parser = argparse.ArgumentParser()
    parser.add_argument('--solver', choices=['auto-sklearn'], required=True)
    parser.add_argument('--test-csv', type=str, required=True)
    parser.add_argument('--prediction-csv', type=str, required=True)
    parser.add_argument('--validation-csv', type=str, required=True)
    parser.add_argument('--model-dir', type=str, required=True)
    parser.add_argument('--need-proba', type=str, choices=['True', 'False'], required=False, default='False')
    args = parser.parse_args()

    print()

    # выбор метода выполнения задачи
    if args.solver == 'auto-sklearn':
        solver = AutoSklearnSolver(model_dir=args.model_dir)
    else:
        exit(1)

    solver.load()
    solver.predict(test_csv=args.test_csv,
                   prediction_csv=args.prediction_csv,
                   validation_csv=args.validation_csv,
                   need_proba=bool(args.need_proba == 'True'))


if __name__ == '__main__':
    main()
