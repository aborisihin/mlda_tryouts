""" Model fitting script.
"""

import argparse

from solvers.auto_sklearn_solver import AutoSklearnSolver


def main():
    # описание файла настроек можно найти в README.md проекта.
    parser = argparse.ArgumentParser()
    parser.add_argument('--solver', type=str, choices=['auto-sklearn'], required=True)
    parser.add_argument('--mode', type=str, choices=['classification', 'regression'], required=True)
    parser.add_argument('--metrics', type=str, required=True)
    parser.add_argument('--train-csv', type=str, required=True)
    parser.add_argument('--model-dir', type=str, required=True)
    parser.add_argument('--time-limit', type=int, required=True)
    parser.add_argument('--memory-limit', type=int, required=True)
    parser.add_argument('--save-processed-data', type=str, choices=['True', 'False'], required=True)
    args = parser.parse_args()

    print()

    # выбор метода выполнения задачи
    if args.solver == 'auto-sklearn':
        solver = AutoSklearnSolver(model_dir=args.model_dir,
                                   time_limit=args.time_limit,
                                   memory_limit=args.memory_limit)
    else:
        exit(1)

    solver.fit(train_csv=args.train_csv, mode=args.mode, metrics_name=args.metrics,
               save_processed_data=bool(args.save_processed_data == 'True'))
    solver.save()


if __name__ == '__main__':
    main()
