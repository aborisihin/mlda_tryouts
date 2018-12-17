""" logger module.
Contains time_logging() method decorator and log() method.
"""

import time

__all__ = ['time_logging', 'log']

# текущий уровень отступа логирования
nesting_level = 0


def time_logging(method):
    """Execution time logging method decorator.
    Декоратор для методов, используется для отслеживания времени исполнения.
    """
    def time_logged(*args, **kw):
        global nesting_level

        log("Start {}".format(method.__name__))
        nesting_level += 1

        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()

        nesting_level -= 1
        log("End {}; time: {:0.2f} sec".format(method.__name__, end_time - start_time))

        return result

    return time_logged


def log(entry):
    """Log message method.
    Вывод строки в лог с текущим уровнем отступа.
    """
    global nesting_level
    space = " " * (4 * nesting_level)
    print("{}{}".format(space, entry))
