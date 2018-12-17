""" config module.
Contains Config class. Used for model settings storing.
"""

import os
import time
import pickle

from typing import Any

__all__ = ['Config']


class Config:
    """ Model configuration
    Класс реализует контейнер параметров, опредеяемых в процессе инициализации/обучения
    модели, и необходимых для ее воспроизведения и использования.

    Args:
        model_dir: Путь к директории модели
        time_limit: Временной лимит на обучение модели (с)
        memory_limit: Лимит на объем используемой памяти (Мб)

    Attributes:
        data (dict[string, Any]): Словарь, содержаший параметры модели

    """

    def __init__(self, model_dir: str, time_limit: int, memory_limit: int) -> None:
        self.data = {'model_dir': model_dir,
                     'time_limit': time_limit,
                     'memory_limit': memory_limit,
                     'start_time': time.time()}

    def time_left(self) -> int:
        """Calculate remaining time for model fitting
        Расчет оставшегося времени для обучения модели
        """
        if self['time_limit'] is not None:
            return int(round(self['time_limit'] - (time.time() - self['start_time'])))
        else:
            return None

    def save(self) -> None:
        """Save parameters
        Сохранение параметров модели
        """
        with open(os.path.join(self.data['model_dir'], 'config.pkl'), 'wb') as f:
            pickle.dump(self.data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self) -> None:
        """Load parameters
        Загрузка параметров модели
        """
        with open(os.path.join(self.data['model_dir'], 'config.pkl'), 'rb') as f:
            data = pickle.load(f)
        self.data = {**data, **self.data}

    def is_fit(self) -> bool:
        """Check is fit
        Проверка режима работы модели (обучение)
        """
        return self.data['task'] == 'fit'

    def is_predict(self) -> bool:
        """Check is predict
        Проверка режима работы модели (предсказание)
        """
        return self.data['task'] == 'predict'

    def update(self, d: dict) -> None:
        self.data.update(d)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value

    def __delitem__(self, key: str) -> None:
        del self.data[key]

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        repr_string = ''
        for key, val in self.data.items():
            repr_string += '{}: {}\n'.format(key, val)
        return repr_string
