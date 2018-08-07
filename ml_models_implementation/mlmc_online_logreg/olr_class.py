""" olr_class module.
Contains OnlineLogisticRegression class.
"""

import numpy as np

from collections import defaultdict

__all__ = ['OnlineLogisticRegression']


class OnlineLogisticRegression():

    """ OnlineLogisticRegression classifier
    Класс реализует модель multiclass/multilabel классификации текстов, используя методы онлайн
    обучения (стохастический градиентный спуск). В качестве регуляризатора используется ElasticNet
    (комбинация L1 и L2). Поддерживаются повторные проходы по тренировочному датасету и ограничения
    на размер словаря.

    Parameters
    ----------
    tags : [string]
        Список допустимых для классификации тегов (классов). Не входящие в этот список теги
        игнорируются.

    strategy : ['ovr', 'multinomial'], default: 'ovr'
        Признак типа классификации. Значение 'ovr' задает бинарную классификацию (присутствие/отсутствие)
        для каждого тега, теги независимы (one-vs-rest, multilabel classification). Значение
        'multinomial' задает минимизацию ошибки общего дискретного вероятностного распределения
        (multiclass classification).

    learning_rate : float, default: 0.1
        Скорость обучения градиентного спуска (множитель корректировки параметра модели на каждом шаге).

    lmbda : float, default: 0.0002
        Коэффициент ElasticNet-регуляризации на каждом шаге.

    gamma : float, default: 0.1
        Вес L2-компоненты в ElasticNet.

    store_frequency : bool, default: True
        Флаг хранения частот слов в выборке для последующего снижения размерности признакового
        пространства (сильно понижающая скорость первоначального обучения операция).

    tolerance : float, default: 1e-16
        Порог для огнаничения значений аргумента логарифма.

    Attributes
    ----------
    vocab_ : dict {string: int}
        Mapping слов-признаков в численные индексы. Слова добавляются в словарь в процессе обучения,
        индексы назначаются инкрементально.
        Т.к. обучение модели ведется по онлайн-схеме, то всего признакового пространства мы не знаем.
        В этом случае пользоваться bag-of-words или, например, CountVectorizer из sklearn, не
        целесообразно (словарь придется пересчитывать при каждом появлении нового слова).

    w_ : dict {string: defaultdict(int)}
        Mapping тегов в словарь {<численный_индекс_признака>: <вес_в_модели>} (для каждого тега
        свой набор параметров модели). Словарь изменяемого размера со значением по умолчанию 0.

    w0_ : dict {string: float}
        Mapping тегов в веса w0 (смещения).

    train_frequency_dict_ : defaultdict {int: int}
        Словарь {<численный_индекс_признака>: <число_вхождений>}. Выполняет подсчет числа
        вхождений признака на всей тренировочной выборке.

    loss_ : [double]
        Список значений функции потерь для последней используемой обучающей выборки.

    """
    def __init__(self, tags, strategy='ovr', learning_rate=0.1, lmbda=0.0002, gamma=0.1,
                 store_frequency=True, tolerance=1e-16):
        self.vocab_ = {}
        self.w_ = {t: defaultdict(int) for t in tags}
        self.w0_ = {t: 0.0 for t in tags}
        self.train_frequency_dict_ = defaultdict(int)
        self.loss_ = []

        self.tags_ = set(tags)
        self.strategy_ = strategy
        self.learning_rate_ = learning_rate
        self.lmbda_ = lmbda
        self.gamma_ = gamma
        self.store_frequency_ = store_frequency
        self.tolerance_ = tolerance

    def fit(self, datasource, labels_datasource, update_vocab=True, return_train_loss=False):
        """Fit/update the model by passing the datasource
        Обучение/дообучение модели одним проходом по источнику данных.

        Parameters
        ----------
        datasource : iterable
            Итерируемый объект как источник данных. Способен возвращать объекты выборки
            (обучающие строки).

        labels_datasource : iterable
            Итерируемый объект как источник данных. Способен возвращать список тегов
            объектов выборки.

        update_vocab : bool, default=True
            Флаг режима добавления слов в словарь (признаковое пространство) во время обучения.

        return_train_loss : bool, default=False
            Флаг сохранения значений функции потерь для каждого примера из обучающей выборки.

        Returns
        -------
        self : object
            Возвращает объект класса
        """
        self.loss_ = []

        for (word_sentence, sample_tags) in zip(datasource, labels_datasource):

            word_sentence = word_sentence.split(' ')
            sample_tags = set(sample_tags)

            # отбор только известных тегов
            sample_tags = sample_tags & self.tags_

            if len(sample_tags) == 0:
                continue

            # словарь z и значение функции потерь для текущего примера
            z_dict = dict()
            sample_loss = 0

            # посчитаем линейные комбинации весов и признаков z для каждого тега
            for tag in self.tags_:

                # инициализируем z смещением
                z_dict[tag] = self.w0_[tag]

                # чтобы не пробегать словарь на каждой строчке в процессе обучения, линейная
                # комбинация весов и признаков z рассчитывается как сумма весов модели для
                # каждого встреченного слова (если слово втречалось несколько раз, то и в
                # итоговую сумму его вес войдет такое же число раз; для остальных слов из
                # словаря вхождений не будет)
                for word in word_sentence:

                    # обработка слова не из словаря
                    if word not in self.vocab_:
                        if update_vocab:
                            self.vocab_[word] = len(self.vocab_)
                        else:
                            continue

                    z_dict[tag] += self.w_[tag][self.vocab_[word]]

                    # обновление частотного словаря
                    if self.store_frequency_ & update_vocab:
                        self.train_frequency_dict_[self.vocab_[word]] += 1

            # градиентный спуск для каждого тега
            for tag in self.tags_:

                # целевая переменная
                y = int(tag in sample_tags)

                # вычисляем сигмоид (фактически, это вероятность наличия тега)
                sigma = self.get_sigma(z_dict, tag)

                # обновляем значение функции потерь для текущего примера
                sample_loss += self.get_sample_loss(y, sigma)

                # обновим параметры модели

                # вычисляем частную производную функции потерь по текущему весу;
                # учет xm будет реализовываться в цикле далее
                dHdw = (sigma - y)

                # делаем градиентный шаг и выполняем регуляризацию;
                # в целях увеличения производительности делаем допущение для регуляризации;
                # чтобы в каждой итерации обучения не выполнять регуляризацию всех параметров,
                # будем учитывать только присутствующие в текущем обучающем примере признаки;
                # естественно, каждый признак должен быть регуляризован только один раз, не
                # учитывая число его вхождений в обучающий пример;
                # будем выполнять регуляризацию во время первого появления признака

                regularized_words = set()

                for word in word_sentence:

                    if word not in self.vocab_:
                        continue

                    regularization = 0.0

                    if self.lmbda_ > 0.0:
                        if self.vocab_[word] not in regularized_words:
                            regularized_words.add(self.vocab_[word])
                            w = self.w_[tag][self.vocab_[word]]
                            regularization = self.lmbda_ * (2 * self.gamma_ * w + (1 - self.gamma_) * np.sign(w))

                    # корректировка веса значением антиградиента;
                    # явное указание 1.0 показывает, что мы не забыли множитель xm
                    self.w_[tag][self.vocab_[word]] -= self.learning_rate_ * (1.0 * dHdw + regularization)

                # смещение не регуляризируется
                self.w0_[tag] -= self.learning_rate_ * 1.0 * dHdw

            # функция потерь для текущего примера
            self.loss_.append(sample_loss)

        return self

    def get_sigma(self, z_dict, tag):
        """Calculate sigmoid
        Вычисление сигмоида по линейной комбинации весов и признаков.

        Parameters
        ----------
        z_dict : {string: float}
            Словарь значений линейной комбинации весов и признаков для тегов.

        tag : string
            Текущий тег.

        Returns
        -------
        float
            Возвращает значение сигмоида.
        """

        # чтобы не столкнуться с overflow, избегаем вычисления экспоненты
        # с очень большим по модулю положительным аргументом
        if self.strategy_ == 'ovr':

            z = z_dict[tag]
            if z >= 0:
                return 1 / (1 + np.exp(-z))
            else:
                return 1 - 1 / (1 + np.exp(z))

        elif self.strategy_ == 'multinomial':

            np_z = np.array(list(z_dict.values()))
            max_z = np.max(np_z)
            np_z = np_z - max_z

            return np.exp(z_dict[tag] - max_z) / np.sum(np.exp(np_z))

    def get_sample_loss(self, y, sigma):
        """Calculate sample loss comonent
        Вычисление слагаемого функции потерь.

        Parameters
        ----------
        y : int
            Признак наличия тега (0, 1).

        sigma:
            Значение функции сигмоида.

        Returns
        -------
        float
            Возвращает значение слагаемого функции потерь
        """

        # чтобы не получить потери точности, избегаем вычисления логарифма с
        # близким к 0 или 1 аргументом, используя порог tolerance

        if self.strategy_ == 'ovr':

            if y == 1:
                return -1 * np.log(np.max([sigma, self.tolerance_]))
            else:
                return -1 * np.log(1 - np.min([1 - self.tolerance_, sigma]))

        elif self.strategy_ == 'multinomial':

            if y == 1:
                return -1 * np.log(np.max([sigma, self.tolerance_]))
            else:
                return 0.0

    def filter_vocab(self, top=10000):
        """ Filtering vocabulary by the top-n words (for all classes)
        Отбор топ-n самых популярных слов в словаре (для всех тегов)

        Parameters
        ----------
        top : int, default=10000
            количество слов для отбора

        Returns
        -------
        self : object
            Возвращает объект класса
        """
        if not self.store_frequency_:
            print('can\'t filter vocabulary case no frequency data')
            return

        sorted_frequency_dict = sorted(self.train_frequency_dict_.items(), key=lambda x: x[1], reverse=True)
        top_words = {k for (idx, (k, v)) in enumerate(sorted_frequency_dict) if idx < top}

        # обновим словарь
        self.vocab_ = {key: val for (key, val) in self.vocab_.items() if val in top_words}

        # обновим словари весов для тегов
        for tag in self.tags_:
            self.w_[tag] = {key: val for (key, val) in self.w_[tag].items() if key in top_words}

        return self

    def predict_proba(self, datasource):
        """
        Предсказание тегов для текста

        Parameters
        ----------
        datasource : iterable
            Итерируемый объект как источник данных. Способен возвращать объекты выборки
            (классифицируемые строки).

        Yields
        ------
        iterable
            Итерируемый объект. Способен возвращать список кортежей вида (<тег>, <вероятность>)
        """
        for line in datasource:

            sentence = line.strip().split(' ')
            line_predicted = []

            # словарь z для текущего примера
            z_dict = dict()

            # посчитаем линейные комбинации весов и признаков z для каждого тега
            for tag in self.tags_:

                z_dict[tag] = self.w0_[tag]

                for word in sentence:
                    if word not in self.vocab_:
                        continue
                    z_dict[tag] += self.w_[tag][self.vocab_[word]]

            # вычисляем вероятность наличия тега
            for tag in self.tags_:
                line_predicted.append((tag, self.get_sigma(z_dict, tag)))

            yield line_predicted

    def score(self, datasource, labels_datasource, threshold=0.5):
        """ Returns the mean Jaccard index on the given dataset and labels
        Возвращает метрику качества (средний коэффициент Жаккара) по выборке с
        указанными метками

        Parameters
        ----------
        datasource : iterable
            Итерируемый объект как источник данных. Способен возвращать объекты выборки
            (классифицируемые строки).

        labels_datasource : iterable
            Итерируемый объект как источник данных. Способен возвращать список тегов
            объектов выборки.

        threshold : float, dafault=0.5
            Порог предсказания тега (метки класса)

        Returns
        -------
        float
            Средний коэффициент Жаккара по выборке
        """
        scores = []

        for (predicted_probs, true_labels) in zip(self.predict_proba(datasource), labels_datasource):

            predicted_labels = [pp_tuple[0] for pp_tuple in predicted_probs if pp_tuple[1] > threshold]

            jaccard_score = (len(set(predicted_labels) & set(true_labels)) /
                             len(set(predicted_labels) | set(true_labels)))

            scores.append(jaccard_score)

        return np.mean(scores)
