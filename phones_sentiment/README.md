# Phone reviews sentiment analysis

Проект в рамках специализации [Машинное обучение и анализ данных (Яндекс, МФТИ)](https://www.coursera.org/specializations/machine-learning-data-analysis). Тема - анализ тональности отзывов пользователей. Проект выполнен в виде docker-решения, реализован полный цикл: сбор данных, обучение модели, запуск в использование. Каждую фазу можно запустить отдельно, например, для расшиерния обучающей выборки и переобучения модели.

Сбор данных производится путем парсинга (web scraping) текстов отзывов на мобильные телефоны с сайта [market.yandex.ru](https://market.yandex.ru/). Используемые средства: [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/), [Scrapy](https://scrapy.org/).

В процессе предобработки данных производится лемматизация текста и его векторизация с помощью TF-IDF подхода. Поиск оптимальной модели и ее параметров, а также параметров векторизации текста, производится с помощью библиотеки hyperopt. Используемые средства: [scikit-learn](https://scikit-learn.org/stable/), [pymystem3](https://github.com/nlpub/pymystem3), [hyperopt](https://github.com/hyperopt/hyperopt).

Запуск модели в использование выполнен путем создания веб-страницы с пользовательским интерфейсом. Используемые средства: [Flask](http://flask.pocoo.org/), [Flask-Bootstrap](https://pythonhosted.org/Flask-Bootstrap/).

## Make-команды для работы

`make scrapy-run` - запустить процесс сбора обучающей выборки (web scraping)<br>
`make model-fit` - запустить процесс построения модели и подбора ее параметров<br>
`make ui-run` - запустить пользовательский интерфейс (доступен по адресу http://127.0.0.1/estimator)<br>
`make bash-run` - запустить терминал в Docker-контейнере<br>
`make docker-build` - сборка Docker-образа, используя конфигурационный файл [Dockerfile](./Dockerfile)<br>
`make docker-push` - отправка Docker-образа на DockerHub<br>

## Описание файла настроек

[sentiment.json](./settings/sentiment.json)<br>
`docker/image` - имя docker контейнера на DockerHub<br>
`scrapping/start_page` - адрес стартовой страницы парсинга обучающей выборки<br>
`scrapping/download_delay` - величина задержки (с) отправки запросов в процессе парсинга<br>
`model_fitting/hyperopt_max_evals` - количество итераций поиска оптимальных параметров модели<br>
