# mlda_tryouts
Tryouts in machine learning and data analysis.

* [phones_sentiment](./phones_sentiment)

  Docker-based sentiment analysis (data gathering, model fitting, web-ui).<br>
  Docker-решение анализа тональности текста (сбор данных, обучение, web-ui).<br>

* ml_models_implementation

  Machine learning models implementation.
  
  - [decision_tree](./ml_models_implementation/decision_tree)<br>
  Decision tree classification and regression model.<br>
  Реализованная модель дерева решений, решающая задачи классификации и регрессии с поддержкой пропусков в данных. В качестве тестирования модели используются сгенерированные датасеты scikit-learn, выполняется сравнение с моделями DecisionTreeClassifier и DecisionTreeRegressor модуля sklearn.tree. [Описание задачи](./ml_models_implementation/decision_tree/description.ipynb).
  
  - [mlmc_online_logreg](./ml_models_implementation/mlmc_online_logreg)<br>
  Multilabel/multiclass online logistic regression model.<br>
  Реализованная по мотивам задания курса OpenDataScience модель multilabel/multiclass классификации, выполняющая онлайн-обучение (стохастический градиентный спуск с логистической функцией потерь). Рассматривается задача классификации текстов. В качестве примера данных используется датасет вопросов на stackoverflow, задачей стоит предсказание тегов вопросов. Реализован механизм формирования малых датасетов для дообучения. [Описание задачи](./ml_models_implementation/mlmc_online_logreg/description.ipynb).

* [brazil_rains](./brazil_rains)

  OpenDataScience capstone project. Rains prediction in Brazil. Work in progress!<br>
  Индивидуальный проект на курсе OpenDataScience (весна 2018). Сдан был частично, работа над проектом продолжается.

