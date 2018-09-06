# mlda_tryouts
Tryouts in machine learning and data analysis.

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

* kaggle_inclass

  Training kaggle competitions (kaggle inclass) for OpenDataScience course.
  
    - [alice](./kaggle_inclass/alice)<br>
    https://www.kaggle.com/c/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2
    
    - [flights](./kaggle_inclass/flights)<br>
    https://www.kaggle.com/c/flight-delays-spring-2018
    
    - [medium](./kaggle_inclass/medium)<br>
    https://www.kaggle.com/c/how-good-is-your-medium-article
    
    - [receipts](./kaggle_inclass/receipts)<br>
    https://www.kaggle.com/c/receipt-categorisation
