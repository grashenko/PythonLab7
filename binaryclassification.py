import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

data_train = pd.read_csv('train.csv', delimiter=',')
data_test = pd.read_csv('train.csv', delimiter=',')

data_train =data_train.dropna()
data_test = data_test.dropna()

print(data_train.head())
print(data_test.head())


print(data_train.columns[data_train.isnull().values.any()].tolist())

arr_name = []
arr_train = []
arr_val = []

cols_x = ['Pclass','Sex', 'Age','SibSp','Parch','Fare']   

col_y = 'Survived'

def test_classifier(classifier, classifier_name):
    # обучаем классификатор
    classifier.fit(data_train[cols_x], data_train[col_y])

    # проверяем классификатор
    y_train = classifier.predict(data_train[cols_x])
    
    # определяем точность
    y_train_acc = accuracy_score(data_train[col_y], y_train)
    # для валидационной выборки
    y_val = classifier.predict(data_test[cols_x])
    y_val_acc = accuracy_score(data_test[col_y], y_val)

    # сохранение информации в массивы
    arr_name.append(classifier_name)
    arr_train.append(y_train_acc)
    arr_val.append(y_val_acc)
    
    # вывод промежуточных результатов
    print('Точность для алгоритма {} на обучающей выборке = {}, \
    на валидационной выборке = {}'\
          .format(classifier_name,\
                  round(y_train_acc, 3),\
                  round(y_val_acc, 3)))
    
    # возвращаем обученный классификатор
    return classifier


classifier = test_classifier(KNeighborsClassifier(), 'KNN')