import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data_train = pd.read_csv('train.csv', delimiter=',')
data_test = pd.read_csv('test.csv', delimiter=',')

data_train = data_train.dropna()
data_test = data_test.dropna()

train, validation = train_test_split(data_train, test_size=0.2)

print(train.columns[train.isnull().values.any()].tolist())

arr_name = []
arr_train = []
arr_val = []

cols_x = ['Pclass','Sex', 'Age','SibSp','Parch','Fare']   

col_y = 'Survived'

def test_classifier(classifier, classifier_name):
    # обучаем классификатор
    classifier.fit(train[cols_x], train[col_y])

    # проверяем классификатор
    y_train = classifier.predict(train[cols_x])
    
    # определяем точность
    y_train_acc = accuracy_score(train[col_y], y_train)
    # для валидационной выборки
    y_val = classifier.predict(validation[cols_x])
    y_val_acc = accuracy_score(validation[col_y], y_val)

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

classifier = test_classifier(LogisticRegression(), 'LR')

classifier = test_classifier(SVC(), 'SVM')

classifier = test_classifier(DecisionTreeClassifier(), 'Tree')
for col, i in zip(cols_x, classifier.feature_importances_):
    print(col, ": ", i)

classifier = test_classifier(GradientBoostingClassifier(), 'GB')
for col, i in zip(cols_x, classifier.feature_importances_):
    print(col, ": ", i)

x = range(len(arr_train))
plt.plot(x, arr_train)
plt.plot(x, arr_val)
plt.xticks(x, arr_name)
plt.ylabel('Точность алгоритма')
plt.legend(['обучение', 'валидация'], loc='lower left')
plt.show(block = False)


cols_x = ['Pclass','Sex', 'Age','SibSp','Parch','Fare']   
col_y = 'Survived'

clf = DecisionTreeClassifier()
clf.fit(data_train[cols_x], data_train[col_y])

occupancy_predicted = clf.predict(data_test[cols_x])

data_test['value'] = occupancy_predicted
data_test.to_csv('base_solution.csv')

print(data_test.head())