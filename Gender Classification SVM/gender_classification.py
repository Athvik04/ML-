# support vector classifier

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# df = pd.read_csv('gender_classification_v7.csv')
df = pd.read_csv(r'd:\5th sem\Machiene learning\Gender_Classification_Support_Vector_Machines_SVM-main\Gender_Classification_Support_Vector_Machines_SVM-main\gender_classification_v7.csv')


# 1 being Male and 0 being Female
df['Male'].unique()

df.info()

df.describe().transpose()


X = df.drop('Male', axis=1)
y = df['Male']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)


# C=1.0, kernel='rbf', gamma='scale'
model = SVC()

model.fit(X_train, y_train)

predictions = model.predict(X_test)


print(confusion_matrix(y_test, predictions))

print(classification_report(y_test, predictions))

grid_params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
               'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}


grid = GridSearchCV(SVC(), grid_params, refit=True, verbose=3)

grid.fit(X_train, y_train)

# Check the best parameters
grid.best_params_
grid.best_estimator_

grid_predictions = grid.predict(X_test)

print(confusion_matrix(y_test, grid_predictions))

print(classification_report(y_test, grid_predictions))

test_index = 1

test_dic = {0: 'Female', 1: 'Male'}
y_test_np = np.array(y_test)
print(f'Actual     --> {test_dic[y_test_np[test_index]]
                        }\nPrediction --> {test_dic[predictions[test_index]]}')
