#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 16:58:48 2018

@author: raghav
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
n=1000
X=[[0 for x in range(2)] for y in range(n)]
x1=np.linspace(0,100,n)
x2=np.linspace(0,100,n)
for i in range(0,n):
    X[i][0],X[i][1]=int(x1[i]**2),int(x2[i]**2)
X=np.reshape(X,(n,2))
y=np.array([0. for i in range(n)])
for i in range(n):
    y[i]=int(X[i][0]) + int(X[i][1])
y=np.reshape(y,(n,1))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

scy=StandardScaler()
y_train=scy.fit_transform(y_train)
y_test=scy.transform(y_test)

#from sklearn.linear_model import LinearRegression
#regg = LinearRegression()
#regg.fit(X_train,y_train)
#
#y_pred= regg.predict(X_test)
#
#out=regg.predict((np.array([[20,20]])))

import keras
from keras.models import Sequential
from keras.layers import Dense 
from keras.layers import Dropout

classifier = Sequential()
classifier.add(Dense(units = 3, activation='tanh',kernel_initializer='uniform',input_dim=2))
classifier.add(Dense(units = 3, activation='tanh',kernel_initializer='uniform'))
classifier.add(Dense(units = 3, activation='linear',kernel_initializer='uniform'))
classifier.add(Dense(units = 3, activation='tanh',kernel_initializer='uniform'))
classifier.add(Dense(units = 3, activation='tanh',kernel_initializer='uniform'))

classifier.add(Dense(units = 1, activation='linear',kernel_initializer='uniform'))

classifier.compile(optimizer='Adagrad',loss='mean_absolute_error',metrics=['accuracy'])

classifier.fit(X_train, y_train,batch_size=10,epochs=200)

y_pred = classifier.predict(X_test)
output=classifier.predict(sc.transform(np.array([[5,5]])))
output=scy.inverse_transform(np.array(output))

#orgout=scy.inverse_transform(y_pred)
#orgout2=scy.inverse_transform(y_test)
#
#plt.scatter(X_test[:,0],orgout, color='blue')
#plt.scatter(X_test[:,0],orgout2,color='red')
