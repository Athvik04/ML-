#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 18:22:09 2018

@author: raghav
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy.interpolate import interp1d

dataset = pd.read_csv('ode_fdm.csv')
X = dataset.iloc[:, 1].values
y = dataset.iloc[:, 2].values  

n_x=len(X)
X=np.reshape(X,(n_x,1))
n_y=len(y)
y=np.reshape(y,(n_y,1))     

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X=sc.fit_transform(X)       

scy=StandardScaler()
y=scy.fit_transform(y)     

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=None)
classifier=Sequential()

classifier.add(Dense(units = 3, kernel_initializer = 'RandomUniform', activation = 'linear',input_dim = 1))
classifier.add(Dense(units = 3, kernel_initializer = 'RandomUniform', activation = 'tanh'))

classifier.add(Dense(units = 1, kernel_initializer = 'RandomUniform', activation = 'linear'))

classifier.compile(optimizer = 'sgd', loss = 'mean_absolute_error')

classifier.fit(X, y, batch_size = 2, epochs = 8000)
  

#time array and mesh
time_array=np.zeros((400,2))
n=0
for j in range(400):
    time1=time.time()
    n=n+10
    y_pred=np.zeros((n))
    X_range=np.linspace(0,2,n)   
    for i in range(n):
        X_test=sc.transform(np.array([[X_range[i]]]))
        y_pred[i]=(scy.inverse_transform(classifier.predict(X_test)))
    interp=interp1d(X_range, y_pred, kind='cubic')
    
    X_range_new=np.linspace(0,2,n)
    y_pred_new=interp(X_range_new)
    
    time2=time.time()
    time_array[j][0]=n
    time_array[j][1]=time2-time1

pd.DataFrame(time_array).to_csv("ode_time_ann4.csv")

array=np.zeros((10000,2))
for i in range(10000):
    array[i][0]=X_range_new[i]
    array[i][1]=y_pred_new[i]
pd.DataFrame(array).to_csv("ode_ann.csv")  

plt.figure()
plt.plot(X_range, y_pred,'green')
plt.show()

plt.figure()
plt.plot(X_range_new, y_pred_new,'blue')
plt.show()