#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 14:29:44 2018

@author: raghav
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#dataset_ann=pd.read_csv("ode_ann.csv")
#x_ann=dataset_ann.iloc[:, 1].values
#y_ann=dataset_ann.iloc[:, 2].values
#
#dataset_fdm=pd.read_csv("ode_fdm.csv")
#x_fdm=dataset_fdm.iloc[:, 1].values
#y_fdm=dataset_fdm.iloc[:, 2].values

dataset_fdm1=pd.read_csv("ode_time_fdm1.csv")
x_time_fdm1=dataset_fdm1.iloc[:, 1].values
y_time_fdm1=dataset_fdm1.iloc[:, 2].values

dataset_ann1=pd.read_csv("ode_time_ann1.csv")
x_time_ann1=dataset_ann1.iloc[:, 1].values
y_time_ann1=dataset_ann1.iloc[:, 2].values

dataset_fdm2=pd.read_csv("ode_time_fdm2.csv")
x_time_fdm2=dataset_fdm2.iloc[:, 1].values
y_time_fdm2=dataset_fdm2.iloc[:, 2].values

dataset_ann2=pd.read_csv("ode_time_ann2.csv")
x_time_ann2=dataset_ann2.iloc[:, 1].values
y_time_ann2=dataset_ann2.iloc[:, 2].values

dataset_fdm3=pd.read_csv("ode_time_fdm3.csv")
x_time_fdm3=dataset_fdm3.iloc[:, 1].values
y_time_fdm3=dataset_fdm3.iloc[:, 2].values

dataset_ann3=pd.read_csv("ode_time_ann3.csv")
x_time_ann3=dataset_ann3.iloc[:, 1].values
y_time_ann3=dataset_ann3.iloc[:, 2].values

dataset_fdm4=pd.read_csv("ode_time_fdm4.csv")
x_time_fdm4=dataset_fdm4.iloc[:, 1].values
y_time_fdm4=dataset_fdm4.iloc[:, 2].values

dataset_ann4=pd.read_csv("ode_time_ann4.csv")
x_time_ann4=dataset_ann4.iloc[:, 1].values
y_time_ann4=dataset_ann4.iloc[:, 2].values

x_time_ann=np.zeros((len(x_time_ann1)))
y_time_ann=np.zeros((len(x_time_ann1)))
x_time_fdm=np.zeros((len(x_time_ann1)))
y_time_fdm=np.zeros((len(x_time_ann1)))

for i in range(len(x_time_ann1)):
    x_time_ann[i]=(x_time_ann1[i] + x_time_ann2[i] + x_time_ann3[i] + x_time_ann4[i])/4
    y_time_ann[i]=(y_time_ann1[i] + y_time_ann2[i] + y_time_ann3[i] + y_time_ann4[i])/4
    x_time_fdm[i]=(x_time_fdm1[i] + x_time_fdm2[i] + x_time_fdm3[i] + x_time_fdm4[i])/4
    y_time_fdm[i]=(y_time_fdm1[i] + y_time_fdm2[i] + y_time_fdm3[i] + y_time_fdm4[i])/4

plt.figure()
plt.plot(x_time_ann, y_time_ann, 'green')
plt.plot(x_time_fdm, y_time_fdm, 'blue')
plt.show()

#actual solution
nx=10
dx=2./nx

def psy_analytic(x):
    return np.exp(-x/5.)*np.sin(x)

X=np.linspace(0,2,nx)
Y=psy_analytic(X)

plt.figure()
plt.plot(x_ann, y_ann, 'green')
plt.plot(x_fdm, y_fdm, 'blue')
plt.plot(X, Y, 'red')
plt.show()