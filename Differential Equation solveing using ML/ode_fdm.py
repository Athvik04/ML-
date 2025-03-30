#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 13:06:38 2018

@author: raghav
"""

import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad
import autograd.numpy.random as npr
import pandas as pd
import time

nx=0
time_array=np.zeros((400,2))
for j in range(400):
#time array and mesh
    time1=time.time()
    nx=nx+10
    dx=2./nx
    
    X=np.linspace(0,2,nx)
    
    #FDM BC
    def fun(x):
        return -1./5 * np.exp(-x/5)*np.cos(x)
    psy_fd=np.zeros((nx))
    psy_fd[0]=0
    psy_fd[nx-1]=np.sin(1)*np.exp(-1./5)
    
    sol=np.zeros((nx))
    A1=np.zeros((nx,nx))
    A1[0][0]=1
    A1[nx-1][nx-1]=1
    for i in range(1,nx-1):
        A1[i][i-1]=1
        A1[i][i]=-2-dx/5+dx**2
        A1[i][i+1]=1+dx/5
        psy_fd[i]=fun(X[i])*dx**2
        
    invA=np.linalg.inv(A1)
    
    sol=np.dot(invA, psy_fd)
    array=np.zeros((nx,2))
    for i in range(nx):
        array[i][0]=X[i]
        array[i][1]=sol[i]
    
    time2=time.time()
    time_array[j][0]=nx
    time_array[j][1]=time2-time1

pd.DataFrame(time_array).to_csv("ode_time_fdm4.csv")  

plt.figure()
plt.plot(X,sol,'blue')
plt.show()