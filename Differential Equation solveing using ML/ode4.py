#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 11:08:20 2018

@author: raghav
"""

import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad
import autograd.numpy.random as npr

nx=10
dx=2./nx
#actual
def actual(x):
    return np.exp(-x/5)*np.sin(x)

X=np.linspace(0,2,nx)
Y=actual(X)

def f(x):
    return np.exp(-x/5)*np.cos(x)

#FDM
psy_fd=np.zeros((nx))
A=np.zeros((nx,nx))
B=np.zeros((nx))
B[0]=0

for i in range(1,nx):
    A[i][i-1]=-1
    A[i][i]=1+dx/5
    B[i]=f(X[i])*dx
    
invA=np.linalg.inv(A)
psy_fd=np.dot(invA,B)

plt.figure()
plt.plot(X,Y,'red')
plt.plot(X,psy_fd,'blue')
plt.show()
