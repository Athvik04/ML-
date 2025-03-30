#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 11:50:41 2018

@author: raghav
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 11:32:41 2018

@author: raghav
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 11:52:37 2018

@author: raghav
"""

import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad
import autograd.numpy.random as npr

nx=10
dx=1./nx

def A(x):
    return 1./5
def B(x):
    return np.exp(-x/5)*np.cos(x)
def f(x, psy):
    return B(x) - A(x)*psy

def psy_analytic(x):
    return np.exp(-x/5)*np.sin(x)

X=np.linspace(0,1,nx)
Y=psy_analytic(X)

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

#ANN
def sigmoid(x):
    #softsign
    return x / (abs(x) + 1)


def neuralnet(w,x):
    a1 = sigmoid( np.dot(x, w[0]))
    return np.dot(a1, w[1])

def nn_grad(w,x,k=1):
    return np.dot(np.dot(w[1].T,w[0].T**k),grad(sigmoid)(x))
    
def psy_trial(x,net_out):
    return x*np.sin(1)*np.exp(-1./5)+x*(1-x)*net_out

def loss(w,x):
    loss_sum=0.
    for xi in x:
        net_out = neuralnet(w,xi)[0][0]
        psy_t=psy_trial(xi,net_out)
        d_psy_t=grad(psy_trial)(xi,net_out)
        err=(d_psy_t - f(xi,psy_t))**2
    loss_sum = loss_sum + err
    return loss_sum

w=[np.random.uniform(low=-0.5, high=1, size=(1,10)),np.random.uniform(low=-0.5, high=1, size=(10,1))]
lmb=0.001
for i in range(1000):
    loss_grad=grad(loss)(w,X)
    w[0]=w[0] - lmb*loss_grad[0]
    w[1]=w[1] - lmb*loss_grad[1]
    
res=[xi*np.sin(1)*np.exp(-1./5)+xi*(1-xi)*neuralnet(w,xi)[0][0] for xi in X]

plt.figure()
plt.plot(X,Y,'red')
plt.plot(X,sol,'blue')
plt.plot(X,res,'green')
plt.show()