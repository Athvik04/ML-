#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 12:26:53 2018

@author: raghav
"""

import autograd.numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from autograd import grad
import autograd.numpy.random as npr
from autograd.core import primitive

nx=10
dx=1./nx

def A(x):
    return x+(1.+3.*x**2)/(1.+x+x**3)

def B(x):
    return x**3 + 2.*x + (x**2)*((1. + 3.*x**2)/(1. + x+x**3))

def f(x,psy):
    return B(x)-A(x)*psy

def psy_analytic(x):
    return (np.exp((-x**2)/2.))/(1.+x+x**3)+x**2

X=np.linspace(0,1,nx)
Y=psy_analytic(X)
psy_fd=np.zeros_like(Y)
psy_fd[0]=1.

for i in range(1,len(X)):
    psy_fd[i]=psy_fd[i-1]+B(X[i])*dx-A(X[i])*psy_fd[i-1]*dx
    
def sigmoid(x):
    return 1./(1.+np.exp(-x))

def sigmoid_grad(x):
    return sigmoid(x)*(1-sigmoid(x))

def nn(w,x):
    a1=sigmoid(np.dot(x, w[0]))
    return np.dot(a1, w[1])

def nn_grad(w,x,k=1):
    return np.dot(np.dot(w[1].T,w[0].T**k),sigmoid_grad(x))

def loss(w,x):
    loss_sum=0.
    for xi in x:
        net_out=nn(w,xi)[0][0]
        psy_t=1.+xi*net_out
        d_net_out=nn_grad(w,xi)[0][0]
        err_sqr=(net_out + xi*d_net_out - f(xi,psy_t))**2
        
        loss_sum=loss_sum+err_sqr
    return loss_sum
        
w=[npr.randn(1,10),npr.randn(10,1)]
lmb=0.001

for i in range(1000):
    loss_grad= grad(loss)(w,X)
    w[0]=w[0]-lmb*loss_grad[0]
    w[1]=w[1]-lmb*loss_grad[1]

print(loss(w,X))
res=[1 + xi*nn(w,xi)[0][0] for xi in X]

print(w)
        
plt.figure()    
plt.plot(X,Y,'red')
plt.plot(X,psy_fd,'blue')
plt.plot(X,res,'green')
plt.show()