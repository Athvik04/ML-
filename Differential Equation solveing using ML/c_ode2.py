#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 11:35:18 2018

@author: raghav
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 19:15:10 2018

@author: raghav
"""

import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad
from autograd import jacobian
import autograd.numpy.random as npr

nx=10
dx=2./nx

def A(x):
    return 1./5.
def B(x):
    return np.exp(-x/5)*np.cos(x)
def f(x, psy):
    return B(x) - A(x)*psy

def psy_analytic(x):
    return np.exp(-x/5)*np.sin(x)

X=np.linspace(0,2,nx)
Y=psy_analytic(X)

#FDM

def fun1(x,psy,z):
    return z

def fun2(x,psy,z):
    return (-1./5.)*np.exp(-x/5)*np.cos(x) - 1./5. * z - psy

psy_fd=np.zeros((nx))
z=np.zeros((nx))
#IC
psy_fd[0]=0
z[0]=1

for i in range(1,nx):
    psy_fd[i]=psy_fd[i-1] + fun1(X[i-1],psy_fd[i-1],z[i-1])*dx
    z[i]=z[i-1] + fun2(X[i-1],psy_fd[i-1],z[i-1])*dx

#ANN
def sigmoid(x):
    return 1./(1+np.exp(-x))

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

w=[np.random.uniform(low=-0.5, high=0.5, size=(1,10)),np.random.uniform(low=-0.5, high=0.5, size=(10,1))]
lmb=0.001
for i in range(1000):
    loss_grad=grad(loss)(w,X)
    w[0]=w[0] - lmb*loss_grad[0]
    w[1]=w[1] - lmb*loss_grad[1]
    
res=[xi*np.sin(1)*np.exp(-1./5)+xi*(1-xi)*neuralnet(w,xi)[0][0] for xi in X]

plt.figure()
plt.plot(X,Y,'red')
plt.plot(X,psy_fd,'blue')
plt.plot(X,res,'green')
plt.show()