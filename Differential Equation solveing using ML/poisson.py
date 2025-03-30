#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 14:20:49 2018

@author: raghav
"""

import autograd.numpy as np
from autograd import grad, jacobian
from matplotlib import pyplot as plt
from matplotlib import cm
import autograd.numpy.random as npr
from mpl_toolkits.mplot3d import Axes3D

nx=20
ny=20

X=np.linspace(0,1,nx)
Y=np.linspace(0,1,ny)

def fun(x):
    if x[0]==0:
        return x[1]
    if x[0]==1:
        return 0.6+x[1]

def sigmoid(x):
    return 1./(1. + np.exp(-x))

def neuralnet(x,w):
    a1=sigmoid(np.dot(x,w[0]))
    return np.dot(a1,w[1])

def neuralnet_x(x,w):
    inpnt=np.array([1,x[1]])
    a1=sigmoid(np.dot(inpnt,w[0]))
    return np.dot(a1,w[1])


def psy_trial(x,net_out):
    a=neuralnet_x(x,w)[0]
    b=grad(neuralnet_x)(x,w)[1]
    return (1-x[0])*x[1] + x[0]*(x[1]+0.6) + x[0]*(x[0]-1)*(net_out - a -b + 1)

def loss_fun(X, Y, w):
    loss_sum=0.
    
    for xi in [0,1]:
        for yi in Y:
            in_pnt=np.array([xi,yi])
            net_out=neuralnet(in_pnt, w)[0]
            hessian_psy_t=jacobian(jacobian(psy_trial))(in_pnt, net_out)
            
            grad_psy_x=hessian_psy_t[0][0]
            grad_psy_y=hessian_psy_t[1][1]
            func=fun(in_pnt)
            err=(grad_psy_x + grad_psy_y - func)**2
            loss_sum = loss_sum + err
    return loss_sum

w=[np.random.uniform(low=-0.5, high=0.5, size=(2,10)),np.random.uniform(low=-0.5, high=0.5, size=(10,1))]
lmb=0.001

for i in range(100):
    loss_grad=grad(loss_fun)(X, Y, w)
    w[0]=w[0] - lmb*loss_grad[0]
    w[1]=w[1] - lmb*loss_grad[1]

surface=np.zeros((nx,ny))

for i,x in enumerate(X):
    for j,y in enumerate(Y):
        net_out=neuralnet([x,y],w)
        surface[i][j]=psy_trial([x,y],net_out)
              
fig=plt.figure()
ax=fig.gca(projection='3d')
x,y=np.meshgrid(X, Y)
surf= ax.plot_surface (x, y, surface, rstride=1, cstride=1, cmap=cm.viridis, linewidth=0, antialiased=False)

ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_zlim(0,2)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')