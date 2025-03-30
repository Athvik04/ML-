#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 16:07:03 2018

@author: raghav
"""


import autograd.numpy as np
from autograd import grad, jacobian
from matplotlib import pyplot as plt
from matplotlib import cm
import autograd.numpy.random as npr
from mpl_toolkits.mplot3d import Axes3D

nx=10
ny=10
dx=1./10
dy=1./10
a=3

X=np.linspace(0,1,nx)
Y=np.linspace(0,1,ny)

#Actual solution
def actual(x,y):
    return np.exp(-(a*x+4)/5)*np.sin(a**2*x**2+y)

surface=np.zeros((nx,ny))

for i,x in enumerate(X):
    for j,y in enumerate(Y):
        surface[i][j]=actual(x,y)
        
       
fig=plt.figure()
ax=fig.gca(projection='3d')
x,y=np.meshgrid(X, Y)
surf= ax.plot_surface (x, y, surface, rstride=1, cstride=1, cmap=cm.viridis, linewidth=0, antialiased=False)

ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_zlim(0,2)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

#ANN solution
def f(x,psy,dpsy):
    a1=np.exp(-(a*x[0]+x[1])/5)
    a2=(-4/5 * a**3*x[0] - 2/5 + 2*a**2)*np.cos(a**2*x[0]**2+x[1])
    a3=(1/25 - 1 - 4*a**4*x[0]**2 + a**2/25)*np.sin(a**2*x[0]**2+x[1])
    return a1*(a2 + a3)

#a=3 taken
def A(x):
    a1=(1-x[0])*np.exp(-4/5)*np.sin(x[1]) + x[0]*np.exp(-7/5)*np.sin(9+x[1])
    a2=(1-x[1])*(np.exp(-(3*x[0]+x[1])/5)*np.sin(9*x[0]**2)*x[0]*np.exp(-7/5)*np.sin(9))
    a3=x[1]*(np.exp(-(3*x[0]+4)/5)*np.sin(9*x[0]**2+1)*((1-x[0]*np.exp(-4/5)*np.sin(1)+x[0]*np.exp(-7/5)*np.sin(10))))
    return a1+a2+a3

def psy_trial(x,net_out):
    return A(x) + x[0]*(1-x[0])*x[1]*(1-x[1])*net_out

def sigmoid(x):
    return 1./(1. + np.exp(-x))

def neuralnet(x,w):
    a1=sigmoid(np.dot(x,w[0]))
    return np.dot(a1,w[1])

def loss_fun(X, Y, w):
    loss_sum=0
    
    for xi in X:
        for yi in Y:
            in_pnt=np.array([xi,yi])
            net_out=neuralnet(in_pnt, w)[0]
            
            psy_t=psy_trial(in_pnt, net_out)
            dy_psy_t=grad(psy_trial)(in_pnt, net_out)
            hessian_psy_t=jacobian(jacobian(psy_trial))(in_pnt, net_out)
            
            grad_psy_x=hessian_psy_t[0][0]
            grad_psy_y=hessian_psy_t[1][1]
            func=f(in_pnt, psy_t, dy_psy_t)
            err=(grad_psy_x + grad_psy_y - func)**2
            loss_sum = loss_sum + err
    return loss_sum

w=[npr.randn(2,10),npr.randn(10,1)]
lmb=0.001

for i in range(100):
    loss_grad=grad(loss_fun)(X, Y, w)
    w[0]=w[0] - lmb*loss_grad[0]
    w[1]=w[1] - lmb*loss_grad[1]
    
surface2=np.zeros((nx,ny))

for i,x in enumerate(X):
    for j,y in enumerate(Y):
        net_out=neuralnet([x,y],w)
        surface2[i][j]=psy_trial([x,y],net_out)
              
fig=plt.figure()
ax=fig.gca(projection='3d')
x,y=np.meshgrid(X, Y)
surf= ax.plot_surface (x, y, surface2, rstride=1, cstride=1, cmap=cm.viridis, linewidth=0, antialiased=False)

ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_zlim(0,2)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')