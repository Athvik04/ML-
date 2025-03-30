#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 16:09:13 2018

@author: raghav
"""


import autograd.numpy as np
from autograd import grad,jacobian
import autograd.numpy.random as npr
from matplotlib import pyplot as plt
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
import time

nx=10
ny=10

X=np.linspace(0,1,nx)
Y=np.linspace(0,1,ny)

def analytic(x,y):
    return np.exp(-x)*(x + y**3)

surface=np.zeros((ny,nx))

for i, x in enumerate(X):
    for j, y in enumerate(Y):
        surface[i][j]=analytic(x,y)
        
fig=plt.figure()
ax=fig.gca(projection='3d')
x,y=np.meshgrid(X, Y)
surf= ax.plot_surface (x, y, surface, rstride=1, cstride=1, cmap=cm.viridis, linewidth=0, antialiased=False)

ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_zlim(0,2)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

time1=time.time()

def f(x, y):
    if x==0:
        return y**3
    if x==1:
        return (1+y**3)*np.exp(-1)
    if y==0:
        return x*np.exp(-x)
    if y==1:
        return np.exp(-x)*(x+1)

def sigmoid(x):
    return 1./(1. + np.exp(-x))

def neuralnet(x,w):
    a=sigmoid(np.dot(x,w[0]))
    return np.dot(a,w[1])

def A(x):
    return (1-x[0])*x[1]**3 + x[0]*(1 + x[1]**3)*np.exp(-1) + (1-x[1])*x[0]*(np.exp(-x[0])-np.exp(-1)) + x[1]*((1+x[0])*np.exp(-x[0]) - 1+x[0]-2*x[0]*np.exp(-1))

def psy_trial(x, net_out):
    return A(x) + x[0]*(1-x[0])*x[1]*(1-x[1])*net_out

def loss_func(X, Y, w):
    loss_sum=0.
    
    for xi in X:
        for yi in [0,1]:
            input_pnt=np.array([xi,yi]) 
            
            net_out=neuralnet(input_pnt,w)[0]
            
            psy_t=psy_trial(input_pnt, net_out)
            psy_t_jacobian=jacobian(psy_trial)(input_pnt, net_out)   
            psy_t_hessian=jacobian(jacobian(psy_trial))(input_pnt, net_out)
            
            gradient_t_x=psy_t_hessian[0][0]
            gradient_t_y=psy_t_hessian[1][1]
            
            func=f(xi, yi)
            
            err=(gradient_t_x + gradient_t_y - func)**2
            loss_sum=loss_sum+err
      
    for xi in [0,1]:
        for yi in Y:
            input_pnt=np.array([xi,yi]) 
            
            net_out=neuralnet(input_pnt,w)[0]
            
            psy_t=psy_trial(input_pnt, net_out)
            psy_t_jacobian=jacobian(psy_trial)(input_pnt, net_out)   
            psy_t_hessian=jacobian(jacobian(psy_trial))(input_pnt, net_out)
            
            gradient_t_x=psy_t_hessian[0][0]
            gradient_t_y=psy_t_hessian[1][1]
            
            func=f(xi, yi)
            
            err=(gradient_t_x + gradient_t_y - func)**2
            loss_sum=loss_sum+err      
    return loss_sum

w=[np.random.uniform(low=-0.5, high=0.5, size=(2,10)),np.random.uniform(low=-0.5, high=0.5, size=(10,1))]
lmb=0.001

for i in range(10):
    loss_grad=grad(loss_func)(X,Y,w) 
    w[0] = w[0] - lmb*loss_grad[0]
    w[1] = w[1] - lmb*loss_grad[1]

time2=time.time()
    
surface2=np.zeros((ny,nx))

for i,xi in enumerate(X):
    for j,yi in enumerate(Y):
        net_outt=neuralnet([xi,yi],w)
        surface2[i][j]=psy_trial([xi,yi],net_outt)
      
time3=time.time()        
        
fig=plt.figure()
ax=fig.gca(projection='3d')
x,y=np.meshgrid(X, Y)
surf= ax.plot_surface (x, y, surface2, rstride=1, cstride=1, cmap=cm.viridis, linewidth=0, antialiased=False)

ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_zlim(0,2)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')


train_time=time2-time1
test_time=time3-time2
print("Train time:",train_time)
print("Test time:",test_time)