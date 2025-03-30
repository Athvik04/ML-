#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 16:05:55 2018

@author: raghav
"""

import autograd.numpy as np
from autograd import grad,jacobian
import autograd.numpy.random as npr
from matplotlib import pyplot as plt
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D

nx=10
ny=10

dx=1./(nx-1)
dy=1./(ny-1)

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


def f(x, y):
    return np.exp(-x)*(x - 2 + y**3 + 6*y)

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

w=[npr.randn(2, 10),npr.randn(10, 1)]
lmb=0.001

for i in range(100):
    loss_grad=grad(loss_func)(X,Y,w)
    w[0] = w[0] - lmb*loss_grad[0]
    w[1] = w[1] - lmb*loss_grad[1]
    
surface2=np.zeros((ny,nx))

for i,xi in enumerate(X):
    for j,yi in enumerate(Y):
        net_outt=neuralnet([xi,yi],w)
        surface2[i][j]=psy_trial([xi,yi],net_outt)
      
fig=plt.figure()
ax=fig.gca(projection='3d')
x,y=np.meshgrid(X, Y)
surf= ax.plot_surface (x, y, surface2, rstride=1, cstride=1, cmap=cm.viridis, linewidth=0, antialiased=False)

ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_zlim(0,2)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')  

import time
time1=time.time()

#fdm
def fun(x,y):
    return np.exp(-x)*(x-2+y**3+6*y)

n=nx*ny
A1=np.zeros((n,n))
psy_fd=np.zeros((n))
B=np.zeros((n))
 
for i in range(nx):
    A1[i][i]=1
    B[i]=X[i]*np.exp(-X[i])
    A1[nx*(ny-1)+i-1][nx*(ny-1)+i-1]=1
    B[nx*(ny-1)+i]=np.exp(-1*X[i])*(X[i]+1)

for i in range(ny):
    A1[nx*i][nx*i]=1
    B[nx*i]=Y[i]**3
    A1[nx*i+nx-1][nx*i+nx-1]=1
    B[nx*i+nx-1]=(1+Y[i]**3)*np.exp(-1)
   
for i in range(nx,nx*ny-nx):
    if not(i%nx) is 0 and not((i+1)%ny) is 0:
        A1[i][i-nx]=1
        A1[i][i+nx]=1
        A1[i][i-1]=1
        A1[i][i+1]=1
        A1[i][i]=-4
        j=i%nx
        k=(i-j)//nx
        B[i]=fun(X[j],Y[k])*dx**2

invA1=np.linalg.inv(A1)
psy_fd=np.dot(invA1, B)

surface3=np.zeros((nx,ny))
for i in range(nx*ny):
    a=i%nx
    b=(i-a)//ny
    surface3[a][b]=psy_fd[i]

time2=time.time()

fig=plt.figure()
ax=fig.gca(projection='3d')
x,y=np.meshgrid(X, Y)
surf= ax.plot_surface (x, y, surface3, rstride=1, cstride=1, cmap=cm.viridis, linewidth=0, antialiased=False)

ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_zlim(0,2)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')        