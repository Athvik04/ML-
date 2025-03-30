#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 22:30:33 2018

@author: raghav
"""

import autograd.numpy as np
from autograd import grad, jacobian
from matplotlib import pyplot as plt
from matplotlib import cm
import autograd.numpy.random as npr
from mpl_toolkits.mplot3d import Axes3D
import time


nx=10
ny=10

dx=1./(nx-1)
dy=1./(ny-1)

X=np.linspace(0,1,nx)
Y=np.linspace(0,1,ny)

def actual(x,y):
    return y**2*np.sin(np.pi*x)

surface=np.zeros((nx,ny))

for i,x in enumerate(X):
    for j,y in enumerate(Y):
        surface[i][j]=actual(x,y)
            
def f(x,y):
    return (2-np.pi**2*y**2)*np.sin(np.pi*x)

def B(x,y):
    return y*2*np.sin(np.pi*x)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def neuralnet(x,y,w):
    input_pnt=np.array([x,y])
    a=sigmoid(np.dot(input_pnt,w[0]))
    return np.dot(a,w[1])

def psy_trial(x,y,w):
    a=neuralnet(x,y,w)
    b=neuralnet(x,1,w)
    c=grad(neuralnet)(x,1,w)
    return B(x,y) + x*(1-x)*y*(a-b-c)

def psy_trial_x(x):
    a=neuralnet(x[0],x[1],w)[0]
    b=neuralnet(x[0],1,w)[0]
    c=grad(neuralnet)(x[0],1,w)
    return B(x[0],x[1]) + x[0]*(1-x[0])*x[1]*(a-b-c)

def loss_fun(X, Y, w):
    loss_sum=0.
    
    for xi in X:
        for yi in Y:
            inpnt=np.array([xi,yi])
            psy_t_jacobian=jacobian(psy_trial_x)(inpnt)
            psy_t_hessian=jacobian(jacobian(psy_trial_x))(inpnt)
            
            grad_t_x=psy_t_hessian[0][0]
            grad_t_y=psy_t_hessian[1][1]
            
            func = f(xi, yi)
            err=(grad_t_x + grad_t_y - func)**2
            loss_sum=loss_sum+err
    return loss_sum

w=[npr.randn(2,10),npr.randn(10,1)]
lmb=0.001

for i in range(100):
    loss_grad=grad(loss_fun)(X, Y, w)
    w[0] = w[0] - lmb*loss_grad[0]
    w[1] = w[1] - lmb*loss_grad[1]
    
surface2=np.zeros((nx,ny))

for i,x in enumerate(X):
    for j,y in enumerate(Y):
        surface2[i][j]=psy_trial(x,y,w)
        
fig=plt.figure()
ax=fig.gca(projection='3d')
x,y=np.meshgrid(X, Y)
surf= ax.plot_surface (x, y, surface, rstride=1, cstride=1, cmap=cm.viridis, linewidth=0, antialiased=False)

ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_zlim(0,2)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

       
fig=plt.figure()
ax=fig.gca(projection='3d')
x,y=np.meshgrid(X, Y)
surf= ax.plot_surface (x, y, surface2, rstride=1, cstride=1, cmap=cm.viridis, linewidth=0, antialiased=False)

ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_zlim(0,2)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

time1=time.time()

#FDM
def fun(x,y):
    return (2-np.pi**2*y**2)*np.sin(np.pi*x)

n=nx*ny
A1=np.zeros((n,n))
psy_fd=np.zeros((n))
B=np.zeros((n))
    
for i in range(nx,nx*ny-nx):
    if not(i/nx) is int and not((i+1)/ny) is int:
        A1[i][i-nx]=1
        A1[i][i+nx]=1
        A1[i-ny][i]=1
        A1[i+ny][i]=1
        A1[i][i]=-4
        j=i%nx-1
        k=(i-j)//nx
        B[i]=fun(X[j],Y[k])*dx**2

#BC for x,y=const        
for i in range(nx):
    A1[i][i]=1
    B[i]=X[i]*np.exp(-X[i])
    A1[nx*(ny-1)+i-1][nx*(ny-1)+i-1]=1
    B[nx*(ny-1)+i-1]=np.exp(-X[i])*(X[i]+1)

#BC for x=cost,y
for i in range(ny):
    A1[nx*i][nx*i]=1
    B[nx*i]=0
    A1[nx*i+nx-1][nx*i+nx-1]=1
    B[nx*i+nx-1]=0

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

print("time:",time2-time1)        