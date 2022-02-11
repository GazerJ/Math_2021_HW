# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 14:19:49 2021

@author: 姜高晓
"""
import numpy as np

from matplotlib import pyplot as plt
from numba import njit
from mpl_toolkits.mplot3d import Axes3D
from scipy.fftpack import fft,ifft
R=np.array([ [0,0,1300],
             [5000,0,1700],
             [0,5000,1700],
             [5000,5000,1300]
             ])
#%%  梯度下降法
@njit
def L(r,A):
    s=0
    for i in range(4):
        s+=(np.sqrt(((r-R[i])**2).sum())-A[i])**2
    return s
@njit
def dL(r,A):
    s=np.zeros(3)
    for i in range(4):##i表示标号
        s[0]+=1*(np.sqrt(((r-R[i])**2).sum())-A[i])*(r[0]-R[i][0])/(np.sqrt(((r-R[i])**2).sum()))
        s[1]+=1*(np.sqrt(((r-R[i])**2).sum())-A[i])*(r[1]-R[i][1])/(np.sqrt(((r-R[i])**2).sum()))
        s[2]+=1*(np.sqrt(((r-R[i])**2).sum())-A[i])*(r[2]-R[i][2])/(np.sqrt(((r-R[i])**2).sum()))
    return s


Nstep=800
rn=[np.array([2000,2000,2000])]
def noInter(d1):
    
    for j in range(Nstep):
        rn.append(rn[-1]-0.4*dL(rn[-1],d1))
    pre=rn[-1]
    pre=pre//10
    pre[2]=pre[2]+(np.sign(pre[2]-150))*20
    return pre
    
def yesInter(d1):
    
    for j in range(Nstep):
        rn.append(rn[-1]-0.4*dL(rn[-1],d1))
    pre=rn[-1]
    pre=pre//10
    #pre[2]=pre[2]+(np.sign(pre[2]-150))*20
    pmin=66
    pmax=271
    lam=(pre[2]-pmin)/(pmax-pmin)
    pre[2]+=np.sign(lam-0.5)*(-50)
    pmin=116
    pmax=221
    lam=(pre[2]-pmin)/(pmax-pmin)
    pre[2]+=np.sin(lam*np.pi)*(-50)
    pmin=106
    pmax=221
    lam=(pre[2]-pmin)/(pmax-pmin)
    pre[2]+=-lam*50

    return pre


a=open("./附件5：动态轨迹数据.txt").read().splitlines()[1:]
for i in range(len(a)):
    a[i]=a[i].split(':')
    
a=np.array(a)
b=np.zeros((len(a[:,0])//4,5))
for i in range(len(a[:,0])//4):
    b[i,0]=a[4*i,1]
    b[i,1]=a[4*i,5]
    b[i,2]=a[4*i+1,5]
    b[i,3]=a[4*i+2,5]
    b[i,4]=a[4*i+3,5]
    
pre=[]
soccer=[]
for i in range(len(b[:,0])):
    d1=b[i,1:]
    pre.append(noInter(d1))
    soccer.append(L(pre[-1]*10,d1))
    if soccer[-1]>3200:
        pre[-1]=(yesInter(d1))
pre=np.array(pre)
for i in range(len(pre[:,0])):
    pre[i,0]=pre[i:i+30,0].mean()
    pre[i,1]=pre[i:i+30,1].mean()
    pre[i,2]=pre[i:i+30,2].mean()
#plt.plot(pre[:,2])
#plt.show()



fig = plt.figure()
ax1 = plt.axes(projection='3d')
ax1.plot3D(pre[:,0],pre[:,1],pre[:,2],'black') 
plt.show()


















