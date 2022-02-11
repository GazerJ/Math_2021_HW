# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 14:19:49 2021

@author: 姜高晓
"""
import numpy as np

from matplotlib import pyplot as plt

R=np.array([ [0,0,1300],
             [5000,0,1700],
             [0,5000,1700],
             [5000,5000,1300]
             ])
#%%  梯度下降法
def L(r,A):
    s=0
    for i in range(4):
        s+=(np.sqrt(((r-R[i])**2).sum())-A[i])**2
    return s
def dL(r,A):
    s=np.zeros(3)
    for i in range(4):##i表示标号
        s[0]+=1*(np.sqrt(((r-R[i])**2).sum())-A[i])*(r[0]-R[i][0])/(np.sqrt(((r-R[i])**2).sum()))
        s[1]+=1*(np.sqrt(((r-R[i])**2).sum())-A[i])*(r[1]-R[i][1])/(np.sqrt(((r-R[i])**2).sum()))
        s[2]+=2*(np.sqrt(((r-R[i])**2).sum())-A[i])*(r[2]-R[i][2])/(np.sqrt(((r-R[i])**2).sum()))*2
    return s


a=open("./正常测试.txt").read().splitlines()
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
rn=[np.array([2000,2000,2000])]
pre=[]
for i in range(5):
    d1=b[i,1:]
    for j in range(300):
        rn.append(rn[-1]-0.35*dL(rn[-1],d1))
    pre.append(rn[-1])
pre=np.array(pre)//10
pre[:,2]=pre[:,2]+(np.sign(pre[:,2]-150))*20







