# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 14:19:49 2021

@author: 姜高晓
"""
import numpy as np

from matplotlib import pyplot as plt

R=np.array([ [0,0,1200],
             [5000,0,1600],
             [0,3000,1600],
             [5000,3000,1200]
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
        s[2]+=1*(np.sqrt(((r-R[i])**2).sum())-A[i])*(r[2]-R[i][2])/(np.sqrt(((r-R[i])**2).sum()))
    return s


a=open("./干扰测试change.txt",encoding='utf-8').read().splitlines()
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
for i in range(5):    
    rn=[np.array([2000,2000,2000])]
    d1=b[i,1:]
    for j in range(800):
        rn.append(rn[-1]-0.3*dL(rn[-1],d1))
    rn=np.array(rn)
    l=[rn[-500:,0].mean(),rn[-500:,1].mean(),rn[-500:,2].mean()]
    pre.append(np.array(l))
pre=np.array(pre)//10

pmin=66
pmax=271
lam=(pre[:,2]-pmin)/(pmax-pmin)
pre[:,2]+=np.sign(lam-0.5)*(-50)

pmin=116
pmax=221
lam=(pre[:,2]-pmin)/(pmax-pmin)
pre[:,2]+=np.sin(lam*np.pi)*(-50)

pmin=106
pmax=221
lam=(pre[:,2]-pmin)/(pmax-pmin)
pre[:,2]+=-lam*50







