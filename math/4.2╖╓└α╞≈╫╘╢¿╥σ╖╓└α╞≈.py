# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 14:19:49 2021

@author: 姜高晓
"""
import numpy as np
from scipy.fftpack import fft,ifft
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import ParameterGrid, GridSearchCV
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
        s[0]+=1*(np.sqrt(((r-R[i])**2).sum())-A[i])*(r[0]-R[i][0])/(np.sqrt(((r-R[i])**2).sum()))*1
        s[1]+=1*(np.sqrt(((r-R[i])**2).sum())-A[i])*(r[1]-R[i][1])/(np.sqrt(((r-R[i])**2).sum()))*1
        s[2]+=1*(np.sqrt(((r-R[i])**2).sum())-A[i])*(r[2]-R[i][2])/(np.sqrt(((r-R[i])**2).sum()))*1
    return s
alpha=0.1
Nstep=800
pre=[]
for i in range(1,325):
    d=np.loadtxt("./out正常/"+str(i//1)+".正常.txt")
    #d=np.loadtxt("./out异常/"+str(i//1)+".异常.txt")
    d1=np.zeros(4)
    d1[0]=d[:,1].mean()
    d1[1]=d[:,2].mean()
    d1[2]=d[:,3].mean()
    d1[3]=d[:,4].mean()
    rn=[np.array([2000,2000,2000])]
    for i in range(Nstep):
        rn.append(rn[-1]-alpha*dL(rn[-1],d1))
    rn=np.array(rn)
    l=[rn[-50:,0].mean(),rn[-50:,1].mean(),rn[-50:,2].mean()]
    lstd=np.array([rn[-500:,0].var(),rn[-500:,1].var(),rn[-500:,2].var()])
    pre.append([L(l,d1),lstd.sum()])
pre=np.array(pre)
kam1=pre[:,0].min()
kam2=pre[:,1].max()
#mean=
pre[:,0]=pre[:,0]/kam1
pre[:,1]=pre[:,1]/kam2

pre1=np.array(pre)
a1=pre[:,0].mean()
a2=pre[:,1].mean()
#%%朴素值    
pre=[]
for i in range(1,325):
    #d=np.loadtxt("./out正常/"+str(i//1)+".正常.txt")
    d=np.loadtxt("./out异常/"+str(i//1)+".异常.txt")
    d1=np.zeros(4)
    d1[0]=d[:,1].mean()
    d1[1]=d[:,2].mean()
    d1[2]=d[:,3].mean()
    d1[3]=d[:,4].mean()
    rn=[np.array([2000,2000,2000])]
    for i in range(Nstep):
        rn.append(rn[-1]-alpha*dL(rn[-1],d1))
    rn=np.array(rn)
    l=[rn[-50:,0].mean(),rn[-50:,1].mean(),rn[-50:,2].mean()]
    lstd=np.array([rn[-500:,0].var(),rn[-500:,1].var(),rn[-500:,2].var()])
    pre.append([L(l,d1),lstd.sum()])
pre=np.array(pre)
pre[:,0]=pre[:,0]/kam1
pre[:,1]=pre[:,1]/kam2
#plt.plot(pre)
#plt.show()
b1=pre[:,0].mean()
b2=pre[:,1].mean()
# 126 124 53
# 118 128 63
# 119 122 138  正常
#1.4 1.4 0.57

pre2=np.array(pre)



a=open("./4.txt",encoding='utf-8').read().splitlines()
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
for i in range(10):    
    rn=[np.array([2000,2000,2000])]
    d1=b[i,1:]
    for j in range(Nstep):
        rn.append(rn[-1]-alpha*dL(rn[-1],d1))
    rn=np.array(rn)
    l=[rn[-50:,0].mean(),rn[-50:,1].mean(),rn[-50:,2].mean()]
    lstd=np.array([rn[-500:,0].var(),rn[-500:,1].var(),rn[-500:,2].var()])
    pre.append([L(l,d1),lstd.sum()])
pre=np.array(pre)
pre[:,0]=pre[:,0]/kam1
pre[:,1]=pre[:,1]/kam2
out=pre

aaaaa=out[:,0]*out[:,1]
label=[]
for i in range(10):
    A1=len(np.where(pre1[:,0]>out[i,0])[0])
    A2=(len(np.where(pre2[:,0]>out[i,0])[0]))
    label.append(A2/(A1+A2))
    
np.array(label)>0.5


































