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
        s[2]+=1*(np.sqrt(((r-R[i])**2).sum())-A[i])*(r[2]-R[i][2])/(np.sqrt(((r-R[i])**2).sum()))*5
    return s
pre=[]
for i in range(1,325):
    d=np.loadtxt("./out异常/"+str(i//1)+".异常.txt")
    d1=np.zeros(4)
    d1[0]=d[:,1].mean()
    d1[1]=d[:,2].mean()
    d1[2]=d[:,3].mean()
    d1[3]=d[:,4].mean()
    rn=[np.array([2000,2000,2000])]
    for i in range(800):
        rn.append(rn[-1]-0.3*dL(rn[-1],d1))
    rn=np.array(rn)
    l=[rn[-500:,0].mean(),rn[-500:,1].mean(),rn[-500:,2].mean()]
    pre.append(np.array(l))
    
    
#%%朴素值    
pre=np.array(pre)//10
print(pre[:,2].min(),pre[:,2].max())
lam=(pre[:,2]-pre[:,2].min())/(pre[:,2].max()-pre[:,2].min())
#pre[:,2]+=np.cos(lam)*np.pi)*(110)  std=29
pre[:,2]+=np.sign(lam-0.5)*(-50)
print(pre[:,2].min(),pre[:,2].max())
lam=(pre[:,2]-pre[:,2].min())/(pre[:,2].max()-pre[:,2].min())
pre[:,2]+=np.sin(lam*np.pi)*(-50)
print(pre[:,2].min(),pre[:,2].max())
lam=(pre[:,2]-pre[:,2].min())/(pre[:,2].max()-pre[:,2].min())
pre[:,2]+=-lam*50

lam=(pre[:,2]-pre[:,2].min())/(pre[:,2].max()-pre[:,2].min())
#pre[:,2]+=np.sign(lam-0.5)*(-50)
#






#%% 求残差，利用机器学习得到拟合出偏置函数

data=open("./Tag.txt",encoding='utf-8').read().splitlines()[2:]
box=[]
for i in range(len(data)-1):
    data[i]=data[i].split(':')[1].split()
    for j in range(3):
        box.append(int(data[i][j]))
data1=np.array(box).reshape(324,3)







delta=pre-data1
pre[:,0]-=delta[:,0].mean()
pre[:,1]-=delta[:,1].mean()
pre[:,2]-=delta[:,2].mean()
delta=pre-data1




#print((delta[:,2]).mean())
print(abs(delta[:,0]).std())
#print((delta[:,2]).mean())
print(abs(delta[:,1]).std())
print((delta[:,2]).mean())
print(abs(delta[:,2]).std())

plt.plot(delta[:,0],label="$err_x$")
plt.plot(delta[:,1],label="$err_y$")
plt.plot(delta[:,2],label="$err_z$")
plt.xlabel("num")
plt.ylabel("err")
plt.legend()
plt.show()

plt.plot(pre[:,2])
plt.show()












