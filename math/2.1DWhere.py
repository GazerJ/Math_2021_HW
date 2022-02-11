# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 14:19:49 2021

@author: 姜高晓
"""
import numpy as np

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
        s[2]+=1*(np.sqrt(((r-R[i])**2).sum())-A[i])*(r[2]-R[i][2])/(np.sqrt(((r-R[i])**2).sum()))
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
    for i in range(300):
        rn.append(rn[-1]-0.15*dL(rn[-1],d1))
    pre.append(rn[-1])
    
    
#%%朴素值    
pre=np.array(pre)//10
#pre[:,2]=pre[:,2]+(np.sign(pre[:,2]-150))*20

#%% 求残差，利用机器学习得到拟合出偏置函数

data=open("./Tag.txt",encoding='utf-8').read().splitlines()[2:]
box=[]
for i in range(len(data)-1):
    data[i]=data[i].split(':')[1].split()
    for j in range(3):
        box.append(int(data[i][j]))
data1=np.array(box).reshape(324,3)







delta=pre-data1
plt.plot(delta[:,0],label="$err_x$")
plt.plot(delta[:,1],label="$err_y$")
plt.plot(delta[:,2],label="$err_z$")
plt.xlabel("num")
plt.ylabel("err")
plt.legend()
plt.show()




plt.scatter(pre[:,2],delta[:,2],label="$err_z$")
plt.xlabel("predict")
plt.ylabel("err")
plt.legend()
plt.show()














