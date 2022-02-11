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
        s[0]+=1*(np.sqrt(((r-R[i])**2).sum())-A[i])*(r[0]-R[i][0])/(np.sqrt(((r-R[i])**2).sum()))*0.1
        s[1]+=1*(np.sqrt(((r-R[i])**2).sum())-A[i])*(r[1]-R[i][1])/(np.sqrt(((r-R[i])**2).sum()))*0.1 
        s[2]+=1*(np.sqrt(((r-R[i])**2).sum())-A[i])*(r[2]-R[i][2])/(np.sqrt(((r-R[i])**2).sum()))*5
    return s
alpha=0.4
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
        rn.append(rn[-1]-alpha*dL(rn[-1],d1)+np.random.normal(0,5,3))
    rn=np.array(rn)
    l=[rn[-500:,0].var(),rn[-500:,1].var(),rn[-500:,2].var()]
    pre.append(np.array(l))
pre=np.array(pre)

pre1=np.array(pre)
plt.plot(pre)
plt.show()
a1=pre[:,0].mean()
a2=pre[:,1].mean()
a3=pre[:,2].mean()  
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
        rn.append(rn[-1]-alpha*dL(rn[-1],d1)+np.random.normal(0,5,3))
    rn=np.array(rn)
    l=[rn[-500:,0].var(),rn[-500:,1].var(),rn[-500:,2].var()]
    pre.append(np.array(l))
pre=np.array(pre)


plt.plot(pre)
plt.show()
b1=pre[:,0].mean()
b2=pre[:,1].mean()
b3=pre[:,2].mean()  
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
        rn.append(rn[-1]-alpha*dL(rn[-1],d1)+np.random.normal(0,5,3))
    rn=np.array(rn)
    l=[rn[-500:,0].var(),rn[-500:,1].var(),rn[-500:,2].var()]
    pre.append(np.array(l))
out=np.array(pre)

x=np.array(list(pre1[:,2])+list(pre2[:,2])).reshape(-1,1)
y=np.array(list(np.ones((len(pre1[:,0]),1)))+list(np.zeros((len(pre2[:,0]),1))))###1是好数据

print("over")
#分割数据
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.95,random_state=0)
#两种核函数



kernel=('rbf ','rbf')
gamma=np.arange(0.1,1.5,0.05)
c=np.arange(0.1,1.5,0.05)
grid={'kernel':kernel,'gamma':gamma,'C':c}
svc_search=GridSearchCV(estimator=svm.SVC(),param_grid=grid,cv=3)
svc_search.fit(x_train,y_train)
print(svc_search.best_params_)
pre_test2=svc_search.predict(x_test)
#测试得分
cf2=confusion_matrix(y_test,pre_test2)
cr2=classification_report(y_test,pre_test2)






pre_test1111=svc_search.predict(out[:,2].reshape(-1,1))






































