# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 10:38:41 2021

@author: 姜高晓
"""

import numpy as np
def Read(num):  
    a=open("./异常数据/"+str(num//1)+".异常.txt").read().splitlines()[1:]
    for i in range(len(a)):
        a[i]=a[i].split(':')
    a=np.array(a)
    b=np.zeros((len(a[:,0])//4,5))
    for i in range(len(a[:,0])//4):
        #print(a[0])
        b[i,0]=a[4*i,1]
        b[i,1]=a[4*i,5]
        b[i,2]=a[4*i+1,5]
        b[i,3]=a[4*i+2,5]
        b[i,4]=a[4*i+3,5]
    where=[0]
    while(len(where)!=0):
        '''
        A0sigma=(b[:,1].std())
        A0mean=b[:,1].mean()
        A1sigma=(b[:,2].std())
        A1mean=b[:,2].mean()
        A2sigma=(b[:,3].std())
        A2mean=b[:,3].mean()
        A3sigma=(b[:,4].std())
        A3mean=b[:,4].mean()
        where1=(np.where((b[:,1]-A0mean)>3*A0sigma))
        where2=(np.where((b[:,2]-A1mean)>3*A1sigma))
        where3=(np.where((b[:,3]-A2mean)>3*A2sigma))
        where4=(np.where((b[:,4]-A3mean)>3*A3sigma))
        if len(where1[0])>0:
            b[where1,1]=np.nan
        if len(where2[0])>0:
            b[where2,2]=np.nan
        if len(where3[0])>0:
            b[where3,3]=np.nan
        if len(where4[0])>0:
            b[where4,4]=np.nan
        where1=(np.where(np.isnan(b[:,1])))[0]
        where2=(np.where(np.isnan(b[:,2])))[0]
        where3=(np.where(np.isnan(b[:,3])))[0]
        where4=(np.where(np.isnan(b[:,4])))[0]
        where=list(where1)+list(where2)+list(where3)+list(where4)
        #print(where)
        b=np.delete(b,where,0)#去异常和缺失
        '''
        for i in range(1,5):
            Asigma=(b[:,i].std())
            Amean=(b[:,i].mean())
            where=np.where((b[:,i]-Amean)>3*Asigma)[0]
            b=np.delete(b,where,0)
            where=(np.where(np.isnan(b[:,i])))[0]
            b=np.delete(b,where,0)
        
    b=np.unique(b,axis=0)#去重复
    np.savetxt("./out异常/"+str(num)+".异常.txt",b)
    return b
for i in range(1,325):
    b=Read(i)
