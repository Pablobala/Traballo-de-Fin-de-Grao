# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 18:30:42 2018

@author: USUARIO
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from scipy.stats import norm
import scipy.interpolate
import random
from numpy import random
import os
plt.close('all')
headers=['m','Px+','Px-','Py+','Py-','Pz+','Pz-']
df=pd.read_csv('data2.csv', names=headers)

n=len(df)
df.astype(float)

for i in range(0,n):
    Pxp=df.iloc[i][1]
    Pxn=df.iloc[i][2]
    Pyp=df.iloc[i][3]
    Pyn=df.iloc[i][4]
    Pzp=df.iloc[i][5]
    Pzn=df.iloc[i][6]
    Pp=np.sqrt(Pxp**2+Pyp**2+Pzp**2)
    Pn=np.sqrt(Pxn**2+Pyn**2+Pzn**2)
    PN=np.sqrt((Pxp+Pxn)**2+(Pyp+Pyn)**2+(Pzp+Pzn)**2)
    cos1=(Pxp*(Pxp+Pxn)+Pyp*(Pyp+Pyn)+Pzp*(Pzp+Pzn))/(Pp*PN)
    cos2=(Pxn*(Pxp+Pxn)+Pyn*(Pyp+Pyn)+Pzn*(Pzp+Pzn))/(Pn*PN)
    PL=Pp*cos1+Pn*cos2
    df.at[i,'pT']=Pp*np.sqrt(1-cos1**2)
    df.at[i,'alpha']=(Pp*cos1-Pn*cos2)/(Pp*cos1+Pn*cos2)
print(df)
#taking lambda data
x=np.asarray(df['alpha'])
y=np.asarray(df['pT'])
x1_L=x[np.argwhere(y<120).T[0]]
y1_L=y[np.argwhere(y<120).T[0]]
x_L=x1_L[np.argwhere(x1_L>0).T[0]]
y_L=y1_L[np.argwhere(x1_L>0).T[0]]
realdata=np.array([x_L,y_L]).T
#Theil index method:
def func(x, m_L, m_p):
    return np.sqrt(abs((-m_L)**2*(1.-x**2)/4.+(m_p**2-139.57018**2)/2.*x-(m_p**2+139.57018**2)/2.))

nps=200000 #Number of pseudo datasets.
ngi=50 #Number of iterations for calculating the Theil index for the same mass pair.
lamda=(1150-1050)*np.random.rand(nps)+1050
proton=(1000-900)*np.random.rand(nps)+900
y_binnum=500 #number of bins along the y-axis
x_binnum=y_binnum #number of bins along the x-axis
#Binning the real data.
StartMatrix,ax,ay=np.histogram2d(realdata[:,1],realdata[:,0],x_binnum,range=[[0,150],[0,1]])
Theil=np.zeros((nps,3)) #[Theil index,Lambda mass,proton mass]
# Creating the histogram of the real data.
a = realdata[:,0]
counts, bins = np.histogram(a, bins=100, density=True)
cum_counts = np.cumsum(counts)
bin_widths = (bins[1:] - bins[:-1])
# Interpolating the histogram and finding the distribution.
x = cum_counts*bin_widths
y = bins[1:]
inverse_density_function = scipy.interpolate.interp1d(x, y)
for p in range(nps): #going trough the pseudo datasets
    #Generating alpha values using the same distribution.
    alpha=inverse_density_function( random.uniform( x[0], x[-1],[ngi,realdata.shape[0]]) )
    pt=func(alpha,lamda[p],proton[p]) #finding the y-coordinate of the bin where an event belongs
    TheilGaus=np.zeros((ngi))
    for jj in range(ngi):
        matrix,ax,ay=np.histogram2d(pt[jj],alpha[jj],x_binnum,range=[[0,150],[0,1]])
        Matrix=matrix + StartMatrix+1
        T=0
        events=np.asarray(Matrix).reshape(-1)
        mean=np.mean(events)
        #calculating the Theil index
        events1=events
        Ev=events*np.log(events1/mean)
        
        T=np.sum(Ev)/mean
        T/=Ev.shape[0]*np.log(Ev.shape[0])
        TheilGaus[jj]=T
    mu, std = norm.fit(TheilGaus)
    Theil[p]=[mu,lamda[p],proton[p]]
np.savetxt('L_data_200000(50).txt',Theil, delimiter=',')