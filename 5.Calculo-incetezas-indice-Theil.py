# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 14:31:50 2018

@author: USUARIO
"""

import numpy as np
import matplotlib.pylab as plt
from scipy.stats import norm
import scipy.interpolate
import random
from numpy import random
import os
#Inceteza kaon
dataK=np.loadtxt("dataK.txt",delimiter=',') #loading the actual collision data
n=len(dataK[:,1])/500
n=np.int(n)
RESK=np.array([0,0,0])
for i in range(n):
    realdata=dataK[i*500:i*500+500,:]
    datas=realdata #ollo ca s!!!!!!
    def func(x, m_Ks, m_pi):
        return np.sqrt(abs((-m_Ks)**2*(1.-x**2)/4.-m_pi**2))

    nps=1000 #Number of pseudo datasets.
    ngi=1 #Number of iterations for calculating the Theil index for the same mass pair.
    kaon=(550-450)*np.random.rand(nps)+450
    pion=(200-100)*np.random.rand(nps)+100
    y_binnum=500 #number of bins along the y-axis
    x_binnum=y_binnum #number of bins along the x-axis
    #Binning the real data.
    StartMatrix,ax,ay=np.histogram2d(realdata[:,1],realdata[:,0],x_binnum,range=[[0,350],[-1,1]])
    Theil=np.zeros((nps,3)) #[Theil index,Kaon mass,pion mass]
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
        alpha=inverse_density_function( random.uniform( x[0], x[-1],[ngi,datas.shape[0]]) )
        pt=func(alpha,kaon[p],pion[p]) #finding the y-coordinate of the bin where an event belongs
        TheilGaus=np.zeros((ngi))
        for jj in range(ngi):
            matrix,ax,ay=np.histogram2d(pt[jj],alpha[jj],x_binnum,range=[[0,350],[-1,1]])
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
        Theil[p]=[mu,kaon[p],pion[p]]
    l=np.where(Theil[:,0]==np.max(Theil[:,0]))
    RESK=np.c_[RESK,Theil[l].T]
np.savetxt('Incertezas K.txt',RESK, delimiter=',')

#Incerteza lambda
dataL=np.loadtxt("dataL.txt",delimiter=',') #loading the actual collision data
nL=len(dataL[:,1])/500
nL=np.int(nL)
RESL=np.array([0,0,0])
for i in range(nL):
    realdata=dataL[i*500:i*500+500,:]
    datas=realdata #ollo ca s!!!!!!
    def func(x, m_L, m_p):
        return np.sqrt(abs((-m_L)**2*(1.-x**2)/4.+(m_p**2-139.57**2)/2.*x-(m_p**2+139.57**2)/2.))

    nps=1000 #Number of pseudo datasets.
    ngi=1 #Number of iterations for calculating the Theil index for the same mass pair.
    lamda=(1150-1050)*np.random.rand(nps)+1050
    proton=(1000-900)*np.random.rand(nps)+900
    y_binnum=500 #number of bins along the y-axis
    x_binnum=y_binnum #number of bins along the x-axis
    #Binning the real data.
    StartMatrix,ax,ay=np.histogram2d(realdata[:,1],realdata[:,0],x_binnum,range=[[0,150],[0,1]])
    Theil=np.zeros((nps,3)) #[Theil index,Kaon mass,pion mass]
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
        alpha=inverse_density_function( random.uniform( x[0], x[-1],[ngi,datas.shape[0]]) )
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
    l=np.where(Theil[:,0]==np.max(Theil[:,0]))
    RESL=np.c_[RESL,Theil[l].T]
np.savetxt('Incertezas L.txt',RESL, delimiter=',')

#Inceteza antilambda
dataA=np.loadtxt("dataAl.txt",delimiter=',') #loading the actual collision data
nA=len(dataA[:,1])/500
nA=np.int(nA)
RESA=np.array([0,0,0])
for i in range(nA):
    realdata=dataA[i*500:i*500+500,:]
    datas=realdata #ollo ca s!!!!!!
    def func(x, m_L, m_p):
        return np.sqrt(abs((-m_L)**2*(1.-x**2)/4.-(m_p**2-139.57**2)/2.*x-(m_p**2+139.57**2)/2.))
    
    nps=1000 #Number of pseudo datasets.
    ngi=1 #Number of iterations for calculating the Theil index for the same mass pair.
    lamda=(1150-1050)*np.random.rand(nps)+1050
    proton=(1000-900)*np.random.rand(nps)+900
    y_binnum=500 #number of bins along the y-axis
    x_binnum=y_binnum #number of bins along the x-axis
    #Binning the real data.
    StartMatrix,ax,ay=np.histogram2d(realdata[:,1],realdata[:,0],x_binnum,range=[[0,150],[-1,0]])
    Theil=np.zeros((nps,3)) #[Theil index,Kaon mass,pion mass]
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
        alpha=inverse_density_function( random.uniform( x[0], x[-1],[ngi,datas.shape[0]]) )
        pt=func(alpha,lamda[p],proton[p]) #finding the y-coordinate of the bin where an event belongs
        TheilGaus=np.zeros((ngi))
        for jj in range(ngi):
            matrix,ax,ay=np.histogram2d(pt[jj],alpha[jj],x_binnum,range=[[0,150],[-1,0]])
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
    l=np.where(Theil[:,0]==np.max(Theil[:,0]))
    RESA=np.c_[RESA,Theil[l].T]
np.savetxt('Incertezas Al.txt',RESA, delimiter=',')