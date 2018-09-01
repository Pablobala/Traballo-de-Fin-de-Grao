# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 18:51:35 2018

@author: USUARIO
"""


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import curve_fit
from matplotlib.colors import LogNorm
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
    df.at[i,'mK']=np.sqrt((np.sqrt(Pp**2+139.57018**2)+np.sqrt(Pn**2+139.57018**2))**2-PN**2)
    df.at[i,'mlambda']=np.sqrt((np.sqrt(Pp**2+938.272081**2)+np.sqrt(Pn**2+139.57018**2))**2-PN**2)
    df.at[i,'mantilambda']=np.sqrt((np.sqrt(Pp**2+139.57018**2)+np.sqrt(Pn**2+938.272081**2))**2-PN**2)
print(df)

x=np.asarray(df['alpha'])
y=np.asarray(df['pT'])
mK=np.asarray(df['mK'])
mL=np.asarray(df['mlambda'])
mA=np.asarray(df['mantilambda'])

#Histogramas das masas
plt.figure(2)
plt.hist(mK,bins=100,range=(450,550))
plt.title(r'masa $ K_S^0$ datos LHCb')
plt.ylabel('contas')
plt.xlabel(r'$masa \; (\frac{MeV}{c^2})$')
plt.grid()
plt.figure(3)
plt.hist(mL,bins=100,range=(1080,1140))
plt.title(r'masa $\Lambda^0$ datos LHCb')
plt.ylabel('contas')
plt.xlabel(r'$masa \; (\frac{MeV}{c^2})$')
plt.grid()
plt.figure(4)
plt.hist(mA,bins=100,range=(1080,1140))
plt.title(r'masa $\bar{\Lambda}^0$ datos LHCb')
plt.ylabel('contas')
plt.xlabel(r'$masa \; (\frac{MeV}{c^2})$')
plt.grid()

#plot
plt.figure(1)
plt.plot(x,y,linestyle='None',marker='o', markersize='0.1',color='g',label='datos LHCb')
plt.title('Armenteros-Podolanski plot')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$P_T \; (\frac{MeV}{c})$')
plt.legend(loc='best',markerscale=18)
plt.grid()
plt.figure(5)
plt.hist2d(x,y,bins=100,norm=LogNorm())
plt.title('Armenteros-Podolanski density plot')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$P_T \; (\frac{MeV}{c})$')
plt.colorbar()
