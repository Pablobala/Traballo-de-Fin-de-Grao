# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 12:39:07 2018

@author: USUARIO
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
plt.close('all')
#leo os meus datos
headers=['m','Px1','Py1','Pz1','Px2','Py2','Pz2']
df=pd.read_csv('kspipi2.csv', names=headers)
n=len(df)
df.astype(float)
for i in range(0,n):
    Px1=df.iloc[i][1]
    Py1=df.iloc[i][2]
    Pz1=df.iloc[i][3]
    Px2=df.iloc[i][4]
    Py2=df.iloc[i][5]
    Pz2=df.iloc[i][6]
    P1=np.sqrt(Px1**2+Py1**2+Pz1**2)
    P2=np.sqrt(Px2**2+Py2**2+Pz2**2)
    PK=np.sqrt((Px1+Px2)**2+(Py1+Py2)**2+(Pz1+Pz2)**2)
    cos1=(Px1*(Px1+Px2)+Py1*(Py1+Py2)+Pz1*(Pz1+Pz2))/(P1*PK)
    cos2=(Px2*(Px1+Px2)+Py2*(Py1+Py2)+Pz2*(Pz1+Pz2))/(P2*PK)
    PL=P1*cos1+P2*cos2
    df.at[i,'pT']=P1*np.sqrt(1-cos1**2)
    df.at[i,'alpha']=(PL*P1-PL*P2)/(PL*P1+PL*P2)
    df.at[i,'mK']=np.sqrt((np.sqrt(P1**2+139.57018**2)+np.sqrt(P2**2+139.57018**2))**2-PK**2)
headers2=['m','Px+','Px-','Py+','Py-','Pz+','Pz-','ID+','ID-']
df2=pd.read_csv('lambda2.csv', names=headers2)
n2=len(df2)
df2.astype(float)
for j in range(0,n2):
    Pxp=df2.iloc[j][1]
    Pxn=df2.iloc[j][2]
    Pyp=df2.iloc[j][3]
    Pyn=df2.iloc[j][4]
    Pzp=df2.iloc[j][5]
    Pzn=df2.iloc[j][6]
    Pp=np.sqrt(Pxp**2+Pyp**2+Pzp**2)
    Pn=np.sqrt(Pxn**2+Pyn**2+Pzn**2)
    Plam=np.sqrt((Pxp+Pxn)**2+(Pyp+Pyn)**2+(Pzp+Pzn)**2)
    cos12=(Pxp*(Pxp+Pxn)+Pyp*(Pyp+Pyn)+Pzp*(Pzp+Pzn))/(Pp*Plam)
    cos22=(Pxn*(Pxp+Pxn)+Pyn*(Pyp+Pyn)+Pzn*(Pzp+Pzn))/(Pn*Plam)
    PL=Pp*cos12+Pn*cos22
    df2.at[j,'pT2']=Pp*np.sqrt(1-cos12**2)
    df2.at[j,'alpha2']=(PL*Pp-PL*Pn)/(PL*Pp+PL*Pn)
    #Quero separar a lambda da antilambda
    if df2.iloc[j][7]==2212:
        df2.at[j,'mlambda']=np.sqrt((np.sqrt(Pp**2+938.272081**2)+np.sqrt(Pn**2+139.57018**2))**2-Plam**2)
    if df2.iloc[j][7]==211:
        df2.at[j,'mantilambda']=np.sqrt((np.sqrt(Pp**2+139.57018**2)+np.sqrt(Pn**2+938.272081**2))**2-Plam**2)

#Seecciono as variables que me interesan
xK=np.asarray(df['alpha'])
yK=np.asarray(df['pT'])
xl=np.asarray(df2['alpha2'])
yl=np.asarray(df2['pT2'])
xL=xl[np.argwhere(xl>0.).T[0]]
yL=yl[np.argwhere(xl>0.).T[0]]
xA=xl[np.argwhere(xl<0.).T[0]]
yA=yl[np.argwhere(xl<0.).T[0]]

#Fago o plot de A-P para os datos simulados
plt.figure(1)
plt.plot(xK,yK,linestyle='None',marker='o', markersize='0.1',color='g',label=r'$K_S^0$')
plt.plot(xL,yL,linestyle='None',marker='o', markersize='0.1',color='r',label=r'$\Lambda^0$')
plt.plot(xA,yA,linestyle='None',marker='o', markersize='0.1',color='b',label=r'$\bar{\Lambda}^0$')
plt.title('Datos simulados')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$P_T \; (\frac{MeV}{c})$')
plt.legend(loc='best',markerscale=18)
plt.grid()

#Histogramas das masas:
mK=np.asarray(df['mK'])
mL=np.asarray(df2['mlambda'])
mA=np.asarray(df2['mantilambda'])
plt.figure(2)
plt.hist(mK,bins=100,range=(450,550))
plt.title(r'$masa \; K_S^0$')
plt.ylabel('contas')
plt.xlabel(r'$masa \; (\frac{MeV}{c^2})$')
plt.grid()
plt.figure(3)
plt.hist(mL,bins=100,range=(1080,1140))
plt.title(r'$masa \; \Lambda^0$')
plt.ylabel('contas')
plt.xlabel(r'$masa \; (\frac{MeV}{c^2})$')
plt.grid()
plt.figure(4)
plt.hist(mA,bins=100,range=(1080,1140))
plt.title(r'$masa \; \bar{\Lambda}^0$')
plt.ylabel('contas')
plt.xlabel(r'$masa \; (\frac{MeV}{c^2})$')
plt.grid()