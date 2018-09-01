# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 13:01:56 2018

@author: USUARIO
"""


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import curve_fit
plt.close('all')
#leo os meus datos
headers=['m','Px+','Px-','Py+','Py-','Pz+','Pz-','ID+','ID-']
df=pd.read_csv('lambda5.csv', names=headers)
n=len(df)
df.astype(float)
#trasformo as variabeis
for i in range(0,n):
    Pxp=df.iloc[i][1]
    Pxn=df.iloc[i][2]
    Pyp=df.iloc[i][3]
    Pyn=df.iloc[i][4]
    Pzp=df.iloc[i][5]
    Pzn=df.iloc[i][6]
    Pp=np.sqrt(Pxp**2+Pyp**2+Pzp**2)
    Pn=np.sqrt(Pxn**2+Pyn**2+Pzn**2)
    Plam=np.sqrt((Pxp+Pxn)**2+(Pyp+Pyn)**2+(Pzp+Pzn)**2)
    cos1=(Pxp*(Pxp+Pxn)+Pyp*(Pyp+Pyn)+Pzp*(Pzp+Pzn))/(Pp*Plam)
    cos2=(Pxn*(Pxp+Pxn)+Pyn*(Pyp+Pyn)+Pzn*(Pzp+Pzn))/(Pn*Plam)
    PL=Pp*cos1+Pn*cos2
    df.at[i,'pT']=Pp*np.sqrt(1-cos1**2)
    df.at[i,'alpha']=(PL*Pp-PL*Pn)/(PL*Pp+PL*Pn)
    #Quero separar a lambda da antilambda
    if df.iloc[i][7]==2212:
        df.at[i,'mlambda']=np.sqrt((np.sqrt(Pp**2+938.272013**2)+np.sqrt(Pn**2+139.57018**2))**2-Plam**2)
    if df.iloc[i][7]==211:
        df.at[i,'mantilambda']=np.sqrt((np.sqrt(Pp**2+139.57018**2)+np.sqrt(Pn**2+938.272013**2))**2-Plam**2)

print(df)

#Axuste
def f(x,M,m1,m2):
    w =np.sqrt(M**2./4.*(1-x**2.)+(m1**2.-m2**2.)*x/2.-(m1**2.+m2**2)/2.)
    return w
#paso a array para poder manexar comodamente os datos co curve_fit
x=np.asarray(df['alpha'])
y=np.asarray(df['pT'])
#Selecciono os datos do (anti)lambda
x_pos=x[np.argwhere(x<0).T[0]]
y_pos=y[np.argwhere(x<0).T[0]]
#parámetros do axuste
p=(1115,139,938)
#Hago el ajuste:
(M,m1,m2), u= curve_fit(f,x_pos,y_pos,p0=p)
#A diagonal da matriz u ten as incertezas ao cadrado dos nosos parámetros
print('M=', M , ';  s(M)=', np.sqrt(np.diag(u)[0]))
print('m1=', m1,  ';  s(m1)=', np.sqrt(np.diag(u)[1]))
print('m2=', m2,  ';  s(m2)=', np.sqrt(np.diag(u)[2]))
#plot:
#ploteo os meus puntos
plt.plot(x_pos,y_pos,markersize='0.4',marker='.',color='k', linestyle='None')
#para plotear o axuste tomo moitos puntos do mesmo e representoos
xx=np.linspace(-0.8726,-0.5091, 2000)
FF=f(xx,M,m1,m2)
plt.plot(xx,FF,color='r')
plt.title(r'Axuste $\bar{\Lambda}^0$ [simulado]')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$P_T \; (\frac{MeV}{c})$')
plt.grid()