# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 00:44:41 2018

@author: NitinKhanna
"""
import os
import pandas as pd
import librosa
import glob 
import matplotlib.pyplot as plt
import librosa.display
import IPython.display as ipd
import librosa
plt.figure(figsize=(12, 4))


rAttention='D:/Dropbox/Upside9/Voice-recognition/crying-dataset/rAttention.wav'
ipd.Audio(rAttention)
dAttention, sampling_rate1 = librosa.load(rAttention)
librosa.display.waveplot(dAttention, sr=sampling_rate1)
plt.title('rAttention')


rHungry='D:/Dropbox/Upside9/Voice-recognition/crying-dataset/rHungry.wav'
ipd.Audio(rHungry)
dHungry, sampling_rate2 = librosa.load(rHungry)
librosa.display.waveplot(dHungry, sr=sampling_rate2)
plt.title('rHungry')

rOvertired='D:/Dropbox/Upside9/Voice-recognition/crying-dataset/rOvertired.wav'
ipd.Audio(rOvertired)
dOvertired, sampling_rate3 = librosa.load(rOvertired)
librosa.display.waveplot(dOvertired, sr=sampling_rate3)
plt.title('rOvertired')

rStressed='D:/Dropbox/Upside9/Voice-recognition/crying-dataset/rStressed.wav'
ipd.Audio(rStressed)
dStressed, sampling_rate4 = librosa.load(rStressed)
librosa.display.waveplot(dStressed, sr=sampling_rate3)
plt.title('rStressed')

rTest='D:/Dropbox/Upside9/Voice-recognition/crying-dataset/rStressed.wav'
ipd.Audio(rTest)
dTest, sampling_rate4 = librosa.load(rTest)
librosa.display.waveplot(dTest, sr=sampling_rate3)
plt.title('rTest')

# import numpy as np

# X=[dAttention,dHungry,dOvertired,dStressed]
# X=np.array(X)

X_Attention=[]
Y_Attention=[]
for raw in dAttention:
    if raw >=0:
        X_Attention.append(raw)
    else:
        Y_Attention.append(raw)
        
X_Hungry=[]
Y_Hungry=[]
for raw in dAttention:
    if raw >=0:
        X_Hungry.append(raw)
    else:
        Y_Hungry.append(raw)
        
X_Overtired=[]
Y_Overtired=[]
for raw in dAttention:
    if raw >=0:
        X_Overtired.append(raw)
    else:
        Y_Overtired.append(raw)
        
X_Stressed=[]
Y_Stressed=[]
for raw in dAttention:
    if raw >=0:
        X_Stressed.append(raw)
    else:
        Y_Stressed.append(raw)
        type(X_Stressed)        

import pandas as pd
dataset = pd.read_csv('Data.csv')

X=pd.DataFrame()
X['X_Attention']=X_Attention
X['X_Hungry']=X_Hungry
X['X_Overtired']=X_Overtired
X['X_Stressed']=X_Stressed

Y=pd.DataFrame()
Y['Y_Attention']=Y_Attention
Y['Y_Hungry']=Y_Hungry
Y['Y_Overtired']=Y_Overtired
Y['Y_Stressed']=Y_Stressed

#X = dataset.iloc[:, :].values
#type(X)
#X=pd.DataFrame(X)
#X.describe()
#X.info()
librosa.display.waveplot(dAttention,sr=sampling_rate3)
type(X)

len(X)
len(Y)
a=len(Y)-len(X)
print (a)
Y=Y.iloc[:-a,:]
from sklearn.cross_validation import train_test_split
X_train, X_test,Y_train, Y_test = train_test_split(X,Y, test_size = 0, random_state = 0)
X_test=pd.DataFrame(X_Hungry)
Y_test=pd.DataFrame(Y_Hungry)
len(X_test)
#X_test=dataset.iloc[:,3].values


# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)
len(X_train)
len(Y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_test), color = 'blue')
plt.plot(Y_train, regressor.predict(Y_test), color = 'blue')
plt.title('rAttention (Training set)')
plt.show()

plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('rHungry (Testing set)')
plt.show()