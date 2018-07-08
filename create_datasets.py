# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 21:50:21 2018

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
import numpy as np

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
for raw in dHungry:
    if raw >=0:
        X_Hungry.append(raw)
    else:
        Y_Hungry.append(raw)
        
X_Overtired=[]
Y_Overtired=[]
for raw in dOvertired:
    if raw >=0:
        X_Overtired.append(raw)
    else:
        Y_Overtired.append(raw)
        
X_Stressed=[]
Y_Stressed=[]
for raw in dStressed:
    if raw >=0:
        X_Stressed.append(raw)
    else:
        Y_Stressed.append(raw)
        

#type(X_Stressed)        

#import pandas as pd
#dataset = pd.read_csv('Data.csv')

X_Attention=pd.DataFrame(X_Attention)
a=len(X_Attention)-1000
X_Attention=X_Attention.iloc[:-a,:]
type(X_Attention)
X_Attention=X_Attention[0].tolist()

Y_Attention=pd.DataFrame(Y_Attention)
a=len(Y_Attention)-1000
Y_Attention=Y_Attention.iloc[:-a,:]
#type(X_Attention)
Y_Attention=Y_Attention[0].tolist()

X_Hungry=pd.DataFrame(X_Hungry)
a=len(X_Hungry)-1000
X_Hungry=X_Hungry.iloc[:-a,:]
#type(X_Attention)
X_Hungry=X_Hungry[0].tolist()

Y_Hungry=pd.DataFrame(Y_Hungry)
a=len(Y_Hungry)-1000
Y_Hungry=Y_Hungry.iloc[:-a,:]
#type(X_Attention)
Y_Hungry=Y_Hungry[0].tolist()

X_Overtired=pd.DataFrame(X_Overtired)
a=len(X_Overtired)-1000
X_Overtired=X_Overtired.iloc[:-a,:]
#type(X_Attention)
X_Overtired=X_Overtired[0].tolist()

Y_Overtired=pd.DataFrame(Y_Overtired)
a=len(Y_Overtired)-1000
Y_Overtired=Y_Overtired.iloc[:-a,:]
#type(X_Attention)
Y_Overtired=Y_Overtired[0].tolist()

X_Stressed=pd.DataFrame(X_Stressed)
a=len(X_Stressed)-1000
X_Stressed=X_Stressed.iloc[:-a,:]
#type(X_Attention)
X_Stressed=X_Stressed[0].tolist()

Y_Stressed=pd.DataFrame(Y_Stressed)
a=len(Y_Stressed)-1000
Y_Stressed=Y_Stressed.iloc[:-a,:]
#type(X_Attention)
Y_Stressed=Y_Stressed[0].tolist()

len(Y_Stressed)




X_Attention=pd.DataFrame(X_Attention)
X_Hungry=pd.DataFrame(X_Hungry)
X_Overtired=pd.DataFrame(X_Overtired)
X_Stressed=pd.DataFrame(X_Stressed)

Y_Attention=pd.DataFrame(Y_Attention)
Y_Hungry=pd.DataFrame(Y_Hungry)
Y_Overtired=pd.DataFrame(Y_Overtired)
Y_Stressed=pd.DataFrame(Y_Stressed)
