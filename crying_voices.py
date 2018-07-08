# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 00:44:41 2018

@author: NitinKhanna
"""

#X=np.vstack((X_Attention, X_Hungry)).T
#X1=np.vstack((X_Overtired, X_Stressed)).T


# =============================================================================
# X=pd.DataFrame()
# X['X_Attention']=X_Attention
# X['X_Hungry']=X_Hungry
# X['X_Overtired']=X_Overtired
# X['X_Stressed']=X_Stressed
# 
# Y=pd.DataFrame()
# Y['Y_Attention']=Y_Attention
# Y['Y_Hungry']=Y_Hungry
# Y['Y_Overtired']=Y_Overtired
# Y['Y_Stressed']=Y_Stressed
# 
# =============================================================================
#X = dataset.iloc[:, :].values
#type(X)
#X=pd.DataFrame(X)
#X.describe()
#X.info()

#librosa.display.waveplot(dAttention,sr=sampling_rate3)
#type(X)
#
#len(X)
#len(Y)
#a=len(Y)-len(X)
#print (a)
#Y=Y.iloc[:-a,:]


from sklearn.cross_validation import train_test_split
X_train_Attention, X_test,Y_train_Attention, Y_test = train_test_split(X_Attention,Y_Attention, test_size = 0, random_state = 0)


from sklearn.cross_validation import train_test_split
X_train_Hungry, X_test,Y_train_Hungry, Y_test = train_test_split(X_Hungry,Y_Hungry, test_size = 0, random_state = 0)

from sklearn.cross_validation import train_test_split
X_train_Overtired, X_test,Y_train_Overtired, Y_test = train_test_split(X_Overtired,Y_Overtired, test_size = 0, random_state = 0)

from sklearn.cross_validation import train_test_split
X_train_Stressed, X_test,Y_train_Stressed, Y_test = train_test_split(X_Stressed,Y_Stressed, test_size = 0, random_state = 0)


X_test=pd.DataFrame(X_Stressed)
Y_test=pd.DataFrame(Y_Stressed)

#len(X_test)
#X_test=dataset.iloc[:,3].values


# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train_Attention,Y_train_Attention)

# Predicting the Test set results
y_pred_Attention = regressor.predict(X_test)

#librosa.display.waveplot(y_pred_Attention, sr=sampling_rate3)
#plt.title('predict_Attention')

regressor1 = LinearRegression()
regressor1.fit(X_train_Hungry,Y_train_Hungry)
y_pred_Hungry= regressor1.predict(X_test)
#librosa.display.waveplot(y_pred_Hungry, sr=sampling_rate3)
#plt.title('predict_Hungry')



regressor2 = LinearRegression()
regressor2.fit(X_train_Overtired,Y_train_Overtired)
y_pred_Overtired= regressor2.predict(X_test)
#librosa.display.waveplot(y_pred_Overtired, sr=sampling_rate3)
#plt.title('predict_Overtired')

regressor3 = LinearRegression()
regressor3.fit(X_train_Stressed,Y_train_Stressed)
y_pred_Stressed= regressor3.predict(X_test)
#librosa.display.waveplot(y_pred_Stressed, sr=sampling_rate3)
#plt.title('predict_Stressed')



plt.scatter(X_train_Stressed, Y_train_Stressed, color = 'red')
plt.plot(X_train_Stressed, regressor.predict(X_test), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
