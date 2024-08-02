#!/usr/bin/env python
# coding: utf-8

# In[ ]:


##LSTM
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.layers import LSTM, Dense,Activation,Bidirectional,Dropout, Flatten
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import numpy as np
from keras.utils import to_categorical
from sklearn.metrics import roc_auc_score
df = pd.read_csv('1000timem')
properties = list(df.columns.values)
properties.remove('target')
X = df[properties]
y = df['target']
print(X.shape,y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
X_train = np.resize(X_train,(X_train.shape[0],X_train.shape[1],1))
y_train = np.resize(y_train,(y_train.shape[0],1))
y = y_test
y_train = to_categorical(y_train)
print(X_train.shape,y_train.shape)
lst = Sequential() # initializing model

# input layer and LSTM layer with 50 neurons
lst.add(LSTM(units=100,activation='tanh',return_sequences=True, input_shape=(X_train.shape[1],1)))
lst.add(Dropout(0.2))
lst.add(LSTM(10, activation='tanh'))
# outpute layer with sigmoid activation
lst.add(Flatten())
# lst.add(Dense(50, activation='sigmoid'))
lst.add(Dense(25, activation='sigmoid'))
lst.add(Dense(10, activation='sigmoid'))
lst.add(Dense(2, activation='sigmoid'))

# defining loss function, optimizer, metrics and then compiling model
lst.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

# training the model on training dataset
history = lst.fit(X_train, y_train, epochs=30, batch_size=32,verbose=2,validation_split=0.2)
X_test = np.resize(X_test,(X_test.shape[0],X_test.shape[1],1))
y_test = np.resize(y_test,(y_test.shape[0],1))
y_test = to_categorical(y_test)
test_loss, test_acc = lst.evaluate(X_test, y_test)
print('Test accuracy:', "{:.2f}%".format(100*test_acc))
p = lst.predict(X_test)
ROC_AUC_score = roc_auc_score(y, p[:,1])
print('Prediction Test accuracy:', "{:.2f}%".format(100*ROC_AUC_score))

##Bi-LSTM
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import tensorflow as tf
from tensorflow import keras  
from keras.layers import LSTM, Dense,Activation,Bidirectional,Dropout, Flatten
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import numpy as np
from keras.utils import to_categorical
from sklearn.metrics import roc_auc_score
df = pd.read_csv("1000timem.csv")
df.head()
df.loc[df.duplicated(),:]
from sklearn.preprocessing import LabelEncoder
IPS = LabelEncoder()
df['Source'] = IPS.fit_transform   (df['Source'])
df.dropna()
df['Destination'] = IPS.fit_transform(df['Destination'])
df['Protocol'] = IPS.fit_transform(df['Protocol'])
df['info'] = IPS.fit_transform(df['info'])
df.dropna()
properties = list(df.columns.values)
properties.remove('Target')
X = df[properties]
y = df['Target']
print(X.shape,y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
model = Sequential()
model.add(Bidirectional(LSTM(100, activation='tanh',return_sequences=True,input_shape=(X_train.shape[1],1))))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(10, activation='tanh')))
model.add(Dense(50, activation='sigmoid'))
model.add(Dense(25, activation='sigmoid'))
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
X_train = np.resize(X_train,(X_train.shape[0],X_train.shape[1],1))
y_train = np.resize(y_train,(y_train.shape[0],1))
print(X_train,y_train)
model.fit(X_train, y_train, epochs=20, batch_size=32,verbose=2)
X_test = np.resize(X_test,(X_test.shape[0],X_test.shape[1],1))
y_test = np.resize(y_test,(y_test.shape[0],1))
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', "{:.2f}%".format(100*test_acc))

