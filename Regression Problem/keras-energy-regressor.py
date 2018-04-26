from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

from keras.models import Sequential
from keras.layers import Dense, Activation,Flatten,Dropout
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from pylab import rcParams
import matplotlib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

tf.logging.set_verbosity(tf.logging.INFO)
sess = tf.InteractiveSession()

train = pd.read_csv('Train Data/train.csv')
print('Shape of the train data with all features:', train.shape)
train = train.select_dtypes(exclude=['object'])
print("")
print('Shape of the train data with numerical features:', train.shape)
train.drop('Observation',axis = 1, inplace = True)
train_labels = train['Energy']
print("train labels:",train_labels.shape)
train.fillna(0,inplace=True)
train.drop('Energy',axis = 1, inplace = True)
test = pd.read_csv('Train Data/test.csv')
test = test.select_dtypes(exclude=['object'])
#ID = test.Id
test.fillna(0,inplace=True)
test.drop('Observation',axis = 1, inplace = True)
test_labels = pd.read_csv('Train Data/sample_submission.csv')
test_labels.drop('Observation',axis = 1, inplace = True)
print("test_labels:",test_labels.columns)

print("")
print("List of features contained our dataset train:",list(train.columns))
print("List of features contained our dataset test:",list(test.columns))

train.head(10)

import warnings
warnings.filterwarnings('ignore')

col_train = list(train.columns)
col_train_bis = list(train.columns)

#col_train_bis.remove('SalePrice')

mat_train = np.matrix(train)
mat_test  = np.matrix(test)
mat_y_test = np.array(test_labels).reshape((3945,1))
mat_y_train = np.array(train_labels).reshape((15780,1))

prepro_y_train = MinMaxScaler()
prepro_y_train.fit(mat_y_train)

prepro_y_test = MinMaxScaler()
prepro_y_test.fit(mat_y_test)

prepro = MinMaxScaler()
prepro.fit(mat_train)

prepro_test = MinMaxScaler()
prepro_test.fit(mat_test)

train = np.array(pd.DataFrame(prepro.transform(mat_train),columns = col_train))
val = train[int(0.8*len(train)):]
train = train[:int(0.8*len(train))]
test  =np.array( pd.DataFrame(prepro_test.transform(mat_test),columns = col_train_bis))
y_train =np.array( pd.DataFrame(prepro_y_train.transform(mat_y_train),columns = ['Energy']))
y_val = y_train[int(0.8*len(y_train)):]
y_train = y_train[:int(0.8*len(y_train))]
y_test =np.array( pd.DataFrame(prepro_y_test.transform(mat_y_test),columns = ['Energy']))

#train.head()
#test.head()
#y_train.head()
#y_test.head()

print(train.shape)
print(test.shape)
print(y_train.shape)
print(y_test.shape)
print(val.shape)
print(y_val.shape)
model = Sequential()
model.add(Dense(256,input_dim = 24,activation='relu'))
#model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(128,activation='tanh'))
#model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32,activation='tanh'))
model.add(Dropout(0.1))
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='tanh'))
#model.add(Activation('tanh'))

model.summary()

model.compile(optimizer='rmsprop',
              loss='mse')

model.fit(train, y_train, epochs=10, batch_size=32,validation_data=(val,y_val))

loss = model.evaluate(x=test,y=y_test)

print("loss:",loss)