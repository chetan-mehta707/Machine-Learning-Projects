import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

df_train = pd.read_csv('E:\TensorFlow Probs\Regression Prob\Train Data/train.csv')
df_train_data = df_train.drop(['Energy','Observation'],axis=1).iloc[1:]
#print(df_train_data)
#print(df_train_data.__class__)
#train_labels = df_train['Energy'].iloc[1:]
#print(train_labels)

feature_columns = list(df_train.keys())
#feature_columns.remove('Observation')
#feature_columns.remove('Energy')
print(feature_columns)

classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    # Two hidden layers of 10 nodes each.
    hidden_units=[10, 10],
    # The model must choose between 3 classes.
    n_classes=3)
print(classifier)

COLUMNS= feature_columns
feature_columns.remove('Observation')
feature_columns.remove('Energy')
FEATURES=feature_columns
LABEL='Energy'



def get_input_fn(data_set, num_epochs=None, shuffle=True):
  x=pd.DataFrame({k: data_set[k].values for k in FEATURES})
  print(x)
  return tf.estimator.inputs.pandas_input_fn(
      x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
      y = pd.Series(data_set[LABEL].values),
      num_epochs=num_epochs,
      shuffle=shuffle)
#training_set = pd.read_csv("E:\TensorFlow Probs\Regression Prob\Train Data/train.csv", skipinitialspace=True,
 #                          skiprows=1, names=feature_columns)
get_input_fn(df_train,10)
#print(training_set)
