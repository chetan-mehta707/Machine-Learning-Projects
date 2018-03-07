import pandas as pd
import tensorflow as tf

TRAIN_PATH = '/Users/chetan/.spyder-py3/Train Data/train.csv'
TEST_PATH = '/Users/chetan/.spyder-py3/Train Data/test.csv'
TEST_LABEL_PATH = '/Users/chetan/.spyder-py3/Train Data/sample_submission.csv'

df_train = pd.read_csv(TRAIN_PATH)

feature_columns = list(df_train.keys())
COLUMNS= list(feature_columns)
feature_columns.remove('Observation')
feature_columns.remove('Energy')
FEATURES=feature_columns
LABEL='Energy'

print('columns',COLUMNS)
print('features',FEATURES)
print('labels',LABEL)

feature_cols = [tf.feature_column.numeric_column(k,dtype=tf.float64) for k in FEATURES]
print(feature_cols)
regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
                                          hidden_units=[10,10,10,10])
print(regressor)

def get_input_fn(data_set,label_set, num_epochs=None, shuffle=True):
  print('in input fn',FEATURES)
  #print({k: data_set[k].values for k in FEATURES})
  #x=pd.DataFrame({k: data_set[k].values for k in FEATURES})
  #print(x)
  return tf.estimator.inputs.pandas_input_fn(
      x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
      y = pd.Series(label_set[LABEL].values),
      num_epochs=num_epochs,
      shuffle=shuffle)


training_set = pd.read_csv(TRAIN_PATH, 
                           skipinitialspace=True,skiprows=1, names=COLUMNS)

test_set = pd.read_csv(TEST_PATH, skipinitialspace=True,
                         skiprows=1, names=COLUMNS)
test_set_label = pd.read_csv(TEST_LABEL_PATH, skipinitialspace=True,
                         skiprows=1, names=['Observation','Energy'])
#get_input_fn(df_train,10)
#print(test_set)
regressor.fit(input_fn=get_input_fn(training_set,training_set), steps=5000)
ev = regressor.evaluate(
      input_fn=get_input_fn(test_set, test_set_label,num_epochs=1, shuffle=False))
loss_score = ev["loss"]
print("Loss: {0:f}".format(loss_score))