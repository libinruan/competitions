# ----------------------------- WORK ON TFRECORDS ---------------------------- #

#%%
# LINK - Source link - https://www.tensorflow.org/tutorials/structured_data/preprocessing_layers
# screenshot: https://i.postimg.cc/4Nw8zfcH/2021-04-01-at-12-05-02.png
import os
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
from datetime import datetime
import optuna
from collections import OrderedDict
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, optimizers, callbacks, utils, losses, metrics, backend as K
from tensorflow.keras.layers.experimental import preprocessing
print(tf.__version__)

RunningInLinux = os.environ.get('PWD') in ['/kaggle/working', '/root', '/']
if RunningInLinux:
    os.chdir('/kaggle/working')
    train_df = pd.read_csv('../input/tabular-playground-series-mar-2021/train.csv', index_col=0)
    test_df = pd.read_csv('../input/tabular-playground-series-mar-2021/test.csv', index_col=0)
    sample_submission = pd.read_csv('../input/tabular-playground-series-mar-2021/sample_submission.csv')
else:
    os.chdir('G:\kagglePlayground')
    train_df = pd.read_csv('train.csv', index_col=0)
    test_df = pd.read_csv('test.csv', index_col=0)
    sample_submission = pd.read_csv('sample_submission.csv') 

train_df, val_df = train_test_split(train_df, test_size=0.2)
print(len(train_df), 'train examples')
print(len(val_df), 'validation examples')
print(len(test_df), 'test examples')

# train.info()
# test.info()

# LINK Source - https://tinyurl.com/ygjaw8eh
# SECTION Write TFRecords files

# ANCHOR Generate image and tabular TFRecords files
# Quick start guide on TFRecords: https://towardsdatascience.com/working-with-tfrecords-and-tf-train-example-36d111b3ff4d

def create_tf_example(features, label, dataset='train'):
    col_dict = dict()
    if dataset == 'train':
        cat_features = train_df.select_dtypes(include=['object']).columns.tolist()
        num_features = train_df.drop('target', axis=1).select_dtypes(include=['float']).columns.tolist()
        for col in cat_features:
            idx = train_df.columns.get_loc(col)
            col_dict[col] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[features[idx].encode('utf-8')])) # string needs to be converted into bytes.
        for col in num_features:
            idx = train_df.columns.get_loc(col)
            col_dict[col] = tf.train.Feature(float_list=tf.train.FloatList(value=[features[idx]])) # NOTE 中括號不能丟 
        idx = train_df.columns.get_loc('target')
        col_dict['target'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        dataset_example = tf.train.Example(features=tf.train.Features(feature=col_dict))
    elif dataset == 'test':
        cat_features = test_df.select_dtypes(include=['object']).columns.tolist()
        num_features = test_df.select_dtypes(include=['float']).columns.tolist()
        for col in cat_features:
            idx = test_df.columns.get_loc(col)
            col_dict[col] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[features[idx].encode('utf-8')]))
        for col in num_features:
            idx = test_df.columns.get_loc(col)  
            col_dict[col] = tf.train.Feature(float_list=tf.train.FloatList(value=[features[idx]]))        
        # test set doesn't have a target feature.
        dataset_example = tf.train.Example(features=tf.train.Features(feature=col_dict))
    else:
        warnings.warn("dataset = 'train' or 'test'")
        
    return dataset_example # serialized example

# Reference - https://www.tensorflow.org/tutorials/structured_data/preprocessing_layers
def write_train_test_datasets_in_TFRecords():
    with tf.io.TFRecordWriter("/kaggle/input/train.tfrecords") as writer:
        for row in tqdm(train_df.values): # don't forget np.ndarray instead of pandas dataframe.
            features, label = row[:-1], row[-1]
            # storing all the features in the tf.Example message.
            example = create_tf_example(features, label, 'train')
            # write the example messages into byte-strings.
            writer.write(example.SerializeToString()) 
    writer.close()    
    with tf.io.TFRecordWriter("/kaggle/input/test.tfrecords") as writer:
        for row in tqdm(test_df.values): # don't forget np.ndarray instead of pandas dataframe.
            features, label = row, tf.constant(np.nan)
            # storing all the features in the tf.Example message.
            example = create_tf_example(features, label, 'test')
            # write the example messages into byte-strings.
            writer.write(example.SerializeToString())
    writer.close()    

# write_train_test_datasets_in_TFRecords() # just remove comment out to write TFRecords dataset.

# !SECTION write tfrecords example 
    
# SECTION Read TFRecords

def parse_csv_line(line, n_fields = 31):
    defs = [tf.constant(np.nan)] * n_fields
    parsed_fields = tf.io.decode_csv(line, record_defaults=defs)
    #tf.stack(): Join the matrix
    x = tf.stack(parsed_fields[0:-1])
    y = tf.stack(parsed_fields[-1:])
    return x, y

# (U) Reference: https://www.programmersought.com/article/25785351536/
# (D) Reference: https://towardsdatascience.com/working-with-tfrecords-and-tf-train-example-36d111b3ff4d

train_file_paths = ['/kaggle/input/train.tfrecords']
test_file_paths = ['/kaggle/input/test.tfrecords']
cat_features = train_df.select_dtypes(include=['object']).columns.tolist()
num_features = train_df.drop('target', axis=1).select_dtypes(include=['float', 'int']).columns.tolist()

# make a test
# ex = next(tf.compat.v1.io.tf_record_iterator('/kaggle/input/train.tfrecords'))
# print(tf.train.Example.FromString(ex))

# step 1 
trn_tfrd = tf.data.TFRecordDataset(train_file_paths)
tst_tfrd = tf.data.TFRecordDataset(test_file_paths)



#%%
def read_tfrecord(serialized_example, type='train'):
    feature_description = dict()
    for cat in cat_features:
        feature_description[cat] = tf.io.FixedLenFeature((), tf.string)
    for num in num_features:
        feature_description[num] = tf.io.FixedLenFeature((), tf.float32)
    if type=='train':
        feature_description['target'] = tf.io.FixedLenFeature((), tf.int64)
    example = tf.io.parse_single_example(serialized_example, feature_description)
    return example
    # x = tf.stack(example[0:-1])
    # y = tf.stack(example[-1:])
    # return x, y

    # cat_columns = [example[cat] for cat in cat_features] # FIXME
    # num_columns = [example[num] for num in num_features] # FIXME
    # return cat_columns, example['target'] #num_columns

    # if type=='train': 
    #     tar_column = example['target']
    #     res = tf.stack([cat_columns + num_columns]), example['target'] # FIXME
    # else:        
    #     res = tf.stack([cat_columns + num_columns]), None # FIXME
    # return res


# step 2
import functools
from functools import partial
trn_ds = trn_tfrd.map(read_tfrecord)    
tst_ds = tst_tfrd.map(functools.partial(read_tfrecord, type='test'))    
for data in trn_ds.take(1):
    print('\ntrain\n')
    print(data)
for data in tst_ds.take(1):
    print('\ntest\n')
    print(data)

# Help - shuffle: https://stackoverflow.com/questions/53514495/what-does-batch-repeat-and-shuffle-do-with-tensorflow-dataset#answer-53517848
# Help - repeat: https://stackoverflow.com/questions/57711103/difference-between-tf-data-dataset-repeat-vs-iterator-initializer#answer-57712269

# !SECTION read TFRecords

#%% SECTION normalization and one-hot encoding

# trn_ds = trn_ds.shuffle(10000).batch(128).prefetch(128)

# Reference - https://www.tensorflow.org/tutorials/structured_data/preprocessing_layers
def get_normalization_layer(name, dataset):
  # Create a Normalization layer for our feature.
  normalizer = preprocessing.Normalization()
  # Prepare a Dataset that only yields our feature.
  feature_ds = dataset.map(lambda x, y: x[name])
  # Learn the statistics of the data.
  normalizer.adapt(feature_ds)
  return normalizer

def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
  # Create a StringLookup layer which will turn strings into integer indices
  if dtype == 'string':
    index = preprocessing.StringLookup(max_tokens=max_tokens)
  else:
    index = preprocessing.IntegerLookup(max_values=max_tokens)
  # Prepare a Dataset that only yields our feature
  feature_ds = dataset.map(lambda x, y: x[name])
  # Learn the set of possible values and assign them a fixed integer index.
  index.adapt(feature_ds)
  # Create a Discretization for our integer indices.
  encoder = preprocessing.CategoryEncoding(max_tokens=index.vocab_size())
  # Prepare a Dataset that only yields our feature.
  feature_ds = feature_ds.map(index)
  # Learn the space of possible indices.
  encoder.adapt(feature_ds)
  # Apply one-hot encoding to our indices. The lambda function captures the
  # layer so we can use them, or include them in the functional model later.
  return lambda feature: encoder(index(feature))

all_inputs = []
encoded_features = []

# Numeric features.
for header in num_features:
  numeric_col = tf.keras.Input(shape=(1,), name=header)
  normalization_layer = get_normalization_layer(header, trn_ds)
  encoded_numeric_col = normalization_layer(numeric_col)
  all_inputs.append(numeric_col)
  encoded_features.append(encoded_numeric_col)

# !SECTION normalization and one-hot encoding

# ---------------------- WORK ON DATA IN MEMORY DIRECTLY --------------------- #

#%%
# SECTION Tabular May 2021 data (non-tfrecords approach)
# ANCHOR Convert data in memory to Dataset objects
# Wrap the dataframes with tf.data in order to shuffle and batch the data
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    # print(id(dataframe)) # the same id as the input 
    dataframe = dataframe.copy()
    labels = dataframe.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    # generate data from data in memory
    # ds: the Dataset object is a Python iterable.
    # why it is called "tensor_slices" Check this LINK https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_tensor_slices.
    # why we use dict here? Ans: dictionary structur is preserved after slicing. See the link above.
    # And how do we combine two tensors like `(dict(dataframe), labels).` Ans: Use tensor slicing method. See the link above.
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size) # use batch_size
    ds = ds.prefetch(batch_size)
    return ds

batch_size = 5

train_ds = df_to_dataset(train, batch_size=batch_size)

# #Take a look at the first batch.
# [(train_features, label_batch)] = train_ds.take(1) # take a look at a batch (of five)
# print(list(train_features.keys())) # train_ds is a Dataset object, Python iterable. And it is created from dictionary. So we use keys() here.
# print(label_batch)


# SECTION Preprocessing layers
# NOTE Tensorflow API:
# For each of the Numeric feature, you will use a
# Normalization() layer to make sure the mean of each feature is 0 and its
# standard deviation is 1.
# NOTE If you many numeric features (hundreds, or more), it is more efficient to
# concatenate them first and use a single normalization layer.
# For example, `tf.stack([train_features['cont0'], train_features['cont1']], axis=-1)`
# we can do `layer(tf.stack([train_features['cont0'], train_features['cont1']], axis=-1)).`

def get_normalization_layer(name, dataset):
    # Create a Normalization layer for our feature.
    normalizer = preprocessing.Normalization()
    # Prepare a Dataset that only yields our feature.
    feature_ds = dataset.map(lambda x, y: x[name])
    # Learn the statistics of the data.
    normalizer.adapt(feature_ds)
    return normalizer

# #Take a look at one numerical column in snapshot
# cont0_col = train_features['cont0'] # tensorflow eager tensor
# layer = get_normalization_layer('cont0', train_ds)
# layer(cont0_col)

# ANCHOR One hot encoding
# Categorical columns
def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
    # Create a StringLookup layer which will turn strings into integer indices
    if dtype == 'string':
      index = preprocessing.StringLookup(max_tokens=max_tokens)
    else:
      index = preprocessing.IntegerLookup(max_values=max_tokens)
    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    # Learn the set of possible values and assign them a fixed integer index.
    index.adapt(feature_ds)
    # Create a Discretization for our integer indices.
    encoder = preprocessing.CategoryEncoding(max_tokens=index.vocab_size())
    # Prepare a Dataset that only yields our feature.
    feature_ds = feature_ds.map(index)
    # Learn the space of possible indices.
    encoder.adapt(feature_ds)
    # Apply one-hot encoding to our indices. The lambda function captures the
    # layer so we can use them, or include them in the functional model later.
    return lambda feature: encoder(index(feature))

## take a look at one-hot encoding on a single feature dataset.
# cat0_col = train_features['cat0']
# layer = get_category_encoding_layer('cat0', train_ds, 'string')
# layer(cat0_col)

## NOTE you can also convert integers into one-hot encoding. See the api link above for details.

# !SECTION

all_inputs = []
encoded_features = []

# Numeric features.
for header in tqdm(num_features):
    numeric_col = tf.keras.Input(shape=(1,), name=header)
    normalization_layer = get_normalization_layer(header, train_ds)
    encoded_numeric_col = normalization_layer(numeric_col)
    all_inputs.append(numeric_col)
    encoded_features.append(encoded_numeric_col)

# skip # Categorical features encoded as integers.

# Categorical features encoded as string.
for header in tqdm(cat_features):
    categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='string')
    encoding_layer = get_category_encoding_layer(header, train_ds, dtype='string',
                                                 max_tokens=5)
    encoded_categorical_col = encoding_layer(categorical_col)
    all_inputs.append(categorical_col)
    encoded_features.append(encoded_categorical_col)

# Create, compile, and train the model
all_features = layers.concatenate(encoded_features)
x = tf.keras.layers.Dense(32, activation="relu")(all_features)
x = tf.keras.layers.Dropout(0.5)(x)
output = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(all_inputs, output)
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.AUC()])
              
utils.plot_model(model, show_shapes=True, rankdir="LR")              

# !SECTION Tabular May 2021 data