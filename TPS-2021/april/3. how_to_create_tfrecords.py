# ------------------- SECTION Basic DNN with non-tfrecords ------------------- #
# Only this section works.
#%%
# https://www.kaggle.com/cdeotte/how-to-create-tfrecords
# LOAD LIBRARIES
import numpy as np, pandas as pd, os
import matplotlib.pyplot as plt, cv2
import tensorflow as tf, re, math

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

train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

cat_features = train_df.select_dtypes(include=['object']).columns.tolist()
num_features = train_df.drop('target', axis=1).select_dtypes(include=['float']).columns.tolist()

# - Impute the Age NaNs to Age mean.
# - Then all other NaNs will be convert to -1 and the other strings will be
#   converted to 0, 1, 2, 3, ... in the order they appear in the printed lists
#   below.

# SOURCE - https://www.tensorflow.org/tutorials/load_data/pandas_dataframe
# Convert thal column which is an object in the dataframe to a discrete numerical value.

cols = test_df.columns
comb = pd.concat([train_df[cols], test_df[cols]], ignore_index=True, axis=0).reset_index(drop=True)

cats = test_df.select_dtypes(include='object').columns.tolist()

for c in cats:
    comb[c], mp = comb[c].factorize()
    print(mp)
    #Alternative -
    # comb[c] = pd.Categorical(comb[c])
    # comb[c] = comb[c].cat.codes

# Rewrite data to dataframe    
idx = [test_df.columns.get_loc(c) for c in cats]
train_df.iloc[:, idx] = comb.iloc[:train_df.shape[0], idx].values # TECH iloc indexing EXCLUDES the end. loc indexing includes the end. Check https://towardsdatascience.com/everything-you-need-to-know-about-loc-and-iloc-of-pandas-79b386cac776.
test_df.iloc[:, idx] = comb.iloc[train_df.shape[0]:, idx].values # TECH pandas series's values is different from dict object's method values()
test_df['target'] = -1

x_trn = train_df.drop('target', axis=1).values # 此values可有可無
y_trn = train_df['target'].values.reshape((-1,1)) # 此.reshape((-1,1))可有可無
print(x_trn.shape, y_trn.shape)
train_ds = tf.data.Dataset.from_tensor_slices((x_trn, y_trn))

# # Look at the first sample
# for x, y in train_ds.take(1):
#     print ('Features: {}, Target: {}'.format(x.shape, y.shape))
# # Looke at the column
# tf.constant(train_df['cat0'])

# train_dataset = train_ds.shuffle(len(x_trn)).batch(1)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

model = Sequential()
model.add(Input(shape=(30,))) # 不可以用 (30,1)
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC'])
model.fit(x=x_trn, y=y_trn, epochs=15) # works.
# model.fit(train_ds, epochs=15) # doesn't work. Why.


# !SECTION Basic DNN with non-tfrecords
# NOTE The issue I haven't solved is why I cann't feed into model.fit() with 



# # Create and train a model
# def get_compiled_model():
#     model = tf.keras.Sequential([
#         tf.keras.layers.Dense(10, activation='relu'),
#         tf.keras.layers.Dense(10, activation='relu'),
#         tf.keras.layers.Dense(1)
#     ])
  
#     model.compile(optimizer='adam',
#                   loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#                   metrics=['auc'])
#     return model
# model = get_compiled_model()
# model.fit(train_ds, epochs=15)

#%% Loot at the first few samples of the csv files
# slices = tf.data.Dataset.from_tensor_slices(dict(train_df))
# for i, feature_batch in enumerate(slices.take(5)):
#     if i == 4:
#         for key, value in feature_batch.items():
#             print("  {!r:20s}: {}".format(key, value))

# %% This section is in development.
# ----------------------- SECTION Write TFRecords files ---------------------- #
# LINK Source - https://tinyurl.com/ygjaw8eh
# ANCHOR Generate image and tabular TFRecords files
# Quick start guide on TFRecords: https://towardsdatascience.com/working-with-tfrecords-and-tf-train-example-36d111b3ff4d
# (D) Reference - TensorFlow API doc (https://www.tensorflow.org/tutorials/load_data/tfrecord)

def parse_csv_line(line, n_fields = 31):
    # defs = [tf.constant(np.nan)] * n_fields
    defs = [tf.constant(1)] * len(cat_features) + [tf.constant(.1)] * len(num_features) + [tf.constant(1)] * 1
    parsed_fields = tf.io.decode_csv(line, record_defaults=defs)
    #tf.stack(): Join the matrix
    x = tf.stack(parsed_fields[0:-1])
    y = tf.stack(parsed_fields[-1:])
    return x, y

def csv_reader_dataset(
    filenames, 
    n_reader=5, 
    batch_size=32, 
    shuffle_buffer_size=10000
):
    dataset = tf.data.Dataset.list_files(filenames)
    #repeat(): No parameter means to repeat countless times
    #Function: When training the model, we use the data more than once, and we need to use the training set data multiple times and terminate it through epoch
    # dataset = dataset.repeat()
    #interleave(): Read data to form a dataset
    dataset = dataset.interleave(
        lambda filename: tf.data.TextLineDataset(filename), # .skip(1),
        cycle_length = n_reader
    )
    # dataset.shuffle(shuffle_buffer_size)
    #map(): Map to the tf.io.decode_csv() function to parse the data
    dataset = dataset.map(
        parse_csv_line, 
        num_parallel_calls=tf.data.AUTOTUNE)
    # dataset = dataset.batch(batch_size)
    return dataset

batch_size = 30
train_filenames = ['../input/tabular-playground-series-mar-2021/train.csv']
train_set = csv_reader_dataset(train_filenames, batch_size=batch_size)

#%%
def serialize_example(x, y):
    """converts x, y to tf.train.Example and serialize"""
    #Need to pay attention to whether it needs to be converted to numpy() form
    # input_features = tf.train.FloatList(value = x.numpy())
    col_dict = dict()
    cat_features = train_df.select_dtypes(include=['int']).columns.tolist()
    num_features = train_df.drop('target', axis=1).select_dtypes(include=['float']).columns.tolist()
    for col in cat_features:
        idx = train_df.columns.get_loc(col)
        col_dict[col] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[x[idx]])) # string needs to be converted into bytes.
    for col in num_features:
        idx = train_df.columns.get_loc(col)
        col_dict[col] = tf.train.Feature(float_list=tf.train.FloatList(value=[x[idx]])) # NOTE 中括號不能丟     
    col_dict['target'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[y]))
    # label = tf.train.FloatList(value = y )
    features = tf.train.Features(
        # feature = {
        #     # "input_features": tf.train.Feature(float_list = input_features),
        #     # "label" : tf.train.Feature(float_list = label)
        # }
        feature = col_dict
    )
    example = tf.train.Example(features = features)
    return example.SerializeToString()

#n_shards: Store as n_shards file
#steps_per_shard: How many pieces of data are in each file
def csv_dataset_to_tfrecords(
    base_filename, 
    dataset, 
    n_shards, 
    setps_per_shard, 
    compression_type = None
):
    options = tf.io.TFRecordOptions(compression_type = compression_type)
    all_filenames = []
    for shard_id in range(n_shards):
        filename_fullpath = '{}_{:05d}-of-{:05d}'.format(base_filename, shard_id, n_shards)
        with tf.io.TFRecordWriter(filename_fullpath, options) as write:
            #Need to write steps_per_shard times
            for x_batch, y_batch in dataset.take(setps_per_shard):
                for x_example, y_example in zip(x_batch, y_batch):
                    write.write(serialize_example(x_example, y_example))
        all_filenames.append(filename_fullpath)
    return all_filenames    
n_shards = 20
train_steps_per_shard = np.floor(300000 / batch_size / n_shards)
train_basement = 'train'
train_tfrecord_filenames = csv_dataset_to_tfrecords(train_basement, train_set, n_shards, train_steps_per_shard, None)

# !SECTION Write TFRecords files
# %%
