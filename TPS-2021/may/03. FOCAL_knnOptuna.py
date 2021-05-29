#%%
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import gc

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, classification_report, log_loss, f1_score, average_precision_score
from sklearn.utils import class_weight

import tensorflow as tf
from keras.utils import to_categorical
from keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization,  Flatten, Embedding, MaxPooling1D, Conv1D
from keras.layers.merge import concatenate
from keras.utils import plot_model
from tensorflow.keras import activations,callbacks
from keras import backend as K

from tqdm.notebook import tqdm
from IPython.display import Image , display

import warnings
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)


# Time Functions
import datetime
import pytz

def dtnow(tz="America/New_York"):
    utc_now = pytz.utc.localize(datetime.datetime.utcnow())
    pst_now = utc_now.astimezone(pytz.timezone("America/New_York"))
    return pst_now.strftime("%Y%m%d%H%M")

def timer(start_time=None):
    if not start_time:
        start_time = datetime.datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('Time taken : %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


# GCS API and Credentials        
from google.cloud import storage

project = "strategic-howl-305522"
bucket_name = "gcs-station-168"           
storage_client = storage.Client(project=project)
bucket = storage_client.bucket(bucket_name)

train = pd.read_csv("/kaggle/input/tabular-playground-series-may-2021/train.csv", index_col = 'id')
test = pd.read_csv("/kaggle/input/tabular-playground-series-may-2021/test.csv", index_col = 'id')
raw_features = test.columns.tolist()

# To assign all out of bounds test value an identical value
CLIP_FEATURES = True

df_all_X = pd.concat([train.drop('target', axis=1), test], axis=0)
if CLIP_FEATURES == False:
    le = LabelEncoder()
    df_all_X = df_all_X.apply(le.fit_transform)
else:    
    missing_integer = df_all_X.max().max() + 15 # we will replace any unseen test value with this one.
    for col in test.columns: # inspect across all test set's columns
        if not all(test[col].isin(train[col].value_counts().index.tolist())): # see if there's out of bounds value
            for i in set(test[col]).difference(set(train[col])):
                test[col].replace(i, missing_integer, inplace=True) # replace the oob value with a dummy value
            train[f'{col}_{missing_integer}'] = 0 # generate a boolean column to mark all those missing values in test set
            test[f'{col}_{missing_integer}'] = 1 # generate a boolean column to mark all those missing values in test set
            test[f'{col}_{missing_integer}'].where(test[col]==missing_integer, 0, inplace=True) 
    df_all_X = pd.concat([train.drop('target', axis=1), test], axis=0)

# Gropu low frequency into one value            
GROUP_LOW_FREQUENCY = True
GROUP_LOW_FREQUENCY_THRESHOLD = 10

if GROUP_LOW_FREQUENCY:
    for col in raw_features:
        value_counts_SS = df_all_X[col].value_counts()
        low_freq_values = value_counts_SS.index[value_counts_SS < GROUP_LOW_FREQUENCY_THRESHOLD]
        if len(low_freq_values) > 0:
            df_all_X[f'{col}_low_freq'] = 0
            for i in low_freq_values.tolist():
                df_all_X[f'{col}_low_freq'].iloc[df_all_X[col]==i] = 1

            



# %%





    pass
else:
    
    Xtrn, Xtst = df_all_X.iloc[:len(train)], df_all_X[len(train):]



# %%
RANDOM_STATE = 2021

construction_mode = True
if construction_mode == True:
    NUM_HEADS = 2 # 05281402 number of heads
    HEAD_EPOCHS = 50 # 05281411 head epochs
    HYDRA_EPOCHS = 30 # 05281410 hydra epochs
    TRAIN_VERBOSE = 2 # https://tinyurl.com/yzfkfoq6
    NUM_CLASS = 4
    

X = train.drop('target', axis = 1)

lencoder = LabelEncoder()
y = pd.DataFrame(lencoder.fit_transform(train['target']), columns=['target'])
df_all = pd.concat([X, test], axis = 0) 
df_all = df_all.apply(lencoder.fit_transform) # TODO 05281333
X, test = df_all[:len(train)], df_all[len(train):]
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y, random_state= RANDOM_STATE) # TODO 05281334
NUM_FEATURES = len(X_train.columns)    

# inspired from mhttps://www.kaggle.com/pourchot/lb-1-0896-keras-nn-with-20-folds

es = callbacks.EarlyStopping(monitor = 'val_loss', # TODO eaerly stopping patience setup 05281335
                             min_delta = 0.0000001, 
                             patience = 2,
                             mode = 'min',
                             baseline = None, 
                             restore_best_weights = True,
                             verbose = 1)

plateau  = callbacks.ReduceLROnPlateau(monitor = 'val_loss', 
                                       factor = 0.5, 
                                       patience = 2, 
                                       mode = 'min', 
                                       min_delt = 0.0000001,
                                       cooldown = 0, 
                                       min_lr = 1e-8,
                                       verbose = 1) 
