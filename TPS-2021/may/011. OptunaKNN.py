# %%


# %%
import pandas as pd
import numpy as np
import random
import os
import pickle
import datetime
import pytz

from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.linear_model import SGDClassifier

from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from category_encoders import OrdinalEncoder
import lightgbm as lgb
from scipy.special import erfinv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC

pd.options.display.max_rows = 100
pd.options.display.max_columns = 100

import warnings
warnings.simplefilter('ignore')

TARGET = 'target'
NUM_CLASSES = 4
SEED = 2021

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
seed_everything(SEED)

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


os.chdir('/kaggle/working')
train = pd.read_csv('../input/tabular-playground-series-may-2021/train.csv')
test = pd.read_csv('../input/tabular-playground-series-may-2021/test.csv')
submission = pd.read_csv('../input/tabular-playground-series-may-2021/sample_submission.csv')

features = train.iloc[:, 1:-1].columns.tolist()
all_df = pd.concat([train, test]).reset_index(drop=True).drop(['id', 'target'], axis=1)

from sklearn.preprocessing import LabelEncoder
all_df = all_df.apply(LabelEncoder().fit_transform)

def rank_gauss_standardscaler(all_df):
    def rank_gauss(x):
        N = x.shape[0]
        temp = x.argsort()
        rank_x = temp.argsort() / N
        rank_x -= rank_x.mean()
        rank_x *= 2
        efi_x = erfinv(rank_x)
        efi_x -= efi_x.mean()
        return efi_x

    all_df = all_df.apply(rank_gauss)    
    for col in features:
        all_df[col] = StandardScaler().fit_transform(all_df[col].values.reshape(-1,1))
    return all_df
all_df = rank_gauss_standardscaler(all_df)    

X_trn, X_tst = all_df.iloc[:len(train)], all_df.iloc[len(train):]
y_trn = LabelEncoder().fit_transform(train['target'])
y_trn_cat = to_categorical(y_trn)
RANDOM_SEED = 2021

import os
import numpy as np
import pandas as pd
from keras.layers import Dense, Conv2D, BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import Input, Flatten, Dropout
from keras.layers import Activation
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.models import Model, load_model
from keras.utils import to_categorical
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import time
from pathlib import Path

# %% NOTE CONSTRUCTION

x_calib1, x_calib2, y_calib1, y_calib2 = train_test_split(X_trn, y_trn_cat,
        train_size=0.5, shuffle=True, stratify=y_trn_cat)

data_dictionary = {}
data_dictionary['x_calib1'] = x_calib1
data_dictionary['y_calib1'] = y_calib1
data_dictionary['x_calib2'] = x_calib2
data_dictionary['y_calib2'] = y_calib2  
x_calib = data_dictionary['x_calib1']
y_calib = data_dictionary['y_calib1']
num_classes = y_calib.shape[1]
number_of_classes = num_classes
dir_save = '/kaggle/working/'
num_cnn_blocks = 4
# num_filters = trial.suggest_categorical('num_filters', [16, 32, 48, 64])
# kernel_size = trial.suggest_int('kernel_size', 2, 4)
num_dense_nodes = 64
dense_nodes_divisor = 8
batch_size = 32
drop_out = 0.2
maximum_epochs = 1000
early_stop_epochs = 10
learning_rate_epochs = 5
optimizer_direction = 'minimize'
number_of_random_points = 0 # 25  # random searches to start opt process
maximum_time = 60 # 4*60*60  # seconds      
results_directory = "/kaggle/working/"
                  
dict_params = {'num_cnn_blocks':num_cnn_blocks,
            #    'num_filters':num_filters,
            #    'kernel_size':kernel_size,
               'num_dense_nodes':num_dense_nodes,
               'dense_nodes_divisor':dense_nodes_divisor,
               'batch_size':batch_size,
               'drop_out':drop_out}
                                      
# start of cnn coding   
input_tensor = Input(shape=(50,1)) # Input(shape=self.input_shape)
 
# 1st cnn block
x = BatchNormalization()(input_tensor)
x = Activation('relu')(x)
# x = Conv2D(filters=dict_params['num_filters'],
#            kernel_size=dict_params['kernel_size'],
#            strides=1, padding='same')(x)
#x = MaxPooling2D()(x)
x = Dense(dict_params['num_dense_nodes'], activation='relu')(x)
x = Dropout(dict_params['drop_out'])(x)
 
# additional cnn blocks
for iblock in range(dict_params['num_cnn_blocks'] - 1):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = Conv2D(filters=dict_params['num_filters'],
    #            kernel_size=dict_params['kernel_size'],
    #            strides=1, padding='same')(x)
    x = Dense(dict_params['num_dense_nodes'], activation='relu')(x)
    # x = MaxPooling2D()(x)
    x = Dropout(dict_params['drop_out'])(x)
                 
# mlp
x = Flatten()(x)
x = Dense(dict_params['num_dense_nodes'], activation='relu')(x)
x = Dropout(dict_params['drop_out'])(x)
x = Dense(dict_params['num_dense_nodes']//dict_params['dense_nodes_divisor'], 
          activation='relu')(x)
output_tensor = Dense(number_of_classes, activation='softmax')(x)
 
# instantiate and compile model
dnn_model = Model(inputs=input_tensor, outputs=output_tensor)
opt = Adam(lr=0.00025)  # default = 0.001
dnn_model.compile(loss='categorical_crossentropy',
                  optimizer=opt, metrics=['accuracy'])
 
 
# callbacks for early stopping and for learning rate reducer
fn = dir_save + str('test') + '_dnn.h5'
callbacks_list = [EarlyStopping(monitor='val_loss', patience=early_stop_epochs),                     
                  ReduceLROnPlateau(monitor='val_loss', factor=0.1, 
                                    patience=learning_rate_epochs, 
                                    verbose=0, mode='auto', min_lr=1.0e-6),
                  ModelCheckpoint(filepath=fn,
                                  monitor='val_loss', save_best_only=True)]
     
# fit the model
h = dnn_model.fit(x=x_calib1, y=y_calib1,
                  batch_size=dict_params['batch_size'],
                  epochs=maximum_epochs,
                  validation_split=0.25,
                  shuffle=True, verbose=1,
                  callbacks=callbacks_list)
         
validation_loss = np.min(h.history['val_loss'])

# %%

class Objective(object):
    def __init__(self, xcalib, ycalib, dir_save,
                 max_epochs, early_stop, learn_rate_epochs,
                 number_of_classes):
        self.xcalib = xcalib
        self.ycalib = ycalib
        self.max_epochs = max_epochs
        self.early_stop = early_stop
        self.dir_save = dir_save
        self.learn_rate_epochs = learn_rate_epochs
        #self.input_shape = input_shape
        self.number_of_classes = number_of_classes
 
    def __call__(self, trial):        
        num_cnn_blocks = trial.suggest_int('num_cnn_blocks', 2, 4)
        # num_filters = trial.suggest_categorical('num_filters', [16, 32, 48, 64])
        # kernel_size = trial.suggest_int('kernel_size', 2, 4)
        num_dense_nodes = trial.suggest_categorical('num_dense_nodes',
                                                    [64, 128, 512, 1024])
        dense_nodes_divisor = trial.suggest_categorical('dense_nodes_divisor',
                                                        [2, 4, 8])
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 96, 128])
        drop_out=trial.suggest_discrete_uniform('drop_out', 0.05, 0.5, 0.05)
                          
        dict_params = {'num_cnn_blocks':num_cnn_blocks,
                    #    'num_filters':num_filters,
                    #    'kernel_size':kernel_size,
                       'num_dense_nodes':num_dense_nodes,
                       'dense_nodes_divisor':dense_nodes_divisor,
                       'batch_size':batch_size,
                       'drop_out':drop_out}
                                              
        # start of cnn coding   
        input_tensor = Input(shape=(50,1)) # Input(shape=self.input_shape)
         
        # 1st cnn block
        x = BatchNormalization()(input_tensor)
        x = Activation('relu')(x)
        # x = Conv2D(filters=dict_params['num_filters'],
        #            kernel_size=dict_params['kernel_size'],
        #            strides=1, padding='same')(x)
        #x = MaxPooling2D()(x)
        x = Dense(dict_params['num_dense_nodes'], activation='relu')(x)
        x = Dropout(dict_params['drop_out'])(x)
         
        # additional cnn blocks
        for iblock in range(dict_params['num_cnn_blocks'] - 1):
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            # x = Conv2D(filters=dict_params['num_filters'],
            #            kernel_size=dict_params['kernel_size'],
            #            strides=1, padding='same')(x)
            x = Dense(dict_params['num_dense_nodes'], activation='relu')(x)
            # x = MaxPooling2D()(x)
            x = Dropout(dict_params['drop_out'])(x)
                         
        # mlp
        x = Flatten()(x)
        x = Dense(dict_params['num_dense_nodes'], activation='relu')(x)
        x = Dropout(dict_params['drop_out'])(x)
        x = Dense(dict_params['num_dense_nodes']//dict_params['dense_nodes_divisor'], 
                  activation='relu')(x)
        output_tensor = Dense(self.number_of_classes, activation='softmax')(x)
         
        # instantiate and compile model
        dnn_model = Model(inputs=input_tensor, outputs=output_tensor)
        opt = Adam(lr=0.00025)  # default = 0.001
        dnn_model.compile(loss='categorical_crossentropy',
                          optimizer=opt, metrics=['accuracy'])
         
         
        # callbacks for early stopping and for learning rate reducer
        fn = self.dir_save + str(trial.number) + '_dnn.h5'
        callbacks_list = [EarlyStopping(monitor='val_loss', patience=self.early_stop),                     
                          ReduceLROnPlateau(monitor='val_loss', factor=0.1, 
                                            patience=self.learn_rate_epochs, 
                                            verbose=1, mode='auto', min_lr=1.0e-6),
                          ModelCheckpoint(filepath=fn,
                                          monitor='val_loss', save_best_only=True)]
             
        # fit the model
        h = dnn_model.fit(x=self.xcalib, y=self.ycalib,
                          batch_size=dict_params['batch_size'],
                          epochs=self.max_epochs,
                          validation_split=0.25,
                          shuffle=True, verbose=1,
                          callbacks=callbacks_list)
                 
        validation_loss = np.min(h.history['val_loss'])
                 
        return validation_loss

x_calib1, x_calib2, y_calib1, y_calib2 = train_test_split(X_trn, y_trn_cat,
        train_size=0.5, shuffle=True, stratify=y_trn_cat)

data_dictionary = {}
data_dictionary['x_calib1'] = x_calib1
data_dictionary['y_calib1'] = y_calib1
data_dictionary['x_calib2'] = x_calib2
data_dictionary['y_calib2'] = y_calib2        

maximum_epochs = 1000
early_stop_epochs = 10
learning_rate_epochs = 5
optimizer_direction = 'minimize'
number_of_random_points = 3 # 25  # random searches to start opt process
maximum_time = 2 * 60 * 60 # 4*60*60  # seconds      
results_directory = "/kaggle/working/"

if not Path(results_directory).is_dir():
        os.mkdir(results_directory)

x_calib = data_dictionary['x_calib1']
y_calib = data_dictionary['y_calib1']
num_classes = y_calib.shape[1]
objective = Objective(x_calib, y_calib, results_directory,
                maximum_epochs, early_stop_epochs,
                learning_rate_epochs, num_classes)
     
# optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction=optimizer_direction,
        sampler=TPESampler(n_startup_trials=number_of_random_points))
    
study.optimize(objective, timeout=maximum_time)         
# %%



def objective(trial, xcalib, ycalib, dir_save,
                 max_epochs, early_stop, learn_rate_epochs,
                 number_of_classes):
    # def __init__(self, xcalib, ycalib, dir_save,
    #              max_epochs, early_stop, learn_rate_epochs,
    #              number_of_classes):
    #     self.xcalib = xcalib
    #     self.ycalib = ycalib
    #     self.max_epochs = max_epochs
    #     self.early_stop = early_stop
    #     self.dir_save = dir_save
    #     self.learn_rate_epochs = learn_rate_epochs
    #     #self.input_shape = input_shape
    #     self.number_of_classes = number_of_classes
 
    # def __call__(self, trial):        
    num_cnn_blocks = trial.suggest_int('num_cnn_blocks', 2, 4)
    # num_filters = trial.suggest_categorical('num_filters', [16, 32, 48, 64])
    # kernel_size = trial.suggest_int('kernel_size', 2, 4)
    num_dense_nodes = trial.suggest_categorical('num_dense_nodes',
                                                [64, 128, 512, 1024])
    dense_nodes_divisor = trial.suggest_categorical('dense_nodes_divisor',
                                                    [2, 4, 8])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 96, 128])
    drop_out=trial.suggest_discrete_uniform('drop_out', 0.05, 0.5, 0.05)
                        
    dict_params = {'num_cnn_blocks':num_cnn_blocks,
                #    'num_filters':num_filters,
                #    'kernel_size':kernel_size,
                    'num_dense_nodes':num_dense_nodes,
                    'dense_nodes_divisor':dense_nodes_divisor,
                    'batch_size':batch_size,
                    'drop_out':drop_out}
                                            
    # start of cnn coding   
    input_tensor = Input(shape=(50,1)) # Input(shape=self.input_shape)
        
    # 1st cnn block
    x = BatchNormalization()(input_tensor)
    x = Activation('relu')(x)
    # x = Conv2D(filters=dict_params['num_filters'],
    #            kernel_size=dict_params['kernel_size'],
    #            strides=1, padding='same')(x)
    #x = MaxPooling2D()(x)
    x = Dense(dict_params['num_dense_nodes'], activation='relu')(x)
    x = Dropout(dict_params['drop_out'])(x)
        
    # additional cnn blocks
    for iblock in range(dict_params['num_cnn_blocks'] - 1):
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        # x = Conv2D(filters=dict_params['num_filters'],
        #            kernel_size=dict_params['kernel_size'],
        #            strides=1, padding='same')(x)
        x = Dense(dict_params['num_dense_nodes'], activation='relu')(x)
        # x = MaxPooling2D()(x)
        x = Dropout(dict_params['drop_out'])(x)
                        
    # mlp
    x = Flatten()(x)
    x = Dense(dict_params['num_dense_nodes'], activation='relu')(x)
    x = Dropout(dict_params['drop_out'])(x)
    x = Dense(dict_params['num_dense_nodes']//dict_params['dense_nodes_divisor'], 
                activation='relu')(x)
    output_tensor = Dense(number_of_classes, activation='softmax')(x)
        
    # instantiate and compile model
    dnn_model = Model(inputs=input_tensor, outputs=output_tensor)
    opt = Adam(lr=0.00025)  # default = 0.001
    dnn_model.compile(loss='categorical_crossentropy',
                        optimizer=opt, metrics=['accuracy'])
        
        
    # callbacks for early stopping and for learning rate reducer
    fn = dir_save + str(trial.number) + '_dnn.h5'
    callbacks_list = [EarlyStopping(monitor='val_loss', patience=early_stop),                     
                        ReduceLROnPlateau(monitor='val_loss', factor=0.1, 
                                        patience=learn_rate_epochs, 
                                        verbose=2, mode='auto', min_lr=1.0e-6),
                        ModelCheckpoint(filepath=fn,
                                        monitor='val_loss', save_best_only=True)]
            
    # fit the model
    h = dnn_model.fit(x=xcalib, y=ycalib,
                        batch_size=dict_params['batch_size'],
                        epochs=max_epochs,
                        validation_split=0.25,
                        shuffle=True, verbose=2,
                        callbacks=callbacks_list)
                
    validation_loss = np.min(h.history['val_loss'])
                
    return validation_loss

# %%
x_calib1, x_calib2, y_calib1, y_calib2 = train_test_split(X_trn, y_trn_cat,
        train_size=0.5, shuffle=True, stratify=y_trn_cat)

data_dictionary = {}
data_dictionary['x_calib1'] = x_calib1
data_dictionary['y_calib1'] = y_calib1
data_dictionary['x_calib2'] = x_calib2
data_dictionary['y_calib2'] = y_calib2        

maximum_epochs = 1000
early_stop_epochs = 10
learning_rate_epochs = 5
optimizer_direction = 'minimize'
number_of_random_points = 3 # 25  # random searches to start opt process
maximum_time = 2 * 60 * 60 # 4*60*60  # seconds      
results_directory = "/kaggle/working/"
n_warmup_steps = 3
n_trials = 64

if not Path(results_directory).is_dir():
        os.mkdir(results_directory)

x_calib = data_dictionary['x_calib1']
y_calib = data_dictionary['y_calib1']
num_classes = y_calib.shape[1]

# optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction=optimizer_direction,
        sampler=TPESampler(n_startup_trials=number_of_random_points, multivariate=True), 
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=n_warmup_steps))
# ob = objective(trial, x_calib, y_calib, results_directory,
#                 maximum_epochs, early_stop_epochs,
#                 learning_rate_epochs, num_classes)
     
    
study.optimize(lambda trial: objective(trial, x_calib, y_calib, results_directory, maximum_epochs, early_stop_epochs, learning_rate_epochs, num_classes), timeout=maximum_time, n_trials=n_trials)  
# %%


