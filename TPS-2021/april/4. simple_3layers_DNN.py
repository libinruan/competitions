#%%
# SOURCE - [Keras Deep Learning on Titanic data](https://www.kaggle.com/stefanbergstein/keras-deep-learning-on-titanic-data)
# This version involves the least feature engineering.

import os
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
from datetime import datetime, timedelta
import optuna
from collections import OrderedDict
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, train_test_split

from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers, optimizers, callbacks, utils, losses, metrics, backend as K
import tensorflow_addons as tfa
from sklearn import metrics as skmetrics, preprocessing
# from tensorflow.keras.layers.experimental import preprocessing
print(tf.__version__)

RunningInLinux = os.environ.get('PWD') in ['/kaggle/working', '/root', '/']
if RunningInLinux:
    os.chdir('/kaggle/working')
    train_df = pd.read_csv('../input/tabular-playground-series-apr-2021/train.csv', index_col=0)
    test_df = pd.read_csv('../input/tabular-playground-series-apr-2021/test.csv', index_col=0)
    sample_submission = pd.read_csv('../input/tabular-playground-series-apr-2021/sample_submission.csv')
# else:
#     os.chdir('G:\kagglePlayground')
#     train_df = pd.read_csv('train.csv', index_col=0)
#     test_df = pd.read_csv('test.csv', index_col=0)
#     sample_submission = pd.read_csv('sample_submission.csv') 

print(len(train_df), 'train examples')
print(len(test_df), 'test examples')

#%% SECTION baseline model with minor preprocessing
print(train_df.isnull().sum())

def prep_data(df):
    # Drop unwanted features
    df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
    
    # Fill missing data: Age and Fare with the mean, Embarked with most frequent value
    df[['Age']] = df[['Age']].fillna(value=df[['Age']].mean())
    df[['Fare']] = df[['Fare']].fillna(value=df[['Fare']].mean())
    df[['Embarked']] = df[['Embarked']].fillna(value=df['Embarked'].value_counts().idxmax())
    
    # Convert categorical  features into numeric
    df['Sex'] = df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
      
    # Convert Embarked to one-hot
    enbarked_one_hot = pd.get_dummies(df['Embarked'], prefix='Embarked')
    df = df.drop('Embarked', axis=1)
    df = df.join(enbarked_one_hot)

    return df

train_df_0 = prep_data(train_df)
print(train_df_0.isnull().sum())

# X contains all columns except 'Survived'  
X = train_df_0.drop(['Survived'], axis=1).values.astype(float)

# It is almost always a good idea to perform some scaling of input values when using neural network models (jb).
scale = StandardScaler()
X = scale.fit_transform(X)

# Y is just the 'Survived' column
Y = train_df_0['Survived'].values


# ANCHOR amateur version
def create_model_0(optimizer='adam', init='uniform'):
    # create model
    if verbose: print(f"**Create model with optimizer: {optimizer}; init: {init}")
    model = Sequential()
    model.add(Dense(16, input_dim=X.shape[1], kernel_initializer=init, activation='relu'))
    model.add(Dense(8, kernel_initializer=init, activation='relu'))
    model.add(Dense(4, kernel_initializer=init, activation='relu'))
    model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

run_gridsearch = False

if run_gridsearch:
    
    start_time = time.time()
    if verbose: print (time.strftime( "%H:%M:%S " + "GridSearch started ... " ) )
    optimizers = ['rmsprop', 'adam']
    inits = ['glorot_uniform', 'normal', 'uniform']
    epochs = [50, 100, 200, 400]
    batches = [5, 10, 20]
    
    model = KerasClassifier(build_fn=create_model_0, verbose=verbose)
    
    param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=inits)
    grid = GridSearchCV(estimator=model, param_grid=param_grid)
    grid_result = grid.fit(X, Y)
    
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    if verbose: 
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
        elapsed_time = time.time() - start_time  
        print ("Time elapsed: ", timedelta(seconds=elapsed_time))
        
    best_epochs = grid_result.best_params_['epochs']
    best_batch_size = grid_result.best_params_['batch_size']
    best_init = grid_result.best_params_['init']
    best_optimizer = grid_result.best_params_['optimizer']
    
else:
    # pre-selected paramters # screenshot https://i.postimg.cc/q7QmBNfT/2021-04-05-at-10-03-19.png 
    best_epochs = 200
    best_batch_size = 5
    best_init = 'glorot_uniform'
    best_optimizer = 'rmsprop'    

Verbose = 1
# Create a classifier with best parameters
model_pred = KerasClassifier(build_fn=create_model_0, 
                             optimizer=best_optimizer, 
                             init=best_init, 
                             epochs=best_epochs, 
                             batch_size=best_batch_size, 
                             verbose=Verbose)
model_pred.fit(X, Y)

# # Read test data
# test_df = pd.read_csv(file_test,index_col=0)
# Prep and clean data
test_df_0 = prep_data(test_df)
# Create X_test
x_test_0 = test_df_0.values.astype(float)
# Scaling
x_test_0 = scale.transform(x_test_0)

# Predict 'Survived'
prediction = model_pred.predict(x_test_0)
submission = pd.DataFrame({
    'PassengerId': test_df.index,
    'Survived': prediction[:,0],
})
submission.sort_values('PassengerId', inplace=True)    
submission.to_csv('submission-simple-cleansing.csv', index=False)
# !SECTION baseline model with minor preprocessing    

#%% SECTION simple with callbacksS neurons 300 per lay
train_df_1 = prep_data(train_df)
print(train_df_1.isnull().sum())
# X contains all columns except 'Survived'  
X = train_df_1.drop(['Survived'], axis=1).values.astype(float)
# It is almost always a good idea to perform some scaling of input values when using neural network models (jb).
scale = StandardScaler()
X = scale.fit_transform(X)
# Y is just the 'Survived' column
Y = train_df_1['Survived'].values

Verbose = 1
def create_model_1():    
    inputs = layers.Input(shape=(X.shape[1],))
    x = layers.Dense(300, activation='relu')(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(300, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)
    y = layers.Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=y)
    return model  
model = create_model_1()
learning_rate = 1e-3
label_smoothing = 0.0 
model.compile(
    optimizer=tfa.optimizers.SWA(tf.keras.optimizers.Adam(learning_rate=learning_rate)),
    loss=losses.BinaryCrossentropy(label_smoothing=label_smoothing),
    metrics=['accuracy'] 
)
es = callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.000001, patience=10, verbose=Verbose, mode='max', baseline=None, restore_best_weights=True)
sb = callbacks.ModelCheckpoint('./nn_model.w8', save_weights_only=True, save_best_only=True, verbose=Verbose, monitor='val_accuracy',mode='max')
plateau  = callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=2, verbose=Verbose, mode='max', min_delta=0.0001, cooldown=0, min_lr=1e-7)
x_train_1, x_val_1, y_train_1, y_val_1 = train_test_split(X, Y, random_state=456)
model.fit(x_train_1,
          y_train_1,
          validation_data=(x_val_1, y_val_1),
          verbose=Verbose,
          batch_size=1024,
          callbacks=[es, sb, plateau],
          epochs=100
)

# prediction
x_test_1 = prep_data(test_df)
x_test_1 = x_test_1.values.astype(float)
x_test_1 = scale.transform(x_test)
prediction = model.predict(x_test_1)
submission = pd.DataFrame({
    'PassengerId': test_df.index,
    'Survived': (prediction[:,0] > 0.5).astype(int),
})
submission.sort_values('PassengerId', inplace=True)    
submission.to_csv('submission-nonamateur-style-dnn.csv', index=False)
# screenshot: https://i.postimg.cc/BZN6GhWj/2021-04-05-at-11-17-04.png
# CV: 0.77268, LB: 0.70234 (implies OVERFITTING!)

# !SECTION


# %%
# SECTION simpler with callbacks and neurons at most 16 per lay
train_df_2 = prep_data(train_df)
print(train_df_2.isnull().sum())
# X contains all columns except 'Survived'  
X = train_df_2.drop(['Survived'], axis=1).values.astype(float)
# It is almost always a good idea to perform some scaling of input values when using neural network models (jb).
scale = StandardScaler()
X = scale.fit_transform(X)
# Y is just the 'Survived' column
Y = train_df_2['Survived'].values

Verbose = 1
def create_model_1():    
    inputs = layers.Input(shape=(X.shape[1],))
    x = layers.Dense(16, activation='relu')(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(8, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)
    y = layers.Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=y)
    return model  
model = create_model_1()
learning_rate = 1e-3
label_smoothing = 0.0 
model.compile(
    optimizer=tfa.optimizers.SWA(tf.keras.optimizers.Adam(learning_rate=learning_rate)),
    loss=losses.BinaryCrossentropy(label_smoothing=label_smoothing),
    metrics=['accuracy'] 
)
es = callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.000001, patience=10, verbose=Verbose, mode='max', baseline=None, restore_best_weights=True)
sb = callbacks.ModelCheckpoint('./nn_model.w8', save_weights_only=True, save_best_only=True, verbose=Verbose, monitor='val_accuracy',mode='max')
plateau  = callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=2, verbose=Verbose, mode='max', min_delta=0.0001, cooldown=0, min_lr=1e-7)
x_train_2, x_val_2, y_train_2, y_val_2 = train_test_split(X, Y, random_state=456)
model.fit(x_train_2,
          y_train_2,
          validation_data=(x_val_2, y_val_2),
          verbose=Verbose,
          batch_size=1024,
          callbacks=[es, sb, plateau],
          epochs=100
)

# prediction
x_test_2 = prep_data(test_df)
x_test_2 = x_test_2.values.astype(float)
x_test_2 = scale.transform(x_test_2)
prediction = model.predict(x_test_2)
submission = pd.DataFrame({
    'PassengerId': test_df.index,
    'Survived': (prediction[:,0] > 0.5).astype(int),
})
submission.sort_values('PassengerId', inplace=True)    
submission.to_csv('submission-simpler-style-dnn.csv', index=False)
# screenshot: https://i.postimg.cc/0jNXKSgx/2021-04-05-at-15-51-32.png
# CV: 0.7689; LB: 0.78832 
# !SECTION simpler with callbacks and neurons at most 16 per lay
# %%
