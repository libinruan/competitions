#%%
"""
Source: https://www.kaggle.com/remekkinas/tps-5-hydra-keras-weighted-embedding-mix

Title: `[TPS-5] Hydra - Keras - weighted, Embedding, mix`

Author: https://www.kaggle.com/remekkinas
"""

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

start_time = timer()        
        
from google.cloud import storage

project = "strategic-howl-305522"
bucket_name = "gcs-station-168"           
storage_client = storage.Client(project=project)
bucket = storage_client.bucket(bucket_name)


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

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

RANDOM_STATE = 2021

construction_mode = True
if construction_mode == True:
    NUM_HEADS = 2 # 05281402 number of heads
    HEAD_EPOCHS = 50 # 05281411 head epochs
    HYDRA_EPOCHS = 30 # 05281410 hydra epochs
    TRAIN_VERBOSE = 2 # https://tinyurl.com/yzfkfoq6
    NUM_CLASS = 4
    
train = pd.read_csv("/kaggle/input/tabular-playground-series-may-2021/train.csv", index_col = 'id')
test = pd.read_csv("/kaggle/input/tabular-playground-series-may-2021/test.csv", index_col = 'id')

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
                             patience = 10, # 2,
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

# Lets define different architecture (brains) for each head :) 
# Let each hydra think differently.

# You can play with configurations - i just randomly created 4 nn architectures

# ANCHOR MODEL

h_model_1 = [
        Dense(64, input_dim = NUM_FEATURES, activation='relu', kernel_initializer='he_uniform'), # TODO 05281336 he_uniform, relu
        Dropout(0.3),
        Dense(32, activation='relu', kernel_initializer='he_uniform'),
        Dropout(0.2),
        Dense(128, activation='softmax', kernel_initializer='he_uniform'),
        Dropout(0.2),
        Dense(NUM_CLASS, activation='softmax', kernel_initializer='he_uniform')
    ]

h_model_1A = [
        Dense(10, input_dim = NUM_FEATURES, activation='relu', kernel_initializer='he_uniform'), 
        Dropout(0.3),
        Dense(20, activation='relu', kernel_initializer='he_uniform'),
        Dropout(0.2),
        Dense(10, activation='softmax', kernel_initializer='he_uniform'),
        Dropout(0.2),
        Dense(NUM_CLASS, activation='softmax', kernel_initializer='he_uniform')
    ]

h_model_2 = [ # not as good as model 1. too complex
        Dense(150, input_dim = NUM_FEATURES, activation='relu', kernel_initializer='he_uniform'),
        Dropout(0.3),
        Dense(150, activation='relu', kernel_initializer='he_uniform'),
        Dropout(0.2),
        Dense(150, activation='softmax', kernel_initializer='he_uniform'),
        Dropout(0.2),
        Dense(NUM_CLASS, activation='softmax', kernel_initializer='he_uniform')
    ]


h_model_3 = [ # not as good as model 1. too complex
        Dense(50, input_dim = NUM_FEATURES, activation='relu', kernel_initializer='he_uniform'),
        Dropout(0.25),
        Dense(25, activation='relu', kernel_initializer='he_uniform'),
        Dropout(0.25),
        Dense(10, activation='softmax', kernel_initializer='he_uniform'),
        Dropout(0.2),
        Dense(NUM_CLASS, activation='softmax', kernel_initializer='he_uniform')
    ]

h_model_4 = [ # not as good as model 1. too complex
        Dense(512, input_dim = NUM_FEATURES, activation='relu', kernel_initializer='he_uniform'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(256, activation='relu', kernel_initializer='he_uniform'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(128, activation='relu', kernel_initializer='he_uniform'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(NUM_CLASS, activation='softmax', kernel_initializer='he_uniform')
    ]


h_model_5 = [
    Embedding(100, 4, input_length = NUM_FEATURES), # https://tinyurl.com/yjs55ys6
    Flatten(),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(NUM_CLASS, activation='softmax')
]

h_model_6 = [
    Embedding(100, 16, input_length = NUM_FEATURES),
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(40, activation='relu'),
    BatchNormalization(), 
    Dropout(0.25),
    Dense(NUM_CLASS, activation='softmax')
]

h_model_6A = [ # not as good as #6
    Embedding(100, 16, input_length = NUM_FEATURES),
    Flatten(),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(32, activation='relu'),
    BatchNormalization(), 
    Dropout(0.25),
    Dense(NUM_CLASS, activation='softmax')
]

h_model_6B = [ # base on #6
    Embedding(100, 16, input_length = NUM_FEATURES),
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(), 
    Dropout(0.25),
    Dense(NUM_CLASS, activation='softmax')
]

h_model_6C = [ # base on #6
    Embedding(100, 16, input_length = NUM_FEATURES),
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(), 
    Dropout(0.25),
    Dense(NUM_CLASS, activation='softmax')
]

h_model_6D = [ # base on #6
    Embedding(100, 16, input_length = NUM_FEATURES),
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(), 
    Dropout(0.25),
    Dense(NUM_CLASS, activation='softmax')
]

head_nn_models = [h_model_1A, h_model_6D] # TODO number of heads        

# custom  weighted categorical crossentropy for Keras

def weight_categorical_crossentropy(weights):
    
    weights = K.variable(weights)

    def loss(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss

class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train.target), y_train.target)
print(class_weights)
# Here is our playground for testing with weights

class_weights = [1.5, 0.5, 1, 1] # TODO use api result instead 05281347

def fit_hydra_head_model(fX_train, fy_train, fX_valid, fy_valid, n_model):
    
    oy_train = to_categorical(fy_train) # https://www.tensorflow.org/api_docs/python/tf/keras/utils/to_categorical
    oy_valid = to_categorical(fy_valid)
    
    model = Sequential(head_nn_models[n_model])
    
    if WEIGHTED_TRAINING: # WEIGHTED TRAINING 05281348 
        model.compile(loss = weight_categorical_crossentropy(class_weights), optimizer = tf.keras.optimizers.Adam(), metrics=['accuracy'])
    else:
        model.compile(loss = "categorical_crossentropy", optimizer = tf.keras.optimizers.Adam(), metrics=['accuracy'])
        
    history = model.fit(fX_train, oy_train, epochs = HEAD_EPOCHS, 
                        verbose = TRAIN_VERBOSE, 
                        validation_data=(fX_valid, oy_valid), 
                        callbacks = [es, plateau], 
                        batch_size = 128) # TODO 05281349 batch size
    
    return model, history

def plot_model_learning(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    
# inspect
print(f'{len(head_nn_models)}, {NUM_HEADS}')    

def train_hydra_df():
    if NUM_HEADS <= len(head_nn_models):     
        for i in tqdm(range(NUM_HEADS)):
            print(f'\n>>>>>>>>>>> Training head {i+1} ... <<<<<<<<<<<')
            model, history = fit_hydra_head_model(X_train, y_train, X_valid, y_valid, i)
            hydra_heads_models_df.append(model)
            print("\n")
            plot_model_learning(history)
    else:
        print("ERROR: param NUM_HEADS > list of defined model heads [head_nn_models].")
        
# If you want to turn on weighted training with custom weighted entropy loss change to True

WEIGHTED_TRAINING = False # TODO 05281359 weighted training        

hydra_heads_models_df = []
train_hydra_df()


from keras import regularizers

def define_hydra_model(heads):
    for i in range(len(heads)):
        model = heads[i]
        
        # Lets freeze all head layers 
        for layer in model.layers:
            layer.trainable = False
            layer._name = 'hydra_head' + str(i+1) + '_' + layer.name

    
    hydra_visible = [model.input for model in heads]
    hydra_outputs = [model.output for model in heads]
    merge = concatenate(hydra_outputs)
    
    # Create Hydra heart layers and train them 
    
    hidden = Dense(NUM_HEADS * NUM_CLASS, activation='relu')(merge)
    #x = Dense(10, activation='softmax', kernel_initializer='he_uniform')(hidden)
    output = Dense(4, activation='softmax')(hidden)
    
    # Architecture will be examined later ... below you can find my experiment
    #x = BatchNormalization()(hidden)
    #x = Dropout(0.3)(x)
    #x = Dense(32, activation='relu')(x)
    #x = BatchNormalization()(x)
    #x = Dropout(0.2)(x)
    # output = Dense(4, activation='softmax')(x)
    
    model = Model(inputs = hydra_visible, outputs = output)
    
    if WEIGHTED_TRAINING:
        model.compile(loss = weight_categorical_crossentropy(class_weights), optimizer='adam',  metrics=['accuracy'])
    else:
        model.compile(loss = "categorical_crossentropy", optimizer='adam',  metrics=['accuracy'])
    
    return model

# Define Hydra model
hydra_model_df = define_hydra_model(hydra_heads_models_df)

from keras import regularizers

def define_hydra_model(heads):
    for i in range(len(heads)):
        model = heads[i]
        
        # Lets freeze all head layers 
        for layer in model.layers:
            layer.trainable = False
            layer._name = 'hydra_head' + str(i+1) + '_' + layer.name

    
    hydra_visible = [model.input for model in heads]
    hydra_outputs = [model.output for model in heads]
    merge = concatenate(hydra_outputs)
    
    # Create Hydra heart layers and train them 
    
    hidden = Dense(NUM_HEADS * NUM_CLASS, activation='relu')(merge)
    #x = Dense(10, activation='softmax', kernel_initializer='he_uniform')(hidden)
    output = Dense(4, activation='softmax')(hidden)
    
    # Architecture will be examined later ... below you can find my experiment
    #x = BatchNormalization()(hidden)
    #x = Dropout(0.3)(x)
    #x = Dense(32, activation='relu')(x)
    #x = BatchNormalization()(x)
    #x = Dropout(0.2)(x)
    # output = Dense(4, activation='softmax')(x)
    
    model = Model(inputs = hydra_visible, outputs = output)
    
    if WEIGHTED_TRAINING:
        model.compile(loss = weight_categorical_crossentropy(class_weights), optimizer='adam',  metrics=['accuracy'])
    else:
        model.compile(loss = "categorical_crossentropy", optimizer='adam',  metrics=['accuracy'])
    
    return model

# Define Hydra model
hydra_model_df = define_hydra_model(hydra_heads_models_df)

def prepare_input_data(model, X_in):
    X = [X_in for _ in range(len(model.input))]
    return X

y_train = to_categorical(y_train)
y_valid = to_categorical(y_valid)

# Now it is time to fit Hydras heart .... to see the world througt emotion :)  
print("\n\n>>>>>>>>>>  Fit  HydraNet <<<<<<<<<<<<")
history = hydra_model_df.fit(prepare_input_data(hydra_model_df, X_train), 
                             y_train, 
                             epochs = HYDRA_EPOCHS, 
                             verbose = TRAIN_VERBOSE, 
                             validation_data=(prepare_input_data(hydra_model_df, X_valid), y_valid),
                             callbacks = [es, plateau], 
                             batch_size = 128)
plot_model_learning(history)


# -----------

# Lets look what Hydra sees
y_valud_preds_df = hydra_model_df.predict(prepare_input_data(hydra_model_df, X_valid), verbose=0)

base_class = np.argmax(y_valid, axis = 1)
preds = np.argmax(y_valud_preds_df, axis = 1)

# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
sns.heatmap(pd.DataFrame(confusion_matrix(base_class, preds)), annot=True, linewidths=.5, fmt="d");
print("f1: {:0.6f} log loss: {:0.6f}".format(f1_score(base_class, preds, average='macro'), log_loss(y_valid, y_valud_preds_df)))
print(timer(start_time))

sub_preds_df = hydra_model_df.predict(prepare_input_data(hydra_model_df, test), verbose=0)
predictions_df = pd.DataFrame(sub_preds_df, columns = ["Class_1", "Class_2", "Class_3", "Class_4"])

blend_l1 = pd.read_csv("/kaggle/input/tps05blender-v2/tps05-remek-blender_v2.csv")

output = predictions_df.copy()
output["Class_1"] = (predictions_df.Class_1 * 0.3 + blend_l1.Class_1 * 0.7)
output["Class_2"] = (predictions_df.Class_2 * 0.3 + blend_l1.Class_2 * 0.7)
output["Class_3"] = (predictions_df.Class_3 * 0.3 + blend_l1.Class_3 * 0.7) 
output["Class_4"] = (predictions_df.Class_4 * 0.3 + blend_l1.Class_4 * 0.7) 

sub = pd.read_csv("/kaggle/input/tabular-playground-series-may-2021/sample_submission.csv")

predictions_df = pd.DataFrame(output, columns = ["Class_1", "Class_2", "Class_3", "Class_4"])
predictions_df['id'] = sub['id']
predictions_df.to_csv(f"{dtnow()}-TPS-05-hydra_df_blended_submission-1A6C.csv", index = False)

predictions_df.drop("id", axis=1).describe().T.style.bar(subset=['mean'], color='#205ff2')\
                            .background_gradient(subset=['std'], cmap='Reds')\
                            .background_gradient(subset=['50%'], cmap='coolwarm')

# output screnshot: https://i.postimg.cc/zB57JmnC/2021-05-28-at-20-44-54.png (f1: 0.228235 log loss: 1.087695)
# kaggle output: https://i.postimg.cc/zBjsNFRf/2021-05-28-16-31-41.png
# %%
