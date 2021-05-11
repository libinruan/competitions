# WARN This code doesn't work as expected: The DNNs embarked imputation works
# but cannot beat the default imputation with S port (which is 70% of the data)
# and the DNNs can only reach 63% accuracy.

# SOURCE "TPS Apr 2021 single LGBMRegressor" by hiro5299834 BIZEN 
# LINK https://www.kaggle.com/hiro5299834/tps-apr-2021-single-lgbmregressor
#%%

import pandas as pd
import numpy as np
import time
from matplotlib import pyplot
from datetime import datetime, timedelta
import lightgbm as lgb

from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.utils import class_weight

import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import callbacks

now = datetime.now()
timestamp = now.strftime("%b-%d-%Y-at-%H-%M")

def label_encoder(c):
    lc = LabelEncoder()
    return lc.fit_transform(c)

train_df = pd.read_csv('/kaggle/input/tabular-playground-series-apr-2021/train.csv')
test_df = pd.read_csv('/kaggle/input/tabular-playground-series-apr-2021/test.csv')
submission = pd.read_csv('/kaggle/input/tabular-playground-series-apr-2021/sample_submission.csv')

all_df = pd.concat([train_df, test_df])    

# SECTION Feature engineering
# Age fillna with mean age for each class
# NOTE HIRO changed it from average age to median age.
age_map = all_df[['Age', 'Pclass']].dropna().groupby('Pclass').median().to_dict() # TECH turn pd.Series to dict object.
all_df.Age = all_df.Age.fillna(all_df.Pclass.map(age_map['Age']))

# Cabin, fillna with 'X' and take first letter
all_df.Cabin = all_df.Cabin.fillna('X').map(lambda x: x[0].strip())

# Ticket, fillna with 'X', split string and take first split 
all_df.Ticket = all_df.Ticket.fillna('X').map(lambda x: str(x).split()[0] if len(str(x).split()) > 1 else 'X')

# Fare, fillna with mean value
all_df.Fare = all_df.Fare.fillna(all_df.Fare.mean())

# Embarked, fillna with 'X' value
# all_df.Embarked = all_df.Embarked.fillna('X') # remove it downward after the DNNs imputation on Embarked

# Name, take only surnames
all_df.Name = all_df.Name.map(lambda x: x.split(',')[0])

all_df['FamilySize'] = all_df['SibSp'] + all_df['Parch'] + 1


for i in range(1, all_df['Pclass'].nunique()+1):
    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    all_df['FC'+str(i)] = np.NaN
    all_df.loc[all_df['Pclass']==i, 'FC'+str(i)] = all_df.loc[all_df['Pclass']==i,'Fare']
    group_mean = all_df.loc[all_df['Pclass']==i, 'FC'+str(i)].mean()
    all_df.loc[:, 'FC'+str(i)] = all_df['FC'+str(i)].fillna(value=group_mean)
    all_df.loc[:, 'FC'+str(i)+'_pt'] = pt.fit_transform(all_df['FC'+str(i)].values.reshape(-1,1))

all_df['Fare'] = np.log(all_df['Fare'])

#%%
# SECTION DNN to impute missing embarked
emb_df = all_df.loc[:, ['Pclass', 'Name', 'Sex', 'SibSp', 'Parch', 'Fare', 'Ticket', 'Embarked']].copy()
emb_df['Fare'] = emb_df['Fare'].astype('float32')

emb_trn = emb_df.loc[emb_df['Embarked'].notnull(), :]
emb_tst = emb_df.loc[emb_df['Embarked'].isnull(), :]
emb_trn, emb_val = train_test_split(emb_trn, test_size=0.2)

map_embarked = {'S': 0, 'C': 1, 'Q': 2}
emb_X_trn, emb_Y_trn = emb_trn.drop('Embarked', axis=1), emb_trn['Embarked'].map(map_embarked).astype('int64')
emb_X_val, emb_Y_val = emb_val.drop('Embarked', axis=1), emb_val['Embarked'].map(map_embarked).astype('int64')
emb_X_tst = emb_tst.drop('Embarked', axis=1)

# for i in [emb_X_trn, emb_Y_trn, emb_X_val, emb_Y_val, emb_X_tst]: 
#     print(i.shape)
#     if not isinstance(i, pd.Series):
#         print(i.info())

# SOURCE - CLASS WEIGHT https://datascience.stackexchange.com/questions/13490/how-to-set-class-weights-for-imbalanced-classes-in-keras
# NOTE Alternative to weight classes or samples (if you one-hot encode labels of multiple classes) Oversampling https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#oversampling
class_weights = class_weight.compute_class_weight(
                    'balanced',
                    np.unique(emb_Y_trn),
                    emb_Y_trn) # LIPIN: The higher frequency of a class the lower its weight.

# Ref: https://www.tensorflow.org/tutorials/structured_data/preprocessing_layers#split_the_dataframe_into_train_validation_and_test
def df_to_dataset(dataframe, labels=None, shuffle=True, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds

batch_size = 256
train_ds = df_to_dataset(emb_X_trn, emb_Y_trn, batch_size=batch_size)  
valid_ds = df_to_dataset(emb_X_val, emb_Y_val, batch_size=batch_size)

def get_normalization_layer(name, dataset):
    # Create a Normalization layer for our feature.
    normalizer = preprocessing.Normalization()
    # Prepare a Dataset that only yields our feature.
    feature_ds = dataset.map(lambda x, y: x[name])
    # Learn the statistics of the data.
    normalizer.adapt(feature_ds)
    return normalizer

all_inputs = []
encoded_features = []

# Numeric features.
for header in ['Fare']:
    numeric_col = tf.keras.Input(shape=(1,), name=header)
    normalization_layer = get_normalization_layer(header, train_ds) # look for the designated columns
    encoded_numeric_col = normalization_layer(numeric_col)

    all_inputs.append(numeric_col)
    encoded_features.append(encoded_numeric_col)

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

# Categorical features encoded as integers.
for header in ['Pclass', 'SibSp', 'Parch']:
    categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='int64')
    encoding_layer = get_category_encoding_layer(header, train_ds, dtype='int64', max_tokens=5) # look for the designated columns
    encoded_categorical_col = encoding_layer(categorical_col)

    all_inputs.append(categorical_col)
    encoded_features.append(encoded_categorical_col)    

for header in ['Name', 'Sex', 'Ticket']:
    categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='string')
    encoding_layer = get_category_encoding_layer(header, train_ds, dtype='string', max_tokens=5) # look for the designated columns
    encoded_categorical_col = encoding_layer(categorical_col)
    
    all_inputs.append(categorical_col)
    encoded_features.append(encoded_categorical_col)    

# WARN Received a label value of 2 which is outside the valie range ... https://stackoverflow.com/questions/44151760/received-a-label-value-of-1-which-is-outside-the-valid-range-of-0-1-python
all_features = tf.keras.layers.concatenate(encoded_features)
n = 32
x = tf.keras.layers.Dense(n, activation="relu")(all_features)
x = tf.keras.layers.Dropout(0.1)(x)
x = tf.keras.layers.Dense(n, activation="relu")(x)
x = tf.keras.layers.Dropout(0.1)(x)
x = tf.keras.layers.Dense(n, activation="relu")(x) # LeakyReLU(alpha=0.2)(x)
x = tf.keras.layers.Dropout(0.1)(x)
output = tf.keras.layers.Dense(3)(x)
model = tf.keras.Model(all_inputs, output)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])  
Verbose = 1
# NOTE How to set class weights for imbalanced classes in Keras? ... https://datascience.stackexchange.com/questions/13490/how-to-set-class-weights-for-imbalanced-classes-in-keras
unique_Y_labels = np.unique(emb_Y_trn)
class_weights_array = class_weight.compute_class_weight('balanced', np.unique(emb_Y_trn), emb_Y_trn)
class_weights = dict(zip(unique_Y_labels, class_weights_array))

es = callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.000001, patience=50, verbose=Verbose, mode='max', baseline=None, restore_best_weights=True)
sb = callbacks.ModelCheckpoint('./nn_model.w8', save_weights_only=True, save_best_only=True, verbose=Verbose, monitor='val_accuracy',mode='max')
plateau  = callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=10, verbose=Verbose, mode='max', min_delta=0.0001, cooldown=0, min_lr=1e-7)

model.fit(train_ds, epochs=200, validation_data=valid_ds, class_weight=class_weights, callbacks=[es, sb, plateau], verbose=2)

# tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")

# !SECTION