# LINK https://towardsdatascience.com/using-neural-networks-with-embedding-layers-to-encode-high-cardinality-categorical-variables-c1b872033ba2

#%%
import pandas as pd
import numpy as np

# truncate numpy & pd printouts to 2 decimals 
np.set_printoptions(precision=3)
pd.set_option('precision', 3)

# create seed for all np.random functions
np_random_state = np.random.RandomState(150)

def f(price, weight, s1, s2):
    return price + s1 + s2 * weight + np_random_state.normal(0, 1)

def generate_secret_data(n, s1_bins, s2_bins, s3_bins):
    data = pd.DataFrame({
        's1': np_random_state.choice(range(1, s1_bins+1), n),
        's2': (np_random_state.choice(range(1, s2_bins+1), n) - s2_bins/2).astype(int),
        's3': np_random_state.choice(range(1, s3_bins+1), n),
        'price': np_random_state.normal(3, 1, n),
        'weight': np_random_state.normal(2, 0.5, n)
    })
    # compute size/y using function
    data['y'] = f(data['price'], data['weight'], data['s1'], data['s2'])
    secret_cols = ['s1', 's1', 's1']
    data[secret_cols] = data[secret_cols].apply(lambda x: x.astype('category')) 
    return data

import seaborn as sns
import matplotlib.pyplot as plt
data = generate_secret_data(n=300, s1_bins=3, s2_bins=6, s3_bins=2)
data.head(10)  
def show_scatter():  
    g = sns.FacetGrid(data, col='s2', hue='s1', col_wrap=3, height=3.5);
    g = g.map_dataframe(plt.scatter, 'weight', 'y');
    plt.show()
    g = sns.FacetGrid(data, col='s2', hue='s1', col_wrap=3, height=3.5); g = g.map_dataframe(plt.scatter, 'price', 'y'); 
    plt.show()
# show_scatter()

import hashlib
def generate_hash(*args):
    s = '_'.join([str(x) for x in args])
    return hashlib.md5(s.encode()).hexdigest()[-4:]

def generate_data(n, s1_bins, s2_bins, s3_bins):
    data = generate_secret_data(n, s1_bins, s2_bins, s3_bins)

    # generate product id from (s1, s3), supplier id from (s2, s3)
    data['product_id'] = data.apply(lambda row: generate_hash(row['s1'], row['s3']), axis=1)
    data['supplier_id'] = data.apply(lambda row: generate_hash(row['s2'], row['s3']), axis=1)
    
    # drop  secret features
    data = data.drop(['s1', 's2', 's3'], axis=1)

    return data[['product_id', 'supplier_id', 'price', 'weight', 'y']]

data = generate_data(n=300, s1_bins=4, s2_bins=1, s3_bins=2)
data.head(10)

# from sklearn.model_selection import train_test_split
# N = 100000
# S1_BINS = 30
# S2_BINS = 3
# S3_BINS = 50
# data = generate_data(n=N, s1_bins=S1_BINS, s2_bins=S2_BINS, s3_bins=S3_BINS)
# data.describe()

from sklearn.model_selection import train_test_split
x = data[['product_id', 'supplier_id', 'price', 'weight']]
y = data[['y']]
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=456)

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
# def create_one_hot_preprocessor():
#     return ColumnTransformer([
#         ('one_hot_encoder', OneHotEncoder(sparse=False, handle_unknown='ignore'), ['product_id', 'supplier_id']),
#         ('standard_scaler', StandardScaler(), ['price', 'weight'])]
#     )
    
# sample_data = data.head(5)
# sample_data    

# one_hot_preprocessor = create_one_hot_preprocessor()
# one_hot_preprocessor.fit_transform(sample_data)

from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
# baseline_pipeline = Pipeline(steps=[
#     ('preprocessor', create_one_hot_preprocessor()),
#     ('model', Ridge())
# ])

# baseline_pipeline.fit(x_train, y_train);
# y_pred_baseline = baseline_pipeline.predict(x_test)

# mean_squared_error(y_test, y_pred_baseline)
# ## 0.34772562548572583
# mean_absolute_error(y_test, y_pred_baseline)
## 0.3877675893786461

# %%

from sklearn.base import BaseEstimator, TransformerMixin
from pandas.api.types import is_string_dtype
from collections import OrderedDict

class ColumnEncoder(BaseEstimator, TransformerMixin):
    def __init__(self): # f, cols=None):
        # self.columns = cols
        self.columns = None
        self.maps = dict()

    def transform(self, X):
        X_copy = X.copy()
        for col in self.columns:
            assert self.maps[col]
            # encode value x of col via dict entry self.maps[col][x]+1 if present, otherwise 0
            X_copy.loc[:,col] = X_copy.loc[:,col].apply(lambda x: self.maps[col].get(x, -1)+1)
        return X_copy

    def inverse_transform(self, X):
        X_copy = X.copy()
        for col in self.columns:
            values = list(self.maps[col].keys())
            # find value in ordered list and map out of range values to None
            X_copy.loc[:,col] = [values[i-1] if 0<i<=len(values) else None for i in X_copy[col]]
        return X_copy

    def fit(self, X, y=None):
        # only apply to string type columns
        self.columns = [col for col in X.columns if is_string_dtype(X[col])]
        for col in self.columns:
            # print(col, ' : ', self.columns)
            self.maps[col] = OrderedDict({value: num for num, value in enumerate(sorted(set(X[col])))})
        return self

ce = ColumnEncoder()        
ce.fit_transform(x_train)
# ce.inverse_transform(ce.transform(x_train))
# %%
# unknown_data = pd.DataFrame({
#     'product_id': ['!§$%&/()'],
#     'supplier_id': ['abcdefg'],
#     'price': [10],
#     'weight': [20],
#   })
# ce.transform(unknown_data)
# ##    product_id  supplier_id  price  weight
# ## 0           0            0     10      20
# ce.inverse_transform(ce.transform(unknown_data))
# ##   product_id supplier_id  price  weight
# ## 0       None        None     10      20

# %% ANCHOR embedding transformer
class EmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, col_indices):
        self.cols = col_indices

    def fit(self, X, y=None):
        self.other_cols = [col for col in range(X.shape[1]) if col not in self.cols] # in this case, ['price', 'weight]
        return self

    def transform(self, X):
        if len(self.cols) == 0:
            return X
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        X_new = [X[:,[col]] for col in self.cols]
        X_new.append(X[:,self.other_cols])
        return X_new

emb = EmbeddingTransformer(col_indices=[0, 1])
emb.fit_transform(x_train.head(5))

# %% ANCHOR pipeline that contains embedding transformer.
def create_embedding_preprocessor():
    encoding_preprocessor = ColumnTransformer([
        ('column_encoder', ColumnEncoder(), ['product_id', 'supplier_id']),
        ('standard_scaler', StandardScaler(), ['price', 'weight'])
    ])
    embedding_preprocessor = Pipeline(steps=[
          ('encoding_preprocessor', encoding_preprocessor),
          # NOTE careful here, column order matters because the output of ColumnTransformer is arranged in the order of its transformers.
          ('embedding_transformer', EmbeddingTransformer(col_indices=[0, 1])),
      ])
    return embedding_preprocessor
embedding_preprocessor = create_embedding_preprocessor()
embedding_preprocessor.fit(x_train);
embedding_preprocessor.transform(x_train)
# %%
from tensorflow.python.keras.layers import Input, Dense, Reshape, Embedding, concatenate, Dropout, Flatten
from tensorflow.python.keras.layers.merge import Dot
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras import Model

# NOTE The processing order is determined by you in the Model's inputs argument.
def create_model(embedding1_vocab_size = 7,
                 embedding1_dim = 3,
                 embedding2_vocab_size = 7,
                 embedding2_dim = 3):
    embedding1_input = Input((1,))
    embedding1 = Embedding(input_dim=embedding1_vocab_size,
                           output_dim=embedding1_dim,
                           name='embedding1')(embedding1_input)

    embedding2_input = Input((1,))
    embedding2 = Embedding(input_dim=embedding2_vocab_size,
                           output_dim=embedding2_dim,
                           name='embedding2')(embedding2_input)

    # LINK flatten https://keras.io/api/layers/reshaping_layers/flatten/
    # LINK concatenate https://keras.io/api/layers/merging_layers/concatenate/
    flatten = Flatten()(concatenate([embedding1, embedding2]))

    normal_input = Input((2,))

    merged_input = concatenate([flatten, normal_input], axis=-1)

    dense1 = Dense(32, activation='relu')(merged_input)
    dropout1 = Dropout(0.1)(dense1)
    dense2 = Dense(32, activation='relu')(dropout1)
    dropout2 = Dropout(0.1)(dense2)

    output = Dense(1, activation='linear')(dropout2)

    model = Model(inputs=[embedding1_input, embedding2_input, normal_input], outputs=output)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    return model

# %%
# ANCHOR It's important to identify unknown values!
C1_SIZE = x_train['product_id'].nunique()
C2_SIZE = x_train['supplier_id'].nunique()
x_train = x_train.copy()
n = x_train.shape[0]

# set a fair share to unknown
idx1 = np_random_state.randint(0, n, int(n / C1_SIZE))
x_train.iloc[idx1,0] = '(unknown)'
idx2 = np_random_state.randint(0, n, int(n / C2_SIZE))
x_train.iloc[idx2,1] = '(unknown)'

x_train.sample(10, random_state=1234)

x_train_emb = embedding_preprocessor.transform(x_train)
x_test_emb = embedding_preprocessor.transform(x_test)

import tensorflow as tf
# ANCHOR Use a custom model
model = create_model(embedding1_vocab_size=C1_SIZE+1, 
                     embedding2_vocab_size=C2_SIZE+1)
# tf.keras.utils.plot_model(
#     model,
#     to_file='../keras_embeddings_model.png',
#     show_shapes=True,
#     show_layer_names=True,
# )
num_epochs = 50
model.fit(
    x_train_emb,
    y_train,
    validation_data=(x_test_emb, y_test),
    epochs=num_epochs,
    batch_size=64,
    verbose=0,
);
y_pred_emb = model.predict(x_test_emb)
mean_squared_error(y_pred_emb, y_test)

# To use the trained embedding in other models
weights = model.get_layer('embedding1').get_weights()
pd.DataFrame(weights[0]).head(11)
