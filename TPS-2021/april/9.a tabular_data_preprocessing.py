#%% 

# ANCHOR loading imports
import numpy as np
import pandas as pd
import time, re
from matplotlib import pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import lightgbm as lgb

import xgboost as xgb

from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.utils import class_weight

import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import callbacks

import warnings
warnings.filterwarnings("ignore")

now = datetime.now()
timestamp = now.strftime("%b-%d-%Y-at-%H-%M")

# NOTE It's safe to use index_col here because non-overlapping index occurs between train and test dataframes.
train_df = pd.read_csv('/kaggle/input/tabular-playground-series-apr-2021/train.csv')
test_df = pd.read_csv('/kaggle/input/tabular-playground-series-apr-2021/test.csv')
# NOTE submission's index is consistent the indices of test_df
test_df['Survived'] = pd.read_csv("/kaggle/input/tps-apr-2021-label/voting_submission_from_3_best.csv")['Survived']
submission = pd.read_csv('/kaggle/input/tabular-playground-series-apr-2021/sample_submission.csv')

# SECTION HIRO'S WORK
# HIRO'S "TPS Apr 2021 voting/pseudo labeling" framework
# https://www.kaggle.com/hiro5299834/tps-apr-2021-voting-pseudo-labeling

TARGET = 'Survived'
N_SPLITS = 10
SEED = 2021
N_ESTIMATORS = 1000
EARLY_STOPPING_ROUNDS = 100
VERBOSE = 100

all_df = pd.concat([train_df, test_df]).reset_index(drop=True)

all_df['Pclass'] = all_df['Pclass'].astype(str) # what I introdcued.

# Age fillna with mean age for each class
all_df['Age'] = all_df['Age'].fillna(all_df['Age'].mean())

# Cabin, fillna with 'X' and take first letter
all_df['Cabin'] = all_df['Cabin'].fillna('X').map(lambda x: x[0].strip())

# Ticket, fillna with 'X', split string and take first split 
all_df['Ticket'] = all_df['Ticket'].fillna('X').map(lambda x: str(x).split()[0] if len(str(x).split()) > 1 else 'X')

# Fare, fillna with mean value
fare_map = all_df[['Fare', 'Pclass']].dropna().groupby('Pclass').median().to_dict()
all_df['Fare'] = all_df['Fare'].fillna(all_df['Pclass'].map(fare_map['Fare']))

# Embarked, fillna with 'X' value
all_df['Embarked'] = all_df['Embarked'].fillna('X')

# Name, take only surnames
all_df['Name'] = all_df['Name'].map(lambda x: x.split(',')[0])

label_cols = ['Name', 'Ticket']
onehot_cols = ['Pclass', 'Sex', 'Cabin', 'Embarked']
numerical_cols = ['Age', 'SibSp', 'Parch', 'Fare', 'Survived']

def label_encoder(c):
    lc = LabelEncoder()
    return lc.fit_transform(c)

onehot_encoded_df = pd.get_dummies(all_df[onehot_cols])
label_encoded_df = all_df[label_cols].apply(label_encoder)
numerical_df = all_df[numerical_cols]

all_df = pd.concat([numerical_df, label_encoded_df, onehot_encoded_df], axis=1)

params = {
    'metric': 'binary_logloss',
    'n_estimators': N_ESTIMATORS,
    'objective': 'binary',
    'random_state': SEED,
    'learning_rate': 0.01,
    'min_child_samples': 150,
    'reg_alpha': 3e-5,
    'reg_lambda': 9e-2,
    'num_leaves': 20,
    'max_depth': 16,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'subsample_freq': 2,
    'max_bin': 240,
}

lgb_oof = np.zeros(train_df.shape[0])
lgb_preds = np.zeros(test_df.shape[0])
feature_importances = pd.DataFrame()

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

for fold, (train_idx, valid_idx) in enumerate(skf.split(all_df, all_df[TARGET])):

    print(f"===== FOLD {fold} =====")

    oof_idx = np.array([idx for idx in valid_idx if idx < train_df.shape[0]])
    preds_idx = np.array([idx for idx in valid_idx if idx >= train_df.shape[0]])

    X_train, y_train = all_df.iloc[train_idx].drop(TARGET, axis=1), all_df.iloc[train_idx][TARGET]
    X_valid, y_valid = all_df.iloc[oof_idx].drop(TARGET, axis=1), all_df.iloc[oof_idx][TARGET]
    X_test = all_df.iloc[preds_idx].drop(TARGET, axis=1)

    # ANCHOR Manual LightGBM LR scheduler
    pre_model = lgb.LGBMRegressor(**params)
    pre_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train),(X_valid, y_valid)],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose=VERBOSE
    )

    params2 = params.copy()
    params2['learning_rate'] = params['learning_rate'] * 0.1
    model = lgb.LGBMRegressor(**params2)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train),(X_valid, y_valid)],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose=VERBOSE,
        init_model=pre_model 
    )
    
    # ANCHOR Record feature importance
    fi_tmp = pd.DataFrame()
    fi_tmp["feature"] = model.feature_name_
    fi_tmp["importance"] = model.feature_importances_
    fi_tmp["fold"] = fold
    fi_tmp["seed"] = SEED
    feature_importances = feature_importances.append(fi_tmp)
    
    # ANCHOR Oof and test prediction
    lgb_oof[oof_idx] = model.predict(X_valid)
    lgb_preds[preds_idx - train_df.shape[0]] = model.predict(X_test)
    
    acc_score = accuracy_score(y_valid, np.where(lgb_oof[oof_idx]>0.5, 1, 0))
    print(f"===== ACCURACY SCORE {acc_score:.6f} =====\n")
    
order = list(feature_importances.groupby("feature").mean().sort_values("importance", ascending=False).index)
plt.figure(figsize=(10, 10))
sns.barplot(x="importance", y="feature", data=feature_importances, order=order)
plt.title("{} importance".format("LGBMRegressor"))
plt.tight_layout()




# !SECTION XGboost imputation

#%%
# SECTION EDA about the Fare feature.

def plot_target_distribution():
    ax = train_label.plot.hist(bins=100, figsize=(12,5), alpha=0.6)
    ax.grid()
    _ = ax.set(title='Train label distribution', xlabel='Target values')
plot_target_distribution()

# !SECTION EDA about the Fare feature.

# %%
# SECTION sklearn processing pipeline
# SOURCE https://scikit-learn.org/stable/auto_examples/ensemble/plot_stack_predictors.html#sphx-glr-auto-examples-ensemble-plot-stack-predictors-py
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer

categories = [all[col].unique() for col in all[obj_cols]]
cat_proc_nlin = make_pipeline( # categorical preprocessor for nonlinera estimator
    # SimpleImputer(missing_values=None, strategy='constant', fill_value='missing'),
    OrdinalEncoder(categories=categories) # NOTE different from its counterpart for linear classifiers.
)
cat_proc_lin = make_pipeline(
    # SimpleImputer(missing_values=None, strategy='constant', fill_value='missing'),
    OneHotEncoder(categories=categories)
)
class dummyTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X
# TODO "10 fold Simple DNN with Rank Gauss" link: https://www.kaggle.com/tottenham/10-fold-simple-dnn-with-rank-gauss
num_proc_nlin = make_pipeline(dummyTransformer())
num_proc_lin = make_pipeline(
    # SimpleImputer(strategy='mean'),
    StandardScaler()
)
int_proc_all = OneHotEncoder()
# transformation to use for non-linear estimators
processor_nlin = make_column_transformer(
    (cat_proc_nlin, obj_cols),
    (int_proc_all, int_cols),
    (num_proc_nlin, num_cols),
    remainder='passthrough')
# transformation to use for linear estimators
processor_lin = make_column_transformer(
    (cat_proc_lin, obj_cols),
    (int_proc_all, int_cols),
    (num_proc_lin, num_cols),
    remainder='passthrough')

# !SECTION preprocessing pipeline

# SECTION RANK GAUSS
# FARE https://i.postimg.cc/Cx86ty7L/2021-04-09-at-22-13-23.png
import matplotlib.pyplot as plt
import seaborn as sns
data.isnull().sum().sort_values(ascending=False)

# SOURCE https://www.kaggle.com/tottenham/10-fold-simple-dnn-with-rank-gauss
from scipy.special import erfinv
def rank_gauss(x):
    N = x.shape[0]
    temp = x.argsort()
    rank_x = temp.argsort() / N
    rank_x -= rank_x.mean()
    rank_x *= 2
    efi_x = erfinv(rank_x)
    efi_x -= efi_x.mean()
    return efi_x
def show_gauss_rank_transformation_effect(col):
    f, ax = plt.subplots(nrows=3, ncols=1, figsize=(4,8))
    sns.distplot(data[col], ax=ax[0])
    # NOTE We found there are at least four peaks:
    sns.distplot(data[col].map(lambda x: np.log(x)), ax=ax[1])
    # sns.distplot(data[['Fare']].apply(rank_gauss), ax=ax[2]) # seems doesn't work.
    sns.distplot(rank_gauss(data.loc[data[col].notnull(), col].values), ax=ax[2])

# show_gauss_rank_transformation_effect('Fare') # four peaks
# show_gauss_rank_transformation_effect('Age') # three peaks

# !SECTION RANK GAUSS

# SECTION DATA PREPROCESSING

train_label = train_df['Survived']
train_id = train_df['PassengerId']
del train_df['Survived'], train_df['PassengerId']
test_id = test_df['PassengerId']
del test_df['PassengerId']

data = train_df.append(test_df)
data.reset_index(inplace=True)
train_rows = train_df.shape[0]

# missing Cabin mark
data['misCount'] = data[['Age', 'Ticket', 'Fare', 'Cabin']].isnull().sum(axis=1)

data = data.assign(misCabin=np.where(data['Cabin'].isnull(), 1, 0)) # add a new column
data['Cabin'] = data['Cabin'].fillna('X-1')
re_letter_number = re.compile("([a-zA-Z]+)(-?[0-9]+)")

# Cabin letter
def func_cab1(x):
    if re_letter_number.match(x):
        return re_letter_number.match(x).groups()[0]
data['Cab1'] = data.Cabin.map(func_cab1)

map_Cab1 = dict()
for k, v in data.value_counts(['Cab1']).items(): # NOTE This command returns a tuple key, so just take the first item of tuple.
    map_Cab1[k[0]] = str(v)
data['Cab1Count'] = data.Cab1.map(map_Cab1) 

# Cabin number embedding
def func_cab2(x):
    if re_letter_number.match(x):
        return re_letter_number.match(x).groups()[1]
data['Cab2'] = data.Cabin.map(func_cab2)

# Recover NaN values to be imputed by GBM
data.Cab1.replace('X', np.nan, inplace=True)
data.Cab2.replace('-1', np.nan, inplace=True)

map_Cab2 = dict()
for k, v in data.value_counts(['Cab2']).items(): # NOTE This command returns a tuple key, so just take the first item of tuple.
    map_Cab2[k[0]] = str(v)
data['Cab2Count'] = data.Cab2.map(map_Cab2) 

tmp = data['Cab2'].copy().fillna('0').str.rjust(5, '0')
for i in range(5):
    data['Cab2_'+str(i)] = tmp.str[:i+1].astype(int).astype(str)
data[[col for col in data.columns if 'Cab2_' in col]].nunique() # NOTE WE know that Tick2_6 and Tick_7 have the same unique values, so drop one of them.
# data[[col for col in data.columns if 'Cab2_' in col]]

# NOTE Remove 'Cabin', 'Cab2' and keep 'Cab1 (c)', 'Cab2 (c)', Cab2Count (c)'
# SANITY CHECK
# data['Cabin'].loc[(data['Cab2']==-1) & (data['Cab1']!='X')] # clean!
# data['Cabin'].loc[(data['Cab2']!=-1) & (data['Cab1']=='X')] # clean!
# data[[col for col in data.columns if 'Cab' in col]]

# NAME
data[['Last', 'First']] = data.Name.str.split(', ', expand=True)

map_Last = dict()
for k, v in data.value_counts(['Last']).items():
    map_Last[k[0]] = str(v)
data['LastCount'] = data.Last.map(map_Last)

map_First = dict()
for k, v in data.value_counts(['First']).items():
    map_First[k[0]] = str(v)
data['FirstCount'] = data.First.map(map_First)

# TICKET
# Clustering the Tick1 feature.
data = data.assign(misTicket=np.where(data['Ticket'].isnull(), 1, 0))
new = data.Ticket.str.split(expand=True)
data['Tick1'] = np.where(new[0].str.isnumeric(), np.nan, new[0]) # TECH str.isnumeric
map_Tick1 = dict()
for k, v in data.value_counts(['Tick1']).items():
    map_Tick1[k[0]] = str(v)
data['Tick1Count'] = data.Tick1.map(map_Tick1)

# Clustering the Tick2 feature.
data['Tick2'] = data.Ticket.str.extract('(\d+)') # TECH str.extract
data = data.assign(misTick2=np.where(data.Tick2.isnull(), 1, 0))

map_Tick2 = dict()
for k, v in data.value_counts(['Tick2']).items(): # NOTE This command returns a tuple key, so just take the first item of tuple.
    map_Tick2[k[0]] = str(v)
data['Tick2Count'] = data.Tick2.map(map_Tick2) 

tmp = data['Tick2'].copy().fillna('0').str.rjust(7, '0')
for i in range(8):
    data['Tick2_'+str(i)] = tmp.str[:i+1].astype(int).astype(str)
# data[[col for col in data.columns if 'Tick2_' in col]].nunique() # NOTE WE know that Tick2_6 and Tick_7 have the same unique values, so drop one of them.

del data['Tick2_7']

# Rank Gauss Transformation
data.loc[data['Age'].notnull(), 'Age_rg'] = rank_gauss(data.loc[data['Age'].notnull(), 'Age'].values)
data.loc[data['Fare'].notnull(), 'Fare_rg'] = rank_gauss(data.loc[data['Fare'].notnull(), 'Fare'].values)

map_sex = {'female': 1, 'male': 0}
data['Sex'] = data['Sex'].map(map_sex)
data['Sex'] = data['Sex'].astype(str)
data['SibSp'] = data['SibSp'].astype(str)
data['Parch'] = data['Parch'].astype(str)
data['Pclass'] = data['Pclass'].astype(str)

del data['index']

# original_cols = ['Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Pclass', 'Cabin', 'Ticket', 'Fare', 'Embarked']
# data.select_dtypes('O').nunique()
name_catcols = ['First', 'Last', 'FirstCount', 'LastCount']
misc_catcols = ['SibSp', 'Parch', 'Pclass', 'Sex', 'Embarked']
cabine_catcols = ['Cab1', 'Cab1Count', 'Cab2', 'Cab2Count', 'Cab2_0', 'Cab2_1', 'Cab2_2', 'Cab2_3', 'Cab2_4']
ticket_catcols = ['Tick1', 'Tick1Count', 'Tick2', 'Tick2Count', 'Tick2_0', 'Tick2_1', 'Tick2_2', 'Tick2_3', 'Tick2_4', 'Tick2_5', 'Tick2_6']
binary_cols = ['misCount', 'misCabin', 'misTicket', 'misTick2']
numeric_cols = ['Age_rg', 'Fare_rg']

# Sanity check
# set(data.select_dtypes('O').columns).difference(set(name_catcols + misc_catcols + cabine_catcols + ticket_catcols))
# set(data.select_dtypes('int').columns).difference(set(binary_cols))
# set(data.select_dtypes('float').columns).difference(set(numeric_cols))
# set(data.columns).difference(set(name_catcols + misc_catcols + cabine_catcols + ticket_catcols + binary_cols + numeric_cols))

# import joblib
# os.chdir('/kaggle/working')
# joblib.dump(data, "data.pkl")

# !SECTION DATA PREPROCESSING