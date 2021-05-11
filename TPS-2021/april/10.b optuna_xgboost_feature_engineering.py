#%%
# ANCHOR package
import os
import numpy as np
import pandas as pd
import time, re
import pickle, os
from matplotlib import pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import lightgbm as lgb

import xgboost as xgb

from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, RepeatedKFold, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, mean_squared_error, log_loss, balanced_accuracy_score
from sklearn.utils import class_weight
from sklearn.metrics import roc_auc_score

import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import callbacks

import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
import optuna

from catboost import CatBoostClassifier, Pool

import warnings
warnings.filterwarnings("ignore")

now = datetime.now()
timestamp = now.strftime("%b-%d-%Y-at-%H-%M")

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('Time taken : %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

RS = 124  # random state
# N_JOBS = 1  # number of parallel threads

# repeated K-folds
N_SPLITS = 5
N_REPEATS = 1

# WARN
# Optuna
N_TRIALS = 25 # 100
MULTIVARIATE = True
TIMEOUT = 2*60*60

# XGBoost
EARLY_STOPPING_ROUNDS = 50

# ANCHOR loading data
# ------------------------------- loading data ------------------------------- #
# TECH time elapse template
# start_time = datetime.now()
# end_time = datetime.now()
# difference = end_time - start_time
# m, s = divmod(difference.total_seconds(), 60)
# print(f'H:M:S is {m//60}:{m%60}:{s}')

train_df = pd.read_csv('/kaggle/input/tabular-playground-series-apr-2021/train.csv')
test_df = pd.read_csv('/kaggle/input/tabular-playground-series-apr-2021/test.csv')

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

train_label = train_df['Survived']
train_id = train_df['PassengerId']
del train_df['Survived'], train_df['PassengerId']
test_id = test_df['PassengerId']
del test_df['PassengerId']

data = train_df.append(test_df)
data.reset_index(inplace=True) # Column Index is added.
train_rows = train_df.shape[0]

#%%
# SECTION DATA PREPROCESSING
# ----------------------- Features grow from 10 to 44. ----------------------- #

# Mark Missing Values by Feature.
data['misCount'] = data[['Age', 'Ticket', 'Fare', 'Cabin']].isnull().sum(axis=1) # integer
data['misAge'] = data['Age'].isnull() # bool
data['misFare'] = data['Fare'].isnull() # bool 
data['misEmbarked'] = data['Embarked'].isnull() # bool
data['misCabin'] = data['Cabin'].isnull() # bool
data['misTicket'] = data['Ticket'].isnull() # bool
# Similarly, but into integer dtype: data = data.assign(misCabin=np.where(data['Cabin'].isnull(), 1, 0)) # add a new column

# Cabin Zone
data['Cabin'] = data['Cabin'].fillna('X-1')
re_letter_number = re.compile("([a-zA-Z]+)(-?[0-9]+)")

# Cabin letter
def func_cab1(x):
    if re_letter_number.match(x):
        return re_letter_number.match(x).groups()[0]
data['Cab1'] = data.Cabin.map(func_cab1) # string

# Cabin number embedding
def func_cab2(x):
    if re_letter_number.match(x):
        return re_letter_number.match(x).groups()[1]
data['Cab2'] = data.Cabin.map(func_cab2) # string

map_Cab1 = dict()
for k, v in data.value_counts(['Cab1']).items(): # NOTE This command returns a tuple key, so just take the first item of tuple.
    map_Cab1[k[0]] = v
data['Cab1Count'] = data.Cab1.map(map_Cab1) # integer

map_Cab2 = dict()
for k, v in data.value_counts(['Cab2']).items(): # NOTE This command returns a tuple key, so just take the first item of tuple.
    map_Cab2[k[0]] = v
data['Cab2Count'] = data.Cab2.map(map_Cab2) # integer

# Recover NaN values so as to be imputed later by GBM
data.Cab1Count.replace(138697, -1, inplace=True)
data.Cab2Count.replace(138697, -1, inplace=True)
data.Cab1.replace('X', np.nan, inplace=True)
data.Cab2.replace('-1', np.nan, inplace=True)

# Inspecting cabin number clustering
tmp = data['Cab2'].copy().fillna('0').str.rjust(5, '0')
for i in range(5):
    data['Cab2_'+str(i)] = tmp.map(lambda x: np.nan if x=='00000' else x[:i+1]) # string

# REVIEW data[[col for col in data.columns if 'Cab2_' in col]].nunique() # NOTE WE know that Tick2_6 and Tick_7 have the same unique values, so drop one of them.

# NOTE Remove 'Cabin', 'Cab2' and keep 'Cab1 (c)', 'Cab2 (c)', Cab2Count (c)'
# SANITY CHECK
# data['Cabin'].loc[(data['Cab2']==-1) & (data['Cab1']!='X')] # clean!
# data['Cabin'].loc[(data['Cab2']!=-1) & (data['Cab1']=='X')] # clean!
# data[[col for col in data.columns if 'Cab' in col]]

# NAME
data[['Last', 'First']] = data.Name.str.split(', ', expand=True)

map_Last = dict()
for k, v in data.value_counts(['Last']).items():
    map_Last[k[0]] = v
data['LastCount'] = data.Last.map(map_Last) # integer

map_First = dict()
for k, v in data.value_counts(['First']).items():
    map_First[k[0]] = v
data['FirstCount'] = data.First.map(map_First) # integer

# TICKET
# Clustering the Tick1 feature.
new = data.Ticket.str.split(expand=True)
data['Tick1'] = np.where(new[0].str.isnumeric(), np.nan, new[0]) # TECH str.isnumeric
data['misTick1'] = data['Tick1'].isnull() # bool
map_Tick1 = dict()
for k, v in data.value_counts(['Tick1']).items():
    map_Tick1[k[0]] = str(v)
data['Tick1Count'] = data.Tick1.map(map_Tick1) # integer

# Clustering the Tick2 feature.
data['Tick2'] = data.Ticket.str.extract('(\d+)') # TECH str.extract
data['misTick2'] = data['Tick2'].isnull() # bool
# data = data.assign(misTick2=np.where(data.Tick2.isnull(), 1, 0))
map_Tick2 = dict()
for k, v in data.value_counts(['Tick2']).items(): # NOTE This command returns a tuple key, so just take the first item of tuple.
    map_Tick2[k[0]] = str(v)
data['Tick2Count'] = data.Tick2.map(map_Tick2) 

tmp = data['Tick2'].copy().fillna('0').str.rjust(7, '0')
for i in range(8):
    data['Tick2_'+str(i)] = tmp.str[:i+1] # string
# data[[col for col in data.columns if 'Tick2_' in col]].nunique() # NOTE WE know that Tick2_6 and Tick_7 have the same unique values, so drop one of them.

del data['Tick2_7']

# Rank Gauss Transformation
data.loc[data['Age'].notnull(), 'Age_rg'] = rank_gauss(data.loc[data['Age'].notnull(), 'Age'].values) # float
data.loc[data['Fare'].notnull(), 'Fare_rg'] = rank_gauss(data.loc[data['Fare'].notnull(), 'Fare'].values) # float

map_sex = {'female': 1, 'male': 0}
data['Sex'] = data['Sex'].map(map_sex) # integer

del data['index']

os.chdir('/kaggle/working')
pickle.dump(data, open('data.pkl', 'wb'))

# Sanity check
# set(data.select_dtypes('O').columns).difference(set(name_catcols + misc_catcols + cabine_catcols + ticket_catcols))
# set(data.select_dtypes('int').columns).difference(set(binary_cols))
# set(data.select_dtypes('float').columns).difference(set(numeric_cols))
# set(data.columns).difference(set(name_catcols + misc_catcols + cabine_catcols + ticket_catcols + binary_cols + numeric_cols))

# # EXPERIMENT Split complete data into datasets by isna() 
# # %%
# import string
# abcde = string.ascii_letters[:5]
# # synthetic data
# d = pd.DataFrame({'letter': list(abcde)})
# d.iloc[2] = np.nan
# d.iloc[4] = np.nan
# # get the index labels that correspond to np.nan.
# idlabel_test = d.loc[d.letter.isna()].index.tolist() # TECH The key one-liner
# idlabel_train = d.loc[~d.letter.isna()].index.tolist()
# # split data by isna()
# test = d.loc[idlabel_test]
# train = d.loc[idlabel_train]
# # update the test data with 'z' (like we make prediction on test data)
# test.loc[:] = 'z'
# # now, test data no longer have missing values
# test
# # update the original dataset
# d.iloc[idlabel_test] = test # TECH Update the original dataset 
# d
# # !EXPERIMENT

data = pickle.load(open('data.pkl', 'rb'))
colsToRemove = ['Name', 'Age', 'Fare', 'Ticket', 'Cabin']
colsToKeep = list(set(data.columns.tolist()).difference(set(colsToRemove)))
data = data[colsToKeep].copy()

cols_flt = data[colsToKeep].select_dtypes(include='float').columns.tolist()
cols_int = data[colsToKeep].select_dtypes(include='int').columns.tolist() # ordinal encoding
cols_obj = data[colsToKeep].select_dtypes(include='O').columns.tolist()
# REVIEW check the columns with missing values
data.columns[data.isnull().any()].tolist()
data

# SOURCE https://datascience.stackexchange.com/questions/39317/difference-between-ordinalencoder-and-labelencoder

for col in cols_obj:
    valid_idx = data.loc[data[col].notnull(), col].index.tolist()
    le = LabelEncoder()
    data.loc[valid_idx, col+'_le'] = le.fit_transform(data.loc[valid_idx, col].to_numpy().reshape(-1,1))
    data.pop(col)

os.chdir('/kaggle/working')
pickle.dump(data, open('data2.pkl', 'wb'))

# !SECTION DATA PREPROCESSING




# ------------------------------- Fare and Age ------------------------------- #




#%%
# SECTION FLOAT VARIABLES - Fare


data = pickle.load(open('data2.pkl', 'rb'))


# SOURCE XGBoost doc https://xgboost.readthedocs.io/en/latest/parameter.html
# NOTE Optuna + XGBoost SKF regressor https://www.kaggle.com/pratikkgandhi/simple-xgboost-starter-optuna
# X_y_train = xgb.DMatrix(data=data.drop('Fare_rg', axis=1), label= data['Fare_rg'])


fit_idx = data.loc[data['Fare_rg'].notnull(),:].index.tolist()
pre_idx = data.loc[~data['Fare_rg'].notnull(),:].index.tolist()

fit_X = data.loc[fit_idx, :].drop('Fare_rg', axis=1)
fit_y = data.loc[fit_idx, :]['Fare_rg']
pre_X = data.loc[pre_idx, :].drop('Fare_rg', axis=1)

def float_col_objective(
    trial,
    X,
    y,
    random_state = 22,
    n_splits = 3,
    n_repeats = 1,
    n_jobs = -1, # SOURCE https://discuss.xgboost.ai/t/n-jobs-1-no-longer-uses-all-cores/1955/6
    early_stopping_rounds = 50
):
    # ANCHOR XGBoost parameters
    params = {
        "objective": "reg:squarederror", # NOTE Learning task parameters - https://xgboost.readthedocs.io/en/latest/parameter.html#learning-task-parameters
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.005, 0.05),
        "colsample_bytree": trial.suggest_loguniform("colsample_bytree", 0.2, 0.8),
        "subsample": trial.suggest_loguniform("subsample", 0.4, 0.8),
        "alpha": trial.suggest_loguniform("alpha", 0.01, 10.0),
        "lambda": trial.suggest_loguniform("lambda", 1e-8, 10.0),
        "gamma": trial.suggest_loguniform("lambda", 1e-8, 10.0), # Minimum loss reduction required to make a further partition on a leaf node of the tree.
        "min_child_weight": trial.suggest_loguniform("min_child_weight", 10, 1000), # A smaller value is chosen because it is a highly imbalanced class problem and leaf nodes can have smaller size groups.
        # "scale_pos_weight": 1, # because of high class imbalance
        "verbosity": 0,  # 0 (silent) - 3 (debug)
        "seed": random_state,
        "tree_method": 'gpu_hist' # NOTE 
    }

    xgb_model = XGBRegressor(**params)
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation_0-rmse") # NOTE observation_keys - https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.XGBoostPruningCallback.html
    # pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation_0-auc")
# CONSTRUCTION
    # NOTE oof - https://www.kaggle.com/vinhnguyen/accelerating-xgboost-with-gpu
    rkf = RepeatedKFold(
        n_splits=n_splits, 
        n_repeats=n_repeats, 
        random_state=random_state
    )
    X_values = X.values
    y_values = y.values
    y_oof = np.zeros_like(y_values)
    res = 0
    for fold, (train_index, valid_index) in enumerate(rkf.split(X_values, y_values)):
        X_A, X_B = X_values[train_index, :], X_values[valid_index, :]
        y_A, y_B = y_values[train_index], y_values[valid_index]
        xgb_model.fit(
            X_A,
            y_A,
            eval_set=[(X_B, y_B)],
            eval_metric="rmse", # NOTE Learning task parameters - https://xgboost.readthedocs.io/en/latest/parameter.html#learning-task-parameters 此用於檢視callback事件是否出現，不是直接用於找optuna hyperparameter 
            early_stopping_rounds=early_stopping_rounds,
            callbacks=[pruning_callback],
            verbose=0
        )
        y_pred = xgb_model.predict(X_B)
        y_oof[valid_index] += y_pred
        res += np.sqrt(mean_squared_error(y_pred, y_B)) / (n_splits * n_repeats) # I added it.

    y_oof /= n_repeats # Original
    trial.set_user_attr(key="best_booster", value=xgb_model) # NOTE update the best model in the optuna's table.
    # return np.sqrt(mean_squared_error(y_values, y_pred)) # Originally, the author uses y_train. I think it's incorrect.
    return res # I changed the last line to this one.

# ANCHOR create study
# SOURCE retrieve the best model - https://stackoverflow.com/questions/62144904/python-how-to-retrive-the-best-model-from-optuna-lightgbm-study#answer-63365355
def best_model_parameters(study, trial):
    if study.best_trial.number == trial.number:
        # Set the best booster as a trial attribute; accessible via study.trials_dataframe.
        study.set_user_attr(key="best_booster", value=trial.user_attrs["best_booster"])
        # SOURCE retrieve the best number of estimators https://github.com/optuna/optuna/issues/1169
        # trial.set_user_attr('n_estimators', len(xgb_cv_results))

# NOTE GMM sampler - https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.TPESampler.html
# NOTE pruner - https://optuna.readthedocs.io/en/latest/tutorial/10_key_features/005_visualization.html

study = optuna.create_study(
    direction = "minimize", 
    sampler = optuna.samplers.TPESampler(seed=RS, multivariate=MULTIVARIATE),
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
)

study.optimize(
    lambda trial: float_col_objective(
        trial,
        fit_X,
        fit_y,
        random_state = RS,
        n_splits = N_SPLITS,
        n_repeats = N_REPEATS,
        n_jobs=-1, # 此n_jobs https://github.com/optuna/optuna/issues/2270
        early_stopping_rounds = EARLY_STOPPING_ROUNDS,
    ),
    callbacks = [best_model_parameters],
    n_trials = N_TRIALS,
    timeout = TIMEOUT,
    n_jobs = 1 # 應該是不同於此n_jobs吧？ https://stackoverflow.com/questions/48269248/limiting-the-number-of-threads-used-by-xgboost
)

# display params
hp = study.best_params
for key, value in hp.items():
    print(f"{key:>20s} : {value}")
print(f"{'best objective value':>20s} : {study.best_value}")    

# NOTE retrieve the optimal tree limit observed in the best round
# SOURCE - end-to-end example on tabular data https://aetperf.github.io/2021/02/16/Optuna-+-XGBoost-on-a-tabular-dataset.html#submit-and-evaluate-the-prediction
# SOURCE - bst.best_ntree_limit https://xgboost.readthedocs.io/en/latest/python/python_intro.html?highlight=early%20stopping#prediction
# SOURCE - (continued, same issue raised in SO) https://stackoverflow.com/questions/53483648/is-the-xgboost-documentation-wrong-early-stopping-rounds-and-best-and-last-it

# SOURCE retrieve the best model - https://stackoverflow.com/questions/62144904/python-how-to-retrive-the-best-model-from-optuna-lightgbm-study#answer-63365355
# study.user_attrs['best_booster'].best_iteration
# study.user_attrs['best_booster'].get_num_boosting_rounds()

def predict_after_study(
    model,
    X,
    y,
    X_imp,
    random_state = 22,
    n_splits = 3,
    n_repeats = 1,
    n_jobs = -1,
    early_stopping_rounds = 50
):
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    X_values = X.values
    y_values = y.values
    X_test = X_imp.values
    y_pred = np.zeros((X_test.shape[0], 1))

    for i, (train_index, test_index) in enumerate(rkf.split(X_values)):
        X_A, X_B = X_values[train_index, :], X_values[test_index, :]
        y_A, y_B = y_values[train_index], y_values[test_index]
        model.fit(
            X_A,
            y_A,
            eval_set=[(X_B, y_B)],
            eval_metric="rmse",
            early_stopping_rounds=early_stopping_rounds, # WARN 
            verbose=0
        )
        # SOURCE model.best_ntree_limit. https://stackoverflow.com/questions/51955256/xgboost-best-iteration
        y_pred += model.predict(X_test, ntree_limit=model.best_ntree_limit).reshape(-1,1) # WARN I added the 2nd argument.
    y_pred /= n_repeats
    return y_pred

y_pred = predict_after_study(
    study.user_attrs['best_booster'],
    fit_X,
    fit_y,
    pre_X,
    random_state=RS,
    n_splits=N_SPLITS,
    n_repeats=N_REPEATS,
    n_jobs=-1,
    early_stopping_rounds=EARLY_STOPPING_ROUNDS)


old_data = pd.DataFrame()
old_data['Fare_rg'+'_old'] = data['Fare_rg'].copy()
data.loc[pre_idx, 'Fare_rg'] = y_pred
old_data['Fare_rg_new'] = data['Fare_rg'].copy()


# ------------------------------------ Age ----------------------------------- #

fit_idx = data.loc[data['Age_rg'].notnull(),:].index.tolist()
pre_idx = data.loc[~data['Age_rg'].notnull(),:].index.tolist()

fit_X = data.loc[fit_idx, :].drop('Age_rg', axis=1)
fit_y = data.loc[fit_idx, :]['Age_rg']
pre_X = data.loc[pre_idx, :].drop('Age_rg', axis=1)

study = optuna.create_study(
    direction = "minimize", 
    sampler = optuna.samplers.TPESampler(seed=RS, multivariate=MULTIVARIATE),
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
)

study.optimize(
    lambda trial: float_col_objective(
        trial,
        fit_X,
        fit_y,
        random_state = RS,
        n_splits = N_SPLITS,
        n_repeats = N_REPEATS,
        n_jobs=-1, 
        early_stopping_rounds = EARLY_STOPPING_ROUNDS,
    ),
    callbacks = [best_model_parameters],
    n_trials = N_TRIALS,
    timeout = TIMEOUT,
    n_jobs = 1 
)

y_pred = predict_after_study(
    study.user_attrs['best_booster'],
    fit_X,
    fit_y,
    pre_X,
    random_state=RS,
    n_splits=N_SPLITS,
    n_repeats=N_REPEATS,
    n_jobs=-1,
    early_stopping_rounds=EARLY_STOPPING_ROUNDS)

old_data['Age_rg'+'_old'] = data['Age_rg'].copy()
data.loc[pre_idx, 'Age_rg'] = y_pred
old_data['Age_rg_new'] = data['Age_rg'].copy()

os.chdir('/kaggle/working')
pickle.dump(data, open('data_fare_age.pkl', 'wb'))
pickle.dump(old_data, open('old_data_fare_age.pkl', 'wb'))

# display params
hp = study.best_params
for key, value in hp.items():
    print(f"{key:>20s} : {value}")
print(f"{'best objective value':>20s} : {study.best_value}")  

# !SECTION FLOAT VARIABLES - Fare and Age

# import seaborn as sns
# def show_gauss_rank_transformation_effect(col):
#     f, ax = plt.subplots(nrows=2, ncols=1, figsize=(4,8))
#     sns.distplot(old_data[col+'_old'], ax=ax[0])
#     sns.distplot(old_data[col+'_new'], ax=ax[1])
# show_gauss_rank_transformation_effect('Age_rg')
# # Screenshot https://i.postimg.cc/MZNNsMNK/2021-04-14-at-23-24-47.png
# 似乎多了很多小孩 （because the imputed dist is skewed to the left）

#%%

# ANCHOR EDA
plt.figure(figsize=(10,8))
sns.heatmap(fit_X.corr())
plt.show()
# Screenshot https://i.postimg.cc/BQbkNjSC/2021-04-16-at-11-07-00.png

plt.pie(data.groupby('Embarked_le').count()['LastCount'], 
    autopct='%1.1f%%', 
    labels=data.groupby('Embarked_le').count()['LastCount'].index.tolist())
plt.show()
# Screenshot https://i.postimg.cc/RVHSYX9P/2021-04-16-at-11-23-09.png

sns.distplot(data['Embarked_le'])
plt.show()
# Screenshot https://i.postimg.cc/Fzcj105h/2021-04-16-at-11-26-22.png


#%%
# SECTION Embarked
# ANCHOR load data
os.chdir('/kaggle/input/tps-apr-2021-label/')
data_0 = pickle.load(open('data_fare_age.pkl', 'rb'))
old_data_0 = pickle.load(open('old_data_fare_age.pkl', 'rb'))

data = data_0.copy()
old_data = old_data_0.copy()

cols_bool = data.select_dtypes(include='bool').columns.tolist() # 7
cols_float = [col for col in data.select_dtypes(include='float') if col not in ['Fare_rg', 'Age_rg', 'Embarked_le']] # 11
cols_int = data.select_dtypes(include='int64').columns.tolist() # 18
# for i in [cols_bool, cols_float, cols_int]:
#     print(len(i))
# set(fit_X.columns.tolist()).difference(set(cols_bool + cols_float + cols_int))
for col in cols_bool:
    data[col] = data[col].astype(str)
for col in cols_float:
    data[col] = data[col].fillna(-1) # NOTE Be sure to fill na before converting into integer type.
    data[col] = data[col].astype(int)
data[cols_float] = data[cols_float].fillna(-1)
for col in data.drop('Embarked_le', axis=1).columns[data.drop('Embarked_le', axis=1).isnull().sum()>0].tolist():
    data[col] = data[col].fillna(-1)

col = 'Embarked_le'
# data[col].isnull().sum()
fit_idx = data.loc[data[col].notnull(),:].index.tolist()
pre_idx = data.loc[~data[col].notnull(),:].index.tolist()
fit_X = data.loc[fit_idx, :].drop(col, axis=1)
fit_y = data.loc[fit_idx, :][col]
pre_X = data.loc[pre_idx, :].drop(col, axis=1)
categorical_features = [col for col in fit_X.columns if col not in ['Fare_rg', 'Age_rg', 'Embarked_le']]

#%%

# #manually balanced the training data
# map_embark_freq = fit_y.value_counts(normalize=True).sort_index()
# map_embark_freq
# class_weights_embark = map_embark_freq.map(lambda x: 1.-x)
# cbmodel = CatBoostClassifier(iterations=1000, task_type="GPU", class_weights=class_weights_embark)

def objective(trial):
    params = {'iterations': 1000,
              'depth': trial.suggest_int("depth", 4, 12),
              'l2_leaf_reg': trial.suggest_float("l2_leaf_reg", 0.0001, 25, log=True),
              'eval_metric': 'MultiClass', # SOURCE https://catboost.ai/docs/concepts/loss-functions-multiclassification.html
              'loss_function': 'MultiClass',
              'auto_class_weights': trial.suggest_categorical( # SOURCE auto_class_weights - https://catboost.ai/docs/concepts/python-reference_parameters-list.html
                  'auto_class_weights', [None,'Balanced','SqrtBalanced']
                  ),
            #   'grow_policy': trial.suggest_categorical(
            #       'grow_policy',['SymmetricTree','Depthwise','Lossguide']
            #       ),              
            #   "boosting_type": trial.suggest_categorical("boosting_type", ["Plain"]), # NOTE On GPU loss MultiClass can't be used with ordered boosting.
            #   "bootstrap_type": trial.suggest_categorical(
            #       "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
            #       ),        
              'early_stopping_rounds': 100,
              'use_best_model': True,
              'task_type': 'GPU', 
              'cat_features': categorical_features,
              'verbose': True
            #   'border_count': 254
             }    
    cbmodel = CatBoostClassifier(**params) 
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    roc_test = []
    y_oof = np.zeros((len(fit_y), fit_y.nunique()))

    for i, (train_index, test_index) in enumerate(kf.split(fit_X, fit_y)):
        # x_train_fold, x_test_fold = fit_X.loc[fit_X.index.intersection(train_index)], fit_X.loc[fit_X.index.intersection(test_index)]
        x_train_fold, x_test_fold = fit_X.iloc[train_index], fit_X.iloc[test_index]
        # y_train_fold, y_test_fold = fit_y.loc[fit_y.index.intersection(train_index)], fit_y.loc[fit_X.index.intersection(test_index)]   
        y_train_fold, y_test_fold = fit_y.iloc[train_index], fit_y.iloc[test_index]   
        # NOTE To avoid the trouble, we may convert the dataframe to numpy array beforehand.
        train_dataset = Pool(data=x_train_fold,
                             label=y_train_fold,
                             cat_features=categorical_features)
        eval_dataset = Pool(data=x_test_fold,
                            label=y_test_fold,
                            cat_features=categorical_features)     
        # if params["bootstrap_type"] == "Bayesian":
        #     params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
        # elif params["bootstrap_type"] == "Bernoulli":
        #     params["subsample"] = trial.suggest_float("subsample", 0.1, 1)                        
        # if params['grow_policy'] in ['Depthwise', 'Lossguide']:
        #     params['min_data_in_leaf'] = trial.suggest_int("min_data_in_leaf", 1, 5000, log=True)
        # if params['grow_policy'] in ['Lossguide']:
        #     params['max_leaves'] = trial.suggest_int("max_leaves", 1, 64)             
        cbmodel.fit(train_dataset, eval_set=eval_dataset, verbose=0)    
        y_pred = cbmodel.predict_proba(eval_dataset)
        roc_test.append(roc_auc_score(y_test_fold, y_pred, multi_class='ovo')) # SOURCE log loss https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html
    return np.mean(roc_test)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30, timeout=2*60*60)
print("Number of finished trials: {}".format(len(study.trials)))
print("Best trial:")
trial = study.best_trial
print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key:>20s} : {value}")

# hp = study.best_params
# for key, value in hp.items():
#     print(f"{key:>20s} : {value}")
# print(f"{'best objective value':>20s} : {study.best_value}")        

# SOURCE - Imbalance classification metrics - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html#sklearn.metrics.balanced_accuracy_score


#%%
# ANCHOR inference 
from sklearn.model_selection import RepeatedStratifiedKFold
params = {'iterations': 10000,
          'eval_metric': 'MultiClass', # SOURCE https://catboost.ai/docs/concepts/loss-functions-multiclassification.html
          'loss_function': 'MultiClass',
          'use_best_model': True,
          'task_type': 'GPU', 
          'cat_features': categorical_features,
          'verbose': True}
params['depth'] = 6
params['l2_leaf_reg'] = 0.0036880754325527318
params['auto_class_weights'] = 'Balanced' 
params['od_type'] = 'Iter'
params['od_wait'] = 100
params['verbose'] = 500

def run_rskf(train, target, test, clf, params):
    train_preds = np.zeros((train.shape[0], target.nunique())) # repeated two.
    test_preds = np.zeros((test.shape[0], target.nunique()))
    N_REPEATS = 1
    N_SPLITS = 5
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=1337)
    for fold, (train_index, val_index) in enumerate(rskf.split(train, target)):
        print("-> Fold {}".format(fold + 1))
        start_time = timer(None)
        x_train, x_valid = train.iloc[train_index], train.iloc[val_index]
        y_train, y_valid = target.iloc[train_index], target.iloc[val_index]
    
        model = clf(**params)
        model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)])

        train_oof_preds = model.predict_proba(x_valid)
        train_preds[val_index, :] += train_oof_preds / N_REPEATS
        test_preds += model.predict_proba(test) / N_REPEATS / N_SPLITS
        print("ROC AUC Score = {}".format(roc_auc_score(y_valid, train_oof_preds, multi_class='ovo')))
        # if fold in [4, 9]:
        #     print("=> Overall ROC AUC Score = {}".format(roc_auc_score(target, train_preds[:, fold//5])))
        timer(start_time)
    # return model, train_preds.mean(axis=1), test_preds
    return test_preds

test_preds = run_rskf(fit_X, fit_y, pre_X, CatBoostClassifier, params)

col = 'Embarked_le'
old_data_0[col + '_old'] = data[col]
data_0.loc[pre_idx, col] = np.where(test_preds.max(axis=1)>0.5, test_preds.argmax(axis=1), np.nan) # TODO change 0.5
data_0[col] = data_0[col].fillna(-1)
old_data_0[col + '_new'] = data_0[col]

os.chdir('/kaggle/input/tps-apr-2021-label')
pickle.dump(data_0, open('data_fare_age_embarked.pkl', 'wb'))
pickle.dump(old_data_0, open('old_data_fare_age_embarked.pkl', 'wb'))

# !SECTION Embarked

#%%
# SECTION cabin
os.chdir('/kaggle/input/tps-apr-2021-label')
data = pickle.load(open('data_fare_age_embarked.pkl', 'rb'))
old_data = pickle.load(open('old_data_fare_age_embarked.pkl', 'rb'))
cols_bool = data.select_dtypes(include='bool').columns.tolist()

for col in cols_bool:
    data[col] = data[col].astype(int) 
data['Embarked_le'] = data['Embarked_le'].astype(int)
for col in data.columns[data.isnull().sum() > 0].tolist():
    data[col] = data[col].fillna(-1)
del data['Cab2_4_le'] # duplicates of column 'Cab2_le'
del data['Tick2_6_le'] # duplicates of column 'Tick2_le'

#%%
# Cabin and Ticket: we process Cabin-related columns first.
cols_cab = [col for col in data.columns.tolist() if col.startswith('Cab') and not col.endswith('Count')]
list_cab_cols = data[cols_cab].apply(lambda x: x.nunique()).sort_values().index.tolist()
data_0 = data.copy()
old_data_0 = old_data.copy()
for col in list_cab_cols[:1]:
    data[col].replace(-1, np.nan)
    fit_idx = data.loc[data[col].notnull(),:].index.tolist()
    pre_idx = data.loc[~data[col].notnull(),:].index.tolist()
    fit_X = data.loc[fit_idx, :].drop(col, axis=1)
    fit_y = data.loc[fit_idx, :][col]
    pre_X = data.loc[pre_idx, :].drop(col, axis=1)    


# cols_tick = [col for col in data.columns.tolist() if col.startswith('Tick')]
# data[cols_tick].apply(lambda x: x.nunique()).sort_values()




# !SECTION cabin


#%%

"""
[I 2021-04-16 03:11:52,541] Trial 17 finished with value: 0.5011334304544032 and parameters: {'n_estimators': 580, 'max_depth': 10, 'learning_rate': 0.028588349806519055, 'colsample_bytree': 0.5889284940137873, 'subsample': 0.6718415830581725, 'alpha': 0.03730438945607468, 'lambda': 1.3191894628188646e-06, 'min_child_weight': 234.4618516221924}. Best is trial 17 with value: 0.5011334304544032.

        n_estimators : 580
           max_depth : 10
       learning_rate : 0.028588349806519055
    colsample_bytree : 0.5889284940137873
           subsample : 0.6718415830581725
               alpha : 0.03730438945607468
              lambda : 1.3191894628188646e-06
    min_child_weight : 234.4618516221924
best objective value : 0.5011334304544032

[I 2021-04-16 03:12:18,566] A new study created in memory with name: no-name-ab516e0d-0de5-4134-ab75-702d0cb3776c
[I 2021-04-16 03:12:50,174] Trial 0 finished with value: 0.6350430193029943 and parameters: {'n_estimators': 150, 'max_depth': 12, 'learning_rate': 0.018675986054482886, 'colsample_bytree': 0.37750149752265827, 'subsample': 0.522237444587806, 'alpha': 0.06636940416485738, 'lambda': 0.010272157854456487, 'min_child_weight': 61.794528315016734}. Best is trial 0 with value: 0.6350430193029943.
[I 2021-04-16 03:14:21,871] Trial 1 finished with value: 0.6337192969765987 and parameters: {'n_estimators': 317, 'max_depth': 13, 'learning_rate': 0.019863883897580023, 'colsample_bytree': 0.23716526273929117, 'subsample': 0.5240026227682785, 'alpha': 8.413274807593929, 'lambda': 0.00016158093114227923, 'min_child_weight': 19.06965408897461}. Best is trial 1 with value: 0.6337192969765987.
[I 2021-04-16 03:14:46,105] Trial 2 finished with value: 0.6374237654129081 and parameters: {'n_estimators': 90, 'max_depth': 12, 'learning_rate': 0.020633624064141605, 'colsample_bytree': 0.6990832131945579, 'subsample': 0.6158732273897668, 'alpha': 0.060306636222079046, 'lambda': 0.0009992153071795662, 'min_child_weight': 64.56140383241545}. Best is trial 1 with value: 0.6337192969765987.
[I 2021-04-16 03:15:24,604] Trial 3 finished with value: 0.6292277116499069 and parameters: {'n_estimators': 184, 'max_depth': 10, 'learning_rate': 0.016807040593762004, 'colsample_bytree': 0.7346913293441047, 'subsample': 0.7775194620933836, 'alpha': 8.980394416174395, 'lambda': 1.1984090595201508e-06, 'min_child_weight': 15.637182208463356}. Best is trial 3 with value: 0.6292277116499069.
[I 2021-04-16 03:15:49,383] Trial 4 finished with value: 0.636519234807417 and parameters: {'n_estimators': 283, 'max_depth': 11, 'learning_rate': 0.007877744754433577, 'colsample_bytree': 0.6316189232442854, 'subsample': 0.5084575276769833, 'alpha': 0.3483730757916216, 'lambda': 0.00892116947224734, 'min_child_weight': 736.629287325251}. Best is trial 3 with value: 0.6292277116499069.
[I 2021-04-16 03:15:50,200] Trial 5 pruned. Trial was pruned at iteration 10.
[I 2021-04-16 03:15:51,192] Trial 6 pruned. Trial was pruned at iteration 10.
[I 2021-04-16 03:15:52,535] Trial 7 pruned. Trial was pruned at iteration 10.
[I 2021-04-16 03:15:53,367] Trial 8 pruned. Trial was pruned at iteration 10.
[I 2021-04-16 03:16:55,704] Trial 9 finished with value: 0.6296789941629127 and parameters: {'n_estimators': 186, 'max_depth': 13, 'learning_rate': 0.025581350036249832, 'colsample_bytree': 0.770043647161806, 'subsample': 0.7876286925323414, 'alpha': 0.31622512584608864, 'lambda': 0.007965846930681735, 'min_child_weight': 34.477585735407466}. Best is trial 3 with value: 0.6292277116499069.
[I 2021-04-16 03:18:21,952] Trial 10 finished with value: 0.6310564984751381 and parameters: {'n_estimators': 182, 'max_depth': 13, 'learning_rate': 0.020590198277764645, 'colsample_bytree': 0.5024779588268609, 'subsample': 0.6852020827792323, 'alpha': 3.0530751047507123, 'lambda': 3.928109762002199e-08, 'min_child_weight': 10.943715118694673}. Best is trial 3 with value: 0.6292277116499069.
[I 2021-04-16 03:18:22,653] Trial 11 pruned. Trial was pruned at iteration 10.
[I 2021-04-16 03:19:02,195] Trial 12 finished with value: 0.6282393914801843 and parameters: {'n_estimators': 558, 'max_depth': 10, 'learning_rate': 0.024803129803901227, 'colsample_bytree': 0.6665430691615799, 'subsample': 0.7896611845413086, 'alpha': 0.35679156488896396, 'lambda': 0.04425775333453395, 'min_child_weight': 38.23459906990938}. Best is trial 12 with value: 0.6282393914801843.
[I 2021-04-16 03:19:04,368] Trial 13 pruned. Trial was pruned at iteration 163.
[I 2021-04-16 03:19:05,184] Trial 14 pruned. Trial was pruned at iteration 10.
[I 2021-04-16 03:19:06,228] Trial 15 pruned. Trial was pruned at iteration 10.
[I 2021-04-16 03:19:48,201] Trial 16 finished with value: 0.6291873884060097 and parameters: {'n_estimators': 948, 'max_depth': 13, 'learning_rate': 0.03318422656073774, 'colsample_bytree': 0.4827164439932763, 'subsample': 0.7972570020634451, 'alpha': 0.27657199575562885, 'lambda': 0.0006298556374089483, 'min_child_weight': 123.52335815505386}. Best is trial 12 with value: 0.6282393914801843.
[I 2021-04-16 03:20:37,031] Trial 17 finished with value: 0.6288650499944104 and parameters: {'n_estimators': 970, 'max_depth': 13, 'learning_rate': 0.025449060002386965, 'colsample_bytree': 0.5868863011873958, 'subsample': 0.6787507295259426, 'alpha': 0.13252068023645633, 'lambda': 0.0009205408839654984, 'min_child_weight': 97.89598815448149}. Best is trial 12 with value: 0.6282393914801843.

        n_estimators : 558
           max_depth : 10
       learning_rate : 0.024803129803901227
    colsample_bytree : 0.6665430691615799
           subsample : 0.7896611845413086
               alpha : 0.35679156488896396
              lambda : 0.04425775333453395
    min_child_weight : 38.23459906990938
best objective value : 0.6282393914801843
"""
# %%
