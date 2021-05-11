#%%
# !pip install ssh-Colab
# !pip install google-colab

import os, subprocess
from google.cloud import storage

COLAB = True

if COLAB:
    COLAB = True
    import sshColab
    os.chdir('/root/.kaggle/')
    json_file = 'gcs-colab.json'
    subprocess.call(f'chmod 600 /root/.kaggle/{json_file}', shell=True)        
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = f'/root/.kaggle/{json_file}' 
    subprocess.call('echo $GOOGLE_APPLICATION_CREDENTIALS', shell=True)    
else:
    subprocess.call('pip install ssh-Colab', shell=lTrue)    
    subprocess.call('pip install google-colab', shell=True)

project = "strategic-howl-305522"
bucket_name = "gcs-station-168"           
storage_client = storage.Client(project=project)
bucket = storage_client.bucket(bucket_name)

# #duplicate a bucket's file system
# >>> sshColab.download_to_colab(project, bucket_name, '/kaggle/input')
# #upload example:
# >>> sshColab.upload_to_gcs(project, bucket_name, 'temp9/tps-apr-2021-label/a.pkl', '/temp8/tps-apr-2021-label/a.pkl')    

#%%
# ANCHOR raw data
import os
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from functools import partial
import gc
import re
import pickle
from scipy.special import erfinv
import optuna
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder

MINIMUM_COUNT = 5

TEST = False
if TEST:
    N_SPLITS = 2
    N_TRIALS = 10
    TIMEOUT = 30
    N_REPEATS = 1
    EARLY_STOPPING_ROUNDS = 50
    CARDINALITY_THRESHOLD = 160
else:
    N_SPLITS = 5
    N_TRIALS = 50
    TIMEOUT = 25 * 60
    N_REPEATS = 1
    EARLY_STOPPING_ROUNDS = 50 
    CARDINALITY_THRESHOLD = 160   

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('Time taken : %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

def rank_gauss(x):
    N = x.shape[0]
    temp = x.argsort()
    rank_x = temp.argsort() / N
    rank_x -= rank_x.mean()
    rank_x *= 2
    efi_x = erfinv(rank_x)
    efi_x -= efi_x.mean()
    return efi_x

program_start = timer()

train_df = pd.read_csv('/kaggle/input/tabular-playground-series-apr-2021/train.csv')
test_df = pd.read_csv('/kaggle/input/tabular-playground-series-apr-2021/test.csv')

train_label = train_df['Survived']
train_id = train_df['PassengerId']
test_id = test_df['PassengerId']
del train_df['Survived'], train_df['PassengerId']
del test_df['PassengerId']

data = train_df.append(test_df)
data.reset_index(inplace=True) # Column Index is added.
train_rows = train_df.shape[0]
del data['index']

#%%
# NOTE You may skip this execution block now.
start_time = timer()
# ANCHOR EDA
def list_features_with_missing_values():
    print(data.loc[:, data.columns[data.isnull().sum() > 0].tolist()].isnull().sum().sort_values())
# list_features_with_missing_values()    

INT_TYPE = 'int8'
# Counting missing values - Fare (-267), Embarked (-527), Age (-6779), Ticket (-9804), Cabin (-138697)
data['misCount'] = data[['Fare', 'Embarked', 'Age', 'Ticket', 'Cabin']].isnull().sum(axis=1).astype('int8')
data['misFare'] = data['Fare'].isnull().astype(INT_TYPE)
data['misEmbarked'] = data['Embarked'].isnull().astype(INT_TYPE)
data['misAge'] = data['Age'].isnull().astype(INT_TYPE)
data['misTicket'] = data['Ticket'].isnull().astype(INT_TYPE)
data['misCabin'] = data['Cabin'].isnull().astype(INT_TYPE)

# SOURCE - https://stackoverflow.com/questions/36808434/label-encoder-encoding-missing-values/36814364
def label_encode_column(col):
    nans = col.isnull()
    nan_lst = []
    nan_idx_lst = []
    label_lst = []
    label_idx_lst = []

    for idx, nan in enumerate(nans):
        if nan:
            nan_lst.append(col[idx])
            nan_idx_lst.append(idx)
        else:
            label_lst.append(col[idx])
            label_idx_lst.append(idx)

    nan_df = pd.DataFrame(nan_lst, index=nan_idx_lst)
    label_df = pd.DataFrame(label_lst, index=label_idx_lst) 

    label_encoder = LabelEncoder()
    label_df = label_encoder.fit_transform(label_df.astype(str).values.flatten())
    label_df = pd.DataFrame(label_df, index=label_idx_lst)
    final_col = pd.concat([label_df, nan_df])
    
    return final_col.sort_index()

# Cabin letter
re_letter_number = re.compile("([a-zA-Z]+)(-?[0-9]+)")
def func_cab1(x):
    if re_letter_number.match(x): # we process only when there is anything matched.
        return re_letter_number.match(x).groups()[0]
data['Cab1'] = data['Cabin'].fillna('X-1').map(func_cab1) # string
data['Cab1'].replace('X', np.nan, inplace=True)

# Cabin number (frequency encoding)
def func_cab2(x):
    if re_letter_number.match(x):
        return re_letter_number.match(x).groups()[1]
data['Cab2'] = data['Cabin'].fillna('X-1').map(func_cab2) # string
data['Cab2'].replace('-1', np.nan, inplace=True)  

def group_minority(series, minimum_count):
    mask = series.map(series.value_counts()) < minimum_count
    return series.mask(mask, 'other')

# group teams of size less than MINIMUM_COUNT into a single one.
group_minority_par = partial(group_minority, minimum_count=MINIMUM_COUNT)

data['Cab2'] = data[['Cab2']].apply(group_minority_par).squeeze()

# map_Cab1 = dict()
# for k, v in data.value_counts(['Cab1']).items(): # NOTE This command returns a tuple key, so just take the first item of tuple.
#     map_Cab1[k[0]] = v
# data['Cab1Count'] = data.Cab1.map(map_Cab1) # integer
# data['Cab1Count'].replace(138697, np.nan, inplace=True) # WARN to recount later after imputation of the original feature

# map_Cab2 = dict()
# for k, v in data.value_counts(['Cab2']).items(): # NOTE This command returns a tuple key, so just take the first item of tuple.
#     map_Cab2[k[0]] = v
# data['Cab2Count'] = data.Cab2.map(map_Cab2) # integer
# data['Cab2Count'].replace(138697, np.nan, inplace=True)

missing_code_map = dict()

data['Cab1_enc'] = label_encode_column(data['Cab1'])
tmp = data['Cab1_enc'].nunique()
data['Cab1_enc'].fillna(tmp, inplace=True)
missing_code_map['Cab1_enc'] = tmp

data['Cab2_enc'] = label_encode_column(data['Cab2'])
tmp = data['Cab2_enc'].nunique()
data['Cab2_enc'].fillna(tmp, inplace=True)
missing_code_map['Cab2_enc'] = tmp

# data['Cab2'].replace('-1', np.nan, inplace=True)
idx_missing = data.loc[data['Cab2'].isnull()].index
idx_present = data.loc[data['Cab2'].notnull()].index
tmp_col = data.iloc[idx_present, data.columns.get_loc('Cab2')].str.rjust(5, '0')
for i in range(5):
    data.loc[idx_present, 'Cab2_first'+str(i+1)] = tmp_col.map(lambda x: x[:i+1]) # string
    data['Cab2_first'+str(i+1)+'_enc'] = label_encode_column(data['Cab2_first'+str(i+1)])

    tmp_missing_code = data['Cab2_first'+str(i+1)+'_enc'].nunique()
    data['Cab2_first'+str(i+1)+'_enc'].fillna(tmp_missing_code, inplace=True)
    missing_code_map['Cab2_first'+str(i+1)+'_enc'] = tmp_missing_code

#     tmp_map = dict()
#     for k, v in data.value_counts(['Cab2_first'+str(i+1)+'_enc']).items():
#         tmp_map[k[0]] = v # NOTE TO get information from the returned tuple from value_counts.
#     data['Cab2_first'+str(i+1)+'Count'] = data['Cab2_first'+str(i+1)+'_enc'].map(tmp_map)
#     data['Cab2_first'+str(i+1)+'Count'+'_enc'] = label_encode_column(data['Cab2_first'+str(i+1)+'Count'])

    del data['Cab2_first'+str(i+1)]  

gc.collect()

# convert float64 into int64
tmp_cols = [col for col in data.columns.tolist() if col.startswith('Cab') and not col in ['Cab1', 'Cab2', 'Cabin']]
data[tmp_cols] = data[tmp_cols].astype('int64')


# OBSOLTE
# # Clustering Cabin numbers; recover missing values to np.nan on the fly.
# data.Cab2 = data.Cab2.replace('-1', np.nan) # NOTE Recover NaN values to prepare for imputation
# tmp = data['Cab2'].copy().fillna('0').str.rjust(5, '0')
# for i in range(5):
#     data['Cab2_first'+str(i+1)] = tmp.map(lambda x: np.nan if x=='00000' else x[:i+1]) # string
#     tmp_map = dict()
#     for k, v in data.value_counts(['Cab2_first'+str(i+1)]).items():
#         tmp_map[k[0]] = v # NOTE TO get information from the returned tuple from value_counts.
#     data['Cab2_first' + str(i+1) + 'Count'] = data['Cab2_first'+str(i+1)].map(tmp_map)
#     del data['Cab2_first'+str(i+1)]

# NAME
data[['Last', 'First']] = data.Name.str.split(', ', expand=True)

# map_Last = dict()
# for k, v in data.value_counts(['Last']).items():
#     map_Last[k[0]] = v
# data['LastnameCount'] = data.Last.map(map_Last) # integer

# map_First = dict()
# for k, v in data.value_counts(['First']).items():
#     map_First[k[0]] = v
# data['FirstnameCount'] = data.First.map(map_First) # integer

data['Last_enc'] = label_encode_column(data['Last'])
data['Last_enc'] = data['Last_enc'].astype('int64')
data['First_enc'] = label_encode_column(data['First'])
data['First_enc'] = data['First_enc'].astype('int64')

# Sex
data['Sex_enc'] = label_encode_column(data['Sex'])
data['Sex_enc'].astype('int8')

data['FamilySize'] = data['SibSp'] + data['Parch']

# Embarked
# data.Embarked.value_counts(normalize=True) # [71%, 22%, 7%]
data['Embarked_enc'] = label_encode_column(data['Embarked'])
tmp_missing_code = data['Embarked_enc'].nunique()
missing_code_map['Embarked_enc'] = tmp_missing_code
data['Embarked_enc'].fillna(tmp_missing_code, inplace=True)
data['Embarked_enc'] = data['Embarked_enc'].astype('int64')

# tmp_map = dict()
# for k, v in data.value_counts(['Embarked']).items():
#     tmp_map[k[0]] = v
# data['EmbarkedCount'] = data['Embarked'].map(tmp_map)
# data[['EmbarkedCount','Embarked']]
# # data.loc[data['Embarked']=='<NA>', 'EmbarkedCount']
# data['EmbarkedCount'] = data['EmbarkedCount'].replace(527, -1)

# TICKET
new = data.Ticket.str.split(expand=True)
data['Tick1'] = np.where(new[0].str.isnumeric(), np.nan, new[0]) # TECH str.isnumeric
data['misTick1'] = data['Tick1'].isnull().astype('int8')

# map_Tick1 = dict()
# for k, v in data.value_counts(['Tick1']).items():
#     map_Tick1[k[0]] = v
# data['Tick1Count'] = data.Tick1.map(map_Tick1) # WARN to recount later after imputation of the original feature
# # NOTE Unlike the procedure we did on feature CABIN, we put off the np.nan imputation and will do it altogether with other features.

data['Tick1_enc'] = label_encode_column(data['Tick1'])
tmp_missing_code = data['Tick1_enc'].nunique()
missing_code_map['Tick1_enc'] = tmp_missing_code
data['Tick1_enc'].fillna(tmp_missing_code, inplace=True)
data['Tick1_enc'] = data['Tick1_enc'].astype('int64')

data['Tick2'] = data.Ticket.str.extract('(\d+)') # TECH str.extract
data['misTick2'] = data['Tick2'].isnull().astype('int8')
# data = data.assign(misTick2=np.where(data.Tick2.isnull(), 1, 0))

# map_Tick2 = dict()
# for k, v in data.value_counts(['Tick2']).items(): # NOTE This command returns a tuple key, so just take the first item of tuple.
#     map_Tick2[k[0]] = v
# data['Tick2Count'] = data.Tick2.map(map_Tick2) 

# Ticket number clustering
# pd.to_numeric(data.Tick2, errors='ignore').max() # to check the maximum number of digits needed.

idx_missing = data.loc[data['Tick2'].isnull()].index
idx_present = data.loc[data['Tick2'].notnull()].index
tmp_col = data.iloc[idx_present, data.columns.get_loc('Tick2')].str.rjust(7, '0')

# def group_minority(series, minimum_count):
#     mask = series.map(series.value_counts()) < minimum_count
#     return series.mask(mask, 'other')

# # group teams of size less than MINIMUM_COUNT into a single one.
# group_minority_par = partial(group_minority, minimum_count=MINIMUM_COUNT)

for i in range(7):
    data.loc[idx_present, 'Tick2_first'+str(i+1)] = tmp_col.map(lambda x: x[:i+1]) # string
    data['Tick2_first'+str(i+1)+'_enc'] = data[['Tick2_first'+str(i+1)]].apply(group_minority_par).squeeze()    
    data['Tick2_first'+str(i+1)+'_enc'].where(data['Tick2_first'+str(i+1)+'_enc']!='other', '-1', inplace=True)
    data['Tick2_first'+str(i+1)+'_enc'] = label_encode_column(data['Tick2_first'+str(i+1)+'_enc'])

    tmp_missing_code = data['Tick2_first'+str(i+1)+'_enc'].nunique()
    data['Tick2_first'+str(i+1)+'_enc'].fillna(tmp_missing_code, inplace=True)
    missing_code_map['Tick2_first'+str(i+1)+'_enc'] = tmp_missing_code    
    data['Tick2_first'+str(i+1)+'_enc'] = data['Tick2_first'+str(i+1)+'_enc'].astype('int64')
    del data['Tick2_first'+str(i+1)]     
    
gc.collect()  

# #OBSOLETE
#     tmp_map = dict()
#     for k, v in data.value_counts(['Tick2_first'+str(i+1)]).items():
#         tmp_map[k[0]] = v
#     data['Tick2_first'+str(i+1)+'Count'] = data['Tick2_first'+str(i+1)].map(tmp_map)

# # Sanity check
# cols = [col for col in data.columns if 'Tick2' in col and not col.endswith('Tick2')]
# for i, col in enumerate(cols):
#     print(data[col].value_counts())
# for i, col in enumerate(cols):
#     print(data[col].isnull().sum())
# data.Tick2.isnull().sum()   

data['Pclass'+'_enc'] = label_encode_column(data.Pclass)

del data['Name']
del data['Last']
del data['First']
del data['Cabin']
del data['Cab1']
del data['Cab2']
# del data['Cab2_enc']
del data['Cab2_first5_enc']
del data['Sex']
del data['Embarked']
del data['Tick1']
del data['Tick2']
del data['Ticket']
del data['Pclass']
gc.collect()

timer(start_time)

# # Age and Fare
# # SOURCE dealing with outliers even for tree-based algorithm easier for splitting - https://www.kdnuggets.com/2018/08/make-machine-learning-models-robust-outliers.html
# # TODO data.loc[data['Age'].notnull(), 'Age_rg'] = rank_gauss(data.loc[data['Age'].notnull(), 'Age'].values) 
# # TODO data.loc[data['Fare'].notnull(), 'Fare_rg'] = rank_gauss(data.loc[data['Fare'].notnull(), 'Fare'].values) 

#DONE!
os.chdir('/kaggle/working')
pickle.dump(data, open('1clean_data.pkl', 'wb'))
blob = bucket.blob('tps-apr-2021-label/1clean_data.pkl')
blob.upload_from_filename('/kaggle/working/1clean_data.pkl')

# missing_code_map
os.chdir('/kaggle/working')
pickle.dump(missing_code_map, open('2missing_code_map.pkl', 'wb'))
blob = bucket.blob('tps-apr-2021-label/2missing_code_map.pkl')
blob.upload_from_filename('/kaggle/working/2missing_code_map.pkl')

# ANCHOR Age

#%%
sshColab.download_to_colab(project, bucket_name, 
    destination_directory = '/kaggle/working', 
    remote_blob_path='tps-apr-2021-label/1clean_data.pkl', 
    local_file_name='1clean_data.pkl') 
data = pickle.load(open('/kaggle/working/1clean_data.pkl', 'rb'))

sshColab.download_to_colab(project, bucket_name, 
    destination_directory = '/kaggle/working', 
    remote_blob_path='tps-apr-2021-label/2missing_code_map.pkl', 
    local_file_name='2missing_code_map.pkl') 
missing_code_map = pickle.load(open('/kaggle/working/2missing_code_map.pkl', 'rb'))

start_time = timer()

the_col = 'Age'
idx_present = data.loc[data[the_col].notnull(), :].index.tolist()
idx_missing = data.loc[data[the_col].isnull(), :].index.tolist()
x_train_full = data.loc[idx_present, :].drop(the_col, axis=1)
y_train_full = data.loc[idx_present, :][the_col]
x_test_full = data.loc[idx_missing, :].drop(the_col, axis=1)

cat_features = [col for col in x_train_full.columns if col not in ['Age', 'Fare']]
x_train_full[cat_features] = x_train_full[cat_features].astype(int) # WARN categorical features can only be integer or string format.
x_test_full[cat_features] = x_test_full[cat_features].astype(int)


# ANCHOR Three out-of-shelf GBMs experimental zone 
# All the three GBMs are used without any hyperparameter setup.
# XGBRegressor
def vanilla_xgbregressor():
    start_time = timer()
    xgbmodel = XGBRegressor(objective="reg:squarederror")
    x_A, x_B, y_A, y_B = train_test_split(
        x_train_full, y_train_full, test_size=0.2, random_state=5)
    xgbmodel.fit(x_A, y_A, eval_set=[(x_B, y_B)], eval_metric='rmse',
        early_stopping_rounds=50, verbose=0)
    print(xgbmodel.best_score) # 15.181376 and takes only 52 seconds
    timer(start_time)

params = {
    'objective': 'reg:squarederror',
    'n_estimators': 700,
    'booster': 'gbtree',
    'verbosity': 0,
    'tree_method': 'gpu_hist'
}

# start_time = timer()
# x_train = x_train_full.values
# y_train = y_train_full.values
# rmse = []
# xgb_regressor = XGBRegressor(**params)
# rkf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS)
# for i, (train_index, valid_index) in enumerate(rkf.split(x_train)):
#     X_A, X_B = x_train[train_index, :], x_train[valid_index, :]
#     y_A, y_B = y_train[train_index], y_train[valid_index]
#     xgb_regressor.fit(
#         X_A, y_A, eval_set=[(X_B, y_B)], early_stopping_rounds=50
#     )
#     tmp = xgb_regressor.predict(X_B)
#     rmse.append(mean_squared_error(y_B, tmp, squared=False))
# timer(start_time)

def objective(trial, x_train, y_train, params=params):
    temp_map = {
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.005, 0.05),
        "colsample_bytree": trial.suggest_loguniform("colsample_bytree", 0.2, 0.8),
        "subsample": trial.suggest_loguniform("subsample", 0.4, 0.8),
        "alpha": trial.suggest_loguniform("alpha", 0.01, 10.0),
        "lambda": trial.suggest_loguniform("lambda", 1e-8, 10.0),
        "gamma": trial.suggest_loguniform("lambda", 1e-8, 10.0),
        "min_child_weight": trial.suggest_loguniform("min_child_weight", 50, 1000),        
        'max_depth': trial.suggest_int('max_depth', 3, 10)
    }
    params.update(temp_map)
    x_train = x_train.values
    y_train = y_train.values
    rmse = []
    xgb_regressor = XGBRegressor(**params)
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation_0-rmse")
    rkf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS)
    for i, (train_index, valid_index) in enumerate(rkf.split(x_train)):
        X_A, X_B = x_train[train_index, :], x_train[valid_index, :]
        y_A, y_B = y_train[train_index], y_train[valid_index]
        xgb_regressor.fit(
            X_A, y_A, eval_set=[(X_B, y_B)], 
            early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=0,
            callbacks=[pruning_callback]
        )
        tmp = xgb_regressor.predict(X_B)
        rmse.append(mean_squared_error(y_B, tmp, squared=False))
    trial.set_user_attr(key="best_booster", value=xgb_regressor) # NOTE update the best model in the optuna's table.
    return np.mean(rmse) 

# SOURCE retrieve the best model - https://stackoverflow.com/questions/62144904/python-how-to-retrive-the-best-model-from-optuna-lightgbm-study#answer-63365355
def save_best(study, trial):
    if study.best_trial.number == trial.number:
        # Set the best booster as a trial attribute; accessible via study.trials_dataframe.
        study.set_user_attr(key="best_booster", value=trial.user_attrs["best_booster"])
        # SOURCE retrieve the best number of estimators https://github.com/optuna/optuna/issues/1169    

study = optuna.create_study(
    direction = "minimize", 
    sampler = optuna.samplers.TPESampler(seed=2021, multivariate=True),
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)
)
study.optimize(lambda trial: objective(trial, x_train_full, y_train_full, params), 
                   n_trials=N_TRIALS, timeout=TIMEOUT, n_jobs=1, callbacks=[save_best]
)
# display params
hp = study.best_params
for key, value in hp.items():
    print(f"{key:>20s} : {value}")
print(f"{'best objective value':>20s} : {study.best_value}")  

# %%
# optuna.visualization.plot_optimization_history(study)

# trials_df = study.trials_dataframe()
# trials_df

def predict_float_missing_value(model, X, y, X_mis):
    rkf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS)
    X_values = X.values
    y_values = y.values
    X_test = X_mis.values
    y_pred = np.zeros((X_test.shape[0], 1))

    for i, (train_index, test_index) in enumerate(rkf.split(X_values)):
        X_A, X_B = X_values[train_index, :], X_values[test_index, :]
        y_A, y_B = y_values[train_index], y_values[test_index]
        model.fit(
            X_A, y_A, eval_set=[(X_B, y_B)],
            eval_metric="rmse",
            early_stopping_rounds=EARLY_STOPPING_ROUNDS, # WARN 
            verbose=0
        )
        # SOURCE model.best_ntree_limit. https://stackoverflow.com/questions/51955256/xgboost-best-iteration
        y_pred += model.predict(X_test, ntree_limit=model.best_ntree_limit).reshape(-1,1) # WARN I added the 2nd argument.
    y_pred /= N_SPLITS * N_REPEATS
    return y_pred    

y_pred = predict_float_missing_value(
    study.user_attrs['best_booster'],
    x_train_full,
    y_train_full,
    x_test_full)

timer(start_time)

imputed_df = pd.DataFrame()
imputed_df[the_col + '_imp'] = data.loc[:, the_col].copy()
imputed_df.loc[idx_missing, the_col + '_imp'] = y_pred.flatten()

os.chdir('/kaggle/working')
pickle.dump(imputed_df, open('3imputed_df.pkl', 'wb'))
sshColab.upload_to_gcs(project, bucket_name, f'tps-apr-2021-label/3imputed_df.pkl', f'/kaggle/working/3imputed_df.pkl')    

# ANCHOR Fare
sshColab.download_to_colab(project, bucket_name, 
    destination_directory = '/kaggle/working', 
    remote_blob_path='tps-apr-2021-label/1clean_data.pkl', 
    local_file_name='1clean_data.pkl') 
data = pickle.load(open('/kaggle/working/1clean_data.pkl', 'rb'))

sshColab.download_to_colab(project, bucket_name, 
    destination_directory = '/kaggle/working', 
    remote_blob_path='tps-apr-2021-label/2missing_code_map.pkl', 
    local_file_name='2missing_code_map.pkl') 
missing_code_map = pickle.load(open('/kaggle/working/2missing_code_map.pkl', 'rb'))

sshColab.download_to_colab(project, bucket_name, 
    destination_directory = '/kaggle/working', 
    remote_blob_path='tps-apr-2021-label/3imputed_df.pkl', 
    local_file_name='imputed_df.pkl') 
imputed_df = pickle.load(open('/kaggle/working/3imputed_df.pkl', 'rb'))

start_time = timer()

the_col = 'Fare'
idx_present = data.loc[data[the_col].notnull(), :].index.tolist()
idx_missing = data.loc[data[the_col].isnull(), :].index.tolist()
data[the_col] = data[the_col].replace(-1, np.nan)
x_train_full = data.loc[idx_present, :].drop(the_col, axis=1)
y_train_full = data.loc[idx_present, :][the_col]
x_test_full = data.loc[idx_missing, :].drop(the_col, axis=1)

cat_features = [col for col in x_train_full.columns if col not in ['Age', 'Fare'] and not col.endswith('_imp')]
x_train_full[cat_features] = x_train_full[cat_features].astype(int) # WARN categorical features can only be integer or string format.
x_test_full[cat_features] = x_test_full[cat_features].astype(int)

study = optuna.create_study(
    direction = "minimize", 
    sampler = optuna.samplers.TPESampler(seed=2021, multivariate=True),
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)
)
study.optimize(lambda trial: objective(trial, x_train_full, y_train_full, params), 
                   n_trials=N_TRIALS, timeout=TIMEOUT, n_jobs=1, callbacks=[save_best]
)
# display params
hp = study.best_params
for key, value in hp.items():
    print(f"{key:>20s} : {value}")
print(f"{'best objective value':>20s} : {study.best_value}")  

y_pred = predict_float_missing_value(
    study.user_attrs['best_booster'],
    x_train_full,
    y_train_full,
    x_test_full)

timer(start_time)

imputed_df[the_col + '_imp'] = data.loc[:, the_col].copy()
imputed_df.loc[idx_missing, the_col + '_imp'] = y_pred.flatten()

os.chdir('/kaggle/working')
pickle.dump(imputed_df, open('4imputed_df.pkl', 'wb'))
sshColab.upload_to_gcs(project, bucket_name, f'tps-apr-2021-label/4imputed_df.pkl', f'/kaggle/working/4imputed_df.pkl')

# ANCHOR Embarked
sshColab.download_to_colab(project, bucket_name, 
    destination_directory = '/kaggle/working', 
    remote_blob_path='tps-apr-2021-label/1clean_data.pkl', 
    local_file_name='1clean_data.pkl') 
data = pickle.load(open('/kaggle/working/1clean_data.pkl', 'rb'))

sshColab.download_to_colab(project, bucket_name, 
    destination_directory = '/kaggle/working', 
    remote_blob_path='tps-apr-2021-label/2missing_code_map.pkl', 
    local_file_name='2missing_code_map.pkl') 
missing_code_map = pickle.load(open('/kaggle/working/2missing_code_map.pkl', 'rb'))

sshColab.download_to_colab(project, bucket_name, 
    destination_directory = '/kaggle/working', 
    remote_blob_path='tps-apr-2021-label/4imputed_df.pkl', 
    local_file_name='imputed_df.pkl') 
imputed_df = pickle.load(open('/kaggle/working/4imputed_df.pkl', 'rb'))

start_time = timer()

the_col = 'Embarked_enc'

missing_bool = data[the_col] == missing_code_map[the_col]
idx_present = data.loc[~missing_bool, :].index.tolist()
idx_missing = data.loc[missing_bool, :].index.tolist()
data[the_col] = data[the_col].replace(missing_code_map[the_col], np.nan) # XGboost works with missing values.
x_train_full = data.loc[idx_present, :].drop(the_col, axis=1)
y_train_full = data.loc[idx_present, :][the_col]
x_test_full = data.loc[idx_missing, :].drop(the_col, axis=1)
# Note: xgboost allows continuous features to be missing.
cat_features = [col for col in x_train_full.columns if col not in ['Age', 'Fare']]
x_train_full[cat_features] = x_train_full[cat_features].astype(int) # WARN categorical features can only be integer or string format.
x_test_full[cat_features] = x_test_full[cat_features].astype(int)

y_train_full = y_train_full.astype(int) # <------- To avoid warning message of Xgboost, ensure target features is integer.

params = {
    'objective': 'multi:softprob',
    'num_class': data[the_col].nunique(), # when you;re doing binary classification, comment out this one.
    'n_estimators': 700,
    'booster': 'gbtree',
    'verbosity': 0,
    'average': 'macro',
    'eval_metric': 'merror',
    'tree_method': 'gpu_hist',
    'use_label_encoder': False # <------- You needs to make sure target feature is integer as well.
}

# start_time = timer()
# x_train = x_train_full.values
# y_train = y_train_full.values
# roc_test = []
# xgb_classifier = XGBClassifier(**params)
# rkf = RepeatedKFold(n_splits=2, n_repeats=1)
# for i, (train_index, valid_index) in enumerate(rkf.split(x_train)):
#     X_A, X_B = x_train[train_index, :], x_train[valid_index, :]
#     y_A, y_B = y_train[train_index], y_train[valid_index]
#     xgb_classifier.fit(
#         X_A, y_A, eval_set=[(X_B, y_B)], early_stopping_rounds=50
#     )
#     tmp = xgb_classifier.predict_proba(X_B)
#     roc_test.append(roc_auc_score(y_B, tmp, multi_class='ovo'))
# print(np.mean(roc_test))
# timer(start_time)

# for i, v in zip(np.arange(3), xgb_classifier.classes_):
#     print(i,v )

def objective(trial, x_train, y_train, params=params):
    temp_map = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.005, 0.05),
        "min_child_weight": trial.suggest_loguniform("min_child_weight", 50, 1000),
        "subsample": trial.suggest_loguniform("subsample", 0.4, 0.8),
        "colsample_bytree": trial.suggest_loguniform("colsample_bytree", 0.2, 0.8),
        "alpha": trial.suggest_loguniform("alpha", 0.01, 10.0),
        "lambda": trial.suggest_loguniform("lambda", 1e-8, 10.0),
        "gamma": trial.suggest_loguniform("lambda", 1e-8, 10.0)
    }
    params.update(temp_map)
    x_train = x_train.values
    y_train = y_train.values
    roc_test = []
    xgb_classifier = XGBClassifier(**params)
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation_0-merror")
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS)
    for i, (train_index, valid_index) in enumerate(rskf.split(x_train, y_train)):
        X_A, X_B = x_train[train_index, :], x_train[valid_index, :]
        y_A, y_B = y_train[train_index], y_train[valid_index]
        xgb_classifier.fit(
            X_A, y_A, eval_set=[(X_B, y_B)], 
            early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=0,
            callbacks=[pruning_callback]
        )
        tmp = xgb_classifier.predict_proba(X_B)
        roc_test.append(roc_auc_score(y_B, tmp, multi_class='ovo'))
    trial.set_user_attr(key="best_booster", value=xgb_classifier) # NOTE update the best model in the optuna's table.
    return np.mean(roc_test) 

# SOURCE retrieve the best model - https://stackoverflow.com/questions/62144904/python-how-to-retrive-the-best-model-from-optuna-lightgbm-study#answer-63365355
def save_best(study, trial):
    if study.best_trial.number == trial.number:
        # Set the best booster as a trial attribute; accessible via study.trials_dataframe.
        study.set_user_attr(key="best_booster", value=trial.user_attrs["best_booster"])
        # SOURCE retrieve the best number of estimators https://github.com/optuna/optuna/issues/1169    

study = optuna.create_study(
    direction = "maximize", 
    sampler = optuna.samplers.TPESampler(seed=2021, multivariate=True),
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)
)
study.optimize(lambda trial: objective(trial, x_train_full, y_train_full, params), 
                   n_trials=N_TRIALS, timeout=TIMEOUT, n_jobs=1, callbacks=[save_best]
)
# display params
hp = study.best_params
for key, value in hp.items():
    print(f"{key:>20s} : {value}")
print(f"{'best objective value':>20s} : {study.best_value}")  

def predict_categorical_missing_value(model, X, y, X_mis):
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS)
    X_values = X.values
    y_values = y.values
    X_test = X_mis.values
    y_pred_class_prob = np.zeros((X_test.shape[0], y.nunique()))

    for i, (train_index, test_index) in enumerate(rskf.split(X_values, y_values)):
        X_A, X_B = X_values[train_index, :], X_values[test_index, :]
        y_A, y_B = y_values[train_index], y_values[test_index]
        model.fit(
            X_A, y_A, eval_set=[(X_B, y_B)],
            eval_metric="merror",
            early_stopping_rounds=EARLY_STOPPING_ROUNDS, # WARN 
            verbose=0
        )
        # SOURCE model.best_ntree_limit. https://stackoverflow.com/questions/51955256/xgboost-best-iteration
        y_pred_class_prob += model.predict_proba(X_test, ntree_limit=model.best_ntree_limit) # WARN I added the 2nd argument.
    y_pred_class_prob /= N_SPLITS * N_REPEATS
    return y_pred_class_prob    

y_pred = predict_categorical_missing_value(
    study.user_attrs['best_booster'],
    x_train_full,
    y_train_full,
    x_test_full)

timer(start_time)

imputed_df[the_col + '_imp'] = data.loc[:, the_col].copy()
class_map = dict(zip(np.arange(3), study.user_attrs['best_booster'].classes_))
imputed_df.loc[idx_missing, the_col + '_imp'] = pd.Series(np.argmax(y_pred, axis=1)).map(class_map).values

os.chdir('/kaggle/working')
pickle.dump(imputed_df, open('5imputed_df.pkl', 'wb'))
sshColab.upload_to_gcs(project, bucket_name, f'tps-apr-2021-label/5imputed_df.pkl', f'/kaggle/working/5imputed_df.pkl')

# ANCHOR Cabin + Ticket
# sshColab.download_to_colab(project, bucket_name, 
#     destination_directory = '/kaggle/working', 
#     remote_blob_path='tps-apr-2021-label/1clean_data.pkl', 
#     local_file_name='1clean_data.pkl') 
# data = pickle.load(open('/kaggle/working/1clean_data.pkl', 'rb'))

sshColab.download_to_colab(project, bucket_name, 
    destination_directory = '/kaggle/working', 
    remote_blob_path='tps-apr-2021-label/2missing_code_map.pkl', 
    local_file_name='2missing_code_map.pkl') 
missing_code_map = pickle.load(open('/kaggle/working/2missing_code_map.pkl', 'rb'))

sshColab.download_to_colab(project, bucket_name, 
    destination_directory = '/kaggle/working', 
    remote_blob_path='tps-apr-2021-label/5imputed_df.pkl', 
    local_file_name='imputed_df.pkl') 
imputed_df = pickle.load(open('/kaggle/working/5imputed_df.pkl', 'rb'))


def non_high_cardinality_features(threshold = 60):
    list_cols = [col for col in data.columns if col.startswith('Cab') or col.startswith('Tick')]
    unique_ss = data[list_cols].nunique()    
    # print(unique_ss)
    return unique_ss.loc[unique_ss < threshold].index.tolist()

list_cols = non_high_cardinality_features(threshold=CARDINALITY_THRESHOLD)  
list_cols

for the_col in list_cols:  

    start_time = timer()

    # added
    sshColab.download_to_colab(project, bucket_name, 
        destination_directory = '/kaggle/working', 
        remote_blob_path='tps-apr-2021-label/1clean_data.pkl', 
        local_file_name='1clean_data.pkl') 
    data = pickle.load(open('/kaggle/working/1clean_data.pkl', 'rb'))

    print(f'\n\n========================= {the_col} ==============================\n\n')
    
    # maybe revised, check again
    missing_bool = data[the_col] == missing_code_map[the_col]
    idx_present = data.loc[~missing_bool, :].index.tolist()
    idx_missing = data.loc[missing_bool, :].index.tolist()
    tmp = data[the_col].values
    data[the_col] = data[the_col].replace(missing_code_map[the_col], np.nan)

    x_train_full = data.loc[idx_present, :].drop(the_col, axis=1)
    y_train_full = data.loc[idx_present, :][the_col]
    y_train_full = y_train_full.astype(int) # <-------
    x_test_full = data.loc[idx_missing, :].drop(the_col, axis=1)
    data[the_col] = tmp.copy()

    # e.g., when impute Cab2 related feature, you should use the rest Cab2 features as features.
    if the_col.startswith('Cab1'):
        cat_features = [col for col in x_train_full.columns if col not in ['Age', 'Fare'] and not col.startswith('Cab1')]
        x_features = [col for col in x_train_full.columns if not col.startswith('Cab1')]           
    if the_col.startswith('Tick1'):
        cat_features = [col for col in x_train_full.columns if col not in ['Age', 'Fare'] and not col.startswith('Tick1')]    
        x_features = [col for col in x_train_full.columns if not col.startswith('Tick1')]    
    if the_col.startswith('Cab2'):
        cat_features = [col for col in x_train_full.columns if col not in ['Age', 'Fare'] and not col.startswith('Cab2')]
        x_features = [col for col in x_train_full.columns if not col.startswith('Cab2')]           
    if the_col.startswith('Tick2'):
        cat_features = [col for col in x_train_full.columns if col not in ['Age', 'Fare'] and not col.startswith('Tick2')]    
        x_features = [col for col in x_train_full.columns if not col.startswith('Tick2')]

    x_train_full[cat_features] = x_train_full[cat_features].astype(int) # WARN categorical features can only be integer or string format.
    x_test_full[cat_features] = x_test_full[cat_features].astype(int)

    x_train_full = x_train_full.loc[:, x_features]
    x_test_full = x_test_full.loc[:, x_features]    


    params = {
        'objective': 'multi:softprob',
        'num_class': data[the_col].nunique(),
        'n_estimators': 700,
        'booster': 'gbtree',
        'verbosity': 0,
        'average': 'macro',
        'eval_metric': 'merror',
        'tree_method': 'gpu_hist',
        'use_label_encoder': False # <-------        
    }

    # start_time = timer()
    # x_train = x_train_full.values
    # y_train = y_train_full.values
    # roc_test = []
    # xgb_classifier = XGBClassifier(**params)
    # rkf = RepeatedKFold(n_splits=2, n_repeats=1)
    # for i, (train_index, valid_index) in enumerate(rkf.split(x_train)):
    #     X_A, X_B = x_train[train_index, :], x_train[valid_index, :]
    #     y_A, y_B = y_train[train_index], y_train[valid_index]
    #     xgb_classifier.fit(
    #         X_A, y_A, eval_set=[(X_B, y_B)], early_stopping_rounds=50
    #     )
    #     tmp = xgb_classifier.predict_proba(X_B)
    #     roc_test.append(roc_auc_score(y_B, tmp, multi_class='ovo'))
    # print(np.mean(roc_test))
    # timer(start_time)

    # for i, v in zip(np.arange(3), xgb_classifier.classes_):
    #     print(i,v )

    def objective(trial, x_train, y_train, params=params):
        temp_map = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.005, 0.05),
            "min_child_weight": trial.suggest_loguniform("min_child_weight", 50, 1000),
            "subsample": trial.suggest_loguniform("subsample", 0.4, 0.8),
            "colsample_bytree": trial.suggest_loguniform("colsample_bytree", 0.2, 0.8),
            "alpha": trial.suggest_loguniform("alpha", 0.01, 10.0),
            "lambda": trial.suggest_loguniform("lambda", 1e-8, 10.0),
            "gamma": trial.suggest_loguniform("lambda", 1e-8, 10.0)
        }
        params.update(temp_map)
        x_train = x_train.values
        y_train = y_train.values
        roc_test = []
        xgb_classifier = XGBClassifier(**params)
        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation_0-merror")
        rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS)
        for i, (train_index, valid_index) in enumerate(rskf.split(x_train, y_train)):
            X_A, X_B = x_train[train_index, :], x_train[valid_index, :]
            y_A, y_B = y_train[train_index], y_train[valid_index]
            xgb_classifier.fit(
                X_A, y_A, eval_set=[(X_B, y_B)], 
                early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=0,
                callbacks=[pruning_callback]
            )
            tmp = xgb_classifier.predict_proba(X_B)
            roc_test.append(roc_auc_score(y_B, tmp, multi_class='ovo'))
        trial.set_user_attr(key="best_booster", value=xgb_classifier) # NOTE update the best model in the optuna's table.
        return np.mean(roc_test) 

    # SOURCE retrieve the best model - https://stackoverflow.com/questions/62144904/python-how-to-retrive-the-best-model-from-optuna-lightgbm-study#answer-63365355
    def save_best(study, trial):
        if study.best_trial.number == trial.number:
            # Set the best booster as a trial attribute; accessible via study.trials_dataframe.
            study.set_user_attr(key="best_booster", value=trial.user_attrs["best_booster"])
            # SOURCE retrieve the best number of estimators https://github.com/optuna/optuna/issues/1169    

    study = optuna.create_study(
        direction = "maximize", 
        sampler = optuna.samplers.TPESampler(seed=2021, multivariate=True),
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)
    )
    study.optimize(lambda trial: objective(trial, x_train_full, y_train_full, params), 
                       n_trials=N_TRIALS, timeout=TIMEOUT, n_jobs=1, callbacks=[save_best]
    )
    # display params
    hp = study.best_params
    for key, value in hp.items():
        print(f"{key:>20s} : {value}")
    print(f"{'best objective value':>20s} : {study.best_value}")  

    def predict_categorical_missing_value(model, X, y, X_mis):
        rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS)
        X_values = X.values
        y_values = y.values
        X_test = X_mis.values
        y_pred_class_prob = np.zeros((X_test.shape[0], y.nunique()))

        for i, (train_index, test_index) in enumerate(rskf.split(X_values, y_values)):
            X_A, X_B = X_values[train_index, :], X_values[test_index, :]
            y_A, y_B = y_values[train_index], y_values[test_index]
            model.fit(
                X_A, y_A, eval_set=[(X_B, y_B)],
                eval_metric="merror",
                early_stopping_rounds=EARLY_STOPPING_ROUNDS, # WARN 
                verbose=0
            )
            # SOURCE model.best_ntree_limit. https://stackoverflow.com/questions/51955256/xgboost-best-iteration
            y_pred_class_prob += model.predict_proba(X_test, ntree_limit=model.best_ntree_limit) # WARN I added the 2nd argument.
        y_pred_class_prob /= N_SPLITS * N_REPEATS
        return y_pred_class_prob    

    y_pred = predict_categorical_missing_value(
        study.user_attrs['best_booster'],
        x_train_full,
        y_train_full,
        x_test_full)
    
    timer(start_time)

    # modified
    imputed_df[the_col + '_imp'] = data.loc[:, the_col].copy()
    imputed_df.loc[idx_missing, the_col + '_imp'] = np.argmax(y_pred, axis=1)

    os.chdir('/kaggle/working')
    pickle.dump(imputed_df, open('6imputed_df.pkl', 'wb'))
    sshColab.upload_to_gcs(project, bucket_name, f'tps-apr-2021-label/6imputed_df.pkl', f'/kaggle/working/6imputed_df.pkl')

timer(program_start)

# construction
imputed_df.info()
for col in imputed_df.columns:
    print(imputed_df[col].nunique())

# construction
def list_difference(a,b):
    a, b = set(a), set(b)
    return a.difference(b)

len(cat_features), len(x_features), the_col, len(x_train_full), len(x_test_full)
len(x_train_full.columns)
# data.info()
len(data.columns)

# construction
sshColab.download_to_colab(project, bucket_name, 
    destination_directory = '/kaggle/working', 
    remote_blob_path='tps-apr-2021-label/6imputed_df.pkl', 
    local_file_name='imputed_df.pkl') 
temp = pickle.load(open('/kaggle/working/6imputed_df.pkl', 'rb'))
temp