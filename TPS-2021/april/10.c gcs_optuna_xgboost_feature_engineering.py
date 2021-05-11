#%% 
import os, subprocess
from google.cloud import storage
import sshColab

# ANCHOR GCS boilerplate
os.chdir('/root/.kaggle/')
json_file = 'gcs-colab.json'
subprocess.call(f'chmod 600 /root/.kaggle/{json_file}', shell=True)        
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = f'/root/.kaggle/{json_file}' 
subprocess.call('echo $GOOGLE_APPLICATION_CREDENTIALS', shell=True)
project = "strategic-howl-305522"
bucket_name = "gcs-station-168"           
storage_client = storage.Client(project=project)
bucket = storage_client.bucket(bucket_name)
# !boilerplate

# duplicate a bucket's file system
# sshColab.download_to_colab(project, bucket_name, '/kaggle/input')

# upload example:
# >>> upload_to_gcs(project, bucket_name, 'temp9/tps-apr-2021-label/a.pkl', '/temp8/tps-apr-2021-label/a.pkl')    

# ANCHOR raw data
import os
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import re
import pickle
import gc
from scipy.special import erfinv
import optuna
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold

def rank_gauss(x):
    N = x.shape[0]
    temp = x.argsort()
    rank_x = temp.argsort() / N
    rank_x -= rank_x.mean()
    rank_x *= 2
    efi_x = erfinv(rank_x)
    efi_x -= efi_x.mean()
    return efi_x

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

# ANCHOR EDA
def list_features_with_missing_values():
    print(data.loc[:, data.columns[data.isnull().sum() > 0].tolist()].isnull().sum().sort_values())
# list_features_with_missing_values()    

# Counting missing values - Fare (-267), Embarked (-527), Age (-6779), Ticket (-9804), Cabin (-138697)
data['misCount'] = data[['Fare', 'Embarked', 'Age', 'Ticket', 'Cabin']].isnull().sum(axis=1)
data['misFare'] = data['Fare'].isnull().astype(int)
data['misEmbarked'] = data['Embarked'].isnull().astype(int)
data['misAge'] = data['Age'].isnull().astype(int)
data['misTicket'] = data['Ticket'].isnull().astype(int)
data['misCabin'] = data['Cabin'].isnull().astype(int)

# Cabin letter
re_letter_number = re.compile("([a-zA-Z]+)(-?[0-9]+)")
def func_cab1(x):
    if re_letter_number.match(x): # we process only when there is anything matched.
        return re_letter_number.match(x).groups()[0]
data['Cab1'] = data['Cabin'].fillna('X-1').map(func_cab1) # string

# Cabin number (frequency encoding)
def func_cab2(x):
    if re_letter_number.match(x):
        return re_letter_number.match(x).groups()[1]
data['Cab2'] = data['Cabin'].fillna('X-1').map(func_cab2) # string

del data['Cabin']

map_Cab1 = dict()
for k, v in data.value_counts(['Cab1']).items(): # NOTE This command returns a tuple key, so just take the first item of tuple.
    map_Cab1[k[0]] = v
data['Cab1Count'] = data.Cab1.map(map_Cab1) # integer
data.Cab1Count.replace(138697, -1, inplace=True) # WARN to recount later after imputation of the original feature

map_Cab2 = dict()
for k, v in data.value_counts(['Cab2']).items(): # NOTE This command returns a tuple key, so just take the first item of tuple.
    map_Cab2[k[0]] = v
data['Cab2Count'] = data.Cab2.map(map_Cab2) # integer
data.Cab2Count.replace(138697, -1, inplace=True)

del data['Cab1'] # NOTE We remove the original feature after creating a frequency feature for it.

# Clustering Cabin numbers; recover missing values to np.nan on the fly.
data.Cab2 = data.Cab2.replace('-1', np.nan) # NOTE Recover NaN values to prepare for imputation
tmp = data['Cab2'].copy().fillna('0').str.rjust(5, '0')
for i in range(5):
    data['Cab2_first'+str(i+1)] = tmp.map(lambda x: np.nan if x=='00000' else x[:i+1]) # string
    tmp_map = dict()
    for k, v in data.value_counts(['Cab2_first'+str(i+1)]).items():
        tmp_map[k[0]] = v # NOTE TO get information from the returned tuple from value_counts.
    data['Cab2_first' + str(i+1) + 'Count'] = data['Cab2_first'+str(i+1)].map(tmp_map)
    del data['Cab2_first'+str(i+1)]
del data['Cab2']

# NAME
data[['Last', 'First']] = data.Name.str.split(', ', expand=True)

map_Last = dict()
for k, v in data.value_counts(['Last']).items():
    map_Last[k[0]] = v
data['LastnameCount'] = data.Last.map(map_Last) # integer

map_First = dict()
for k, v in data.value_counts(['First']).items():
    map_First[k[0]] = v
data['FirstnameCount'] = data.First.map(map_First) # integer
del data['Name']
del data['Last']
del data['First']

# TICKET
new = data.Ticket.str.split(expand=True)
data['Tick1'] = np.where(new[0].str.isnumeric(), np.nan, new[0]) # TECH str.isnumeric
data['misTick1'] = data['Tick1'].isnull().astype(int)

map_Tick1 = dict()
for k, v in data.value_counts(['Tick1']).items():
    map_Tick1[k[0]] = v
data['Tick1Count'] = data.Tick1.map(map_Tick1) # WARN to recount later after imputation of the original feature
# NOTE Unlike the procedure we did on feature CABIN, we put off the np.nan imputation and will do it altogether with other features.

del data['Tick1']

data['Tick2'] = data.Ticket.str.extract('(\d+)') # TECH str.extract
data['misTick2'] = data['Tick2'].isnull().astype(int)
# data = data.assign(misTick2=np.where(data.Tick2.isnull(), 1, 0))

map_Tick2 = dict()
for k, v in data.value_counts(['Tick2']).items(): # NOTE This command returns a tuple key, so just take the first item of tuple.
    map_Tick2[k[0]] = v
data['Tick2Count'] = data.Tick2.map(map_Tick2) 

# Ticket number clustering
# pd.to_numeric(data.Tick2, errors='ignore').max() # to check the maximum number of digits needed.
tmp = data['Tick2'].copy().fillna('0').str.rjust(7, '0')
for i in range(7):
    data['Tick2_first'+str(i+1)] = tmp.map(lambda x: np.nan if x=='0000000' else x[:i+1]) # string
    tmp_map = dict()
    for k, v in data.value_counts(['Tick2_first'+str(i+1)]).items():
        tmp_map[k[0]] = v
    data['Tick2_first'+str(i+1)+'Count'] = data['Tick2_first'+str(i+1)].map(tmp_map)
    del data['Tick2_first'+str(i+1)]    

del data['Tick2']
del data['Ticket']

# Sex
map_sex = {'female': 1, 'male': 0}
data['Sex'] = data['Sex'].map(map_sex) # integer

data['FamilySize'] = data['SibSp'] + data['Parch']

# Embarked
data['Embarked'] = data['Embarked'].fillna('<NA>')
tmp_map = dict()
for k, v in data.value_counts(['Embarked']).items():
    tmp_map[k[0]] = v
data['EmbarkedCount'] = data['Embarked'].map(tmp_map)
data[['EmbarkedCount','Embarked']]
# data.loc[data['Embarked']=='<NA>', 'EmbarkedCount']
data['EmbarkedCount'] = data['EmbarkedCount'].replace(527, -1)

del data['Embarked']

# Age and Fare
# SOURCE dealing with outliers even for tree-based algorithm easier for splitting - https://www.kdnuggets.com/2018/08/make-machine-learning-models-robust-outliers.html

# TODO data.loc[data['Age'].notnull(), 'Age_rg'] = rank_gauss(data.loc[data['Age'].notnull(), 'Age'].values) 
# TODO data.loc[data['Fare'].notnull(), 'Fare_rg'] = rank_gauss(data.loc[data['Fare'].notnull(), 'Fare'].values) 

#DONE!
os.chdir('/kaggle/working')
pickle.dump(data, open('1parsed_data.pkl', 'wb'))
sshColab.upload_to_gcs(project, bucket_name, 'tps-apr-2021-label/1parsed_data.pkl', '/kaggle/working/1parsed_data.pkl')    

#%% 
# ------------------------------------ Age ----------------------------------- #

sshColab.download_to_colab(project, bucket_name, destination_directory = '/kaggle/working', remote_blob_path='tps-apr-2021-label/1parsed_data.pkl', local_file_name='1parsed_data.pkl') 
parsed_data = pickle.load(open('/kaggle/working/1parsed_data.pkl', 'rb'))
data = parsed_data.copy()
data = data.fillna(-1)

the_col = 'Age'
idx_present = data.loc[data[the_col]!=-1, :].index.tolist()
idx_missing = data.loc[data[the_col]==-1, :].index.tolist()
data[the_col] = data[the_col].replace(-1, np.nan)
x_train_full = data.loc[idx_present, :].drop(the_col, axis=1)
y_train_full = data.loc[idx_present, :][the_col]
x_test_full = data.loc[idx_missing, :].drop(the_col, axis=1)

cat_features = [col for col in x_train_full.columns if col not in ['Age', 'Fare']]
x_train_full[cat_features] = x_train_full[cat_features].astype(int) # WARN categorical features can only be integer or string format.
x_test_full[cat_features] = x_test_full[cat_features].astype(int)


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('Time taken : %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


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

N_REPEATS = 1
N_SPLITS = 2
N_TRIALS = 200
TIMEOUT = 2 * 60 * 60
params = {
    'objective': 'reg:squarederror',
    'n_estimators': 1000,
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

# SOURCE retrieve the best model - https://stackoverflow.com/questions/62144904/python-how-to-retrive-the-best-model-from-optuna-lightgbm-study#answer-63365355
def best_model_parameters(study, trial):
    if study.best_trial.number == trial.number:
        # Set the best booster as a trial attribute; accessible via study.trials_dataframe.
        study.set_user_attr(key="best_booster", value=trial.user_attrs["best_booster"])
        # SOURCE retrieve the best number of estimators https://github.com/optuna/optuna/issues/1169

def objective(trial, x_train, y_train, params=params):
    temp_map = {
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.005, 0.05),
        "colsample_bytree": trial.suggest_loguniform("colsample_bytree", 0.2, 0.8),
        "subsample": trial.suggest_loguniform("subsample", 0.4, 0.8),
        "alpha": trial.suggest_loguniform("alpha", 0.01, 10.0),
        "lambda": trial.suggest_loguniform("lambda", 1e-8, 10.0),
        "gamma": trial.suggest_loguniform("lambda", 1e-8, 10.0),
        "min_child_weight": trial.suggest_loguniform("min_child_weight", 10, 1000),        
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
            early_stopping_rounds=50, verbose=0,
            callbacks=[pruning_callback]
        )
        tmp = xgb_regressor.predict(X_B)
        rmse.append(mean_squared_error(y_B, tmp, squared=False))
    trial.set_user_attr(key="best_booster", value=xgb_regressor) # NOTE update the best model in the optuna's table.
    return np.mean(rmse) 

study = optuna.create_study(
    direction = "minimize", 
    sampler = optuna.samplers.TPESampler(seed=2021, multivariate=True),
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
)
study.optimize(lambda trial: objective(trial, x_train_full, y_train_full, params), 
                   n_trials=N_TRIALS, timeout=TIMEOUT, n_jobs=1
)
# display params
hp = study.best_params
for key, value in hp.items():
    print(f"{key:>20s} : {value}")
print(f"{'best objective value':>20s} : {study.best_value}")  

    
    




#%%
# CatBoostRegressor
def vanilla_catboost():
    start_time = timer()
    catmodel = CatBoostRegressor(task_type='GPU', od_wait=50)
    x_A, x_B, y_A, y_B = train_test_split(
        x_train_full, y_train_full, test_size=0.2, random_state=5)
    train_dataset = Pool(data=x_A,
                         label=y_A,
                         cat_features=cat_features)
    eval_dataset = Pool(data=x_B,
                        label=y_B,
                        cat_features=cat_features)   
    catmodel.fit(train_dataset, eval_set = eval_dataset, early_stopping_rounds=50, verbose=200)  
    print(catmodel.best_score_) # 15.4267 and takes 1 minutes 10 seconds
    timer(start_time)

# LightGBM
def vanilla_lgbm():
    start_time = timer()    
    cat_feature_idx = [x_A.columns.get_loc(col) for col in cat_features]
    x_A, x_B, y_A, y_B = train_test_split(
        x_train_full, y_train_full, test_size=0.2, random_state=5)
    lgbm_model = LGBMRegressor(cat_feature=cat_feature_idx, metric='rmse') # , device='gpu'
    lgbm_model.fit(x_A, y_A, eval_set=[(x_B, y_B)],
        early_stopping_rounds=50, verbose=200)
    print(lgbm_model.best_score_['valid_0']['rmse']) # 15.2060 and takes only 3 seconds. However, to reach 15.1844, it takes 5 minutes. Not as good as XGBoost. See the next code block.
    timer(start_time) # 

# ANCHOR LightGBM experimental zone
# LGBM is inferior to XGboost in the present case.
# LGBM can reach a level as low as 15.1844, which is still inferior to XGBoost's 15.181x (in 1 minutes).
def optunning_lgbm():    
    def objective(trial):

        num_iterations=trial.suggest_int('num_iterations',100,1500)
        max_depth=trial.suggest_int('max_depth',3,15)
        num_leaves=trial.suggest_int('num_leaves',10,100)
        min_data_in_leaf=trial.suggest_int('min_data_in_leaf',1,100)
        min_sum_hessian_in_leaf=trial.suggest_int('min_sum_hessian_in_leaf',1,200)
        feature_fraction=trial.suggest_uniform('feature_fraction',1e-5,1.0)
        bagging_fraction=trial.suggest_uniform('bagging_fraction',1e-5,1.0)
        bagging_freq=trial.suggest_int('bagging_freq',1,10)
        lambda_l1=trial.suggest_uniform('lambda_l1',1e-5,5.0)
        lambda_l2=trial.suggest_uniform('lambda_l2',1e-5,10)
    
        cat_feature_idx = [x_A.columns.get_loc(col) for col in cat_features]

        lgbm_model=LGBMRegressor(
            num_iterations=num_iterations,
            max_depth=max_depth,
            num_leaves=num_leaves,
            min_data_in_leaf=min_data_in_leaf,
            min_sum_hessian_in_leaf= min_sum_hessian_in_leaf,
            feature_fraction=feature_fraction,
            bagging_fraction=bagging_fraction,
            bagging_freq=bagging_freq,
            lambda_l1=lambda_l1,
            lambda_l2=lambda_l2, 
            early_stopping_rounds=50,
            cat_feature=cat_feature_idx, 
            metric='rmse',
            verbose=-1
        )
        # lgbm_model = LGBMRegressor(cat_feature=cat_feature_idx, metric='rmse') # , device='gpu'
        lgbm_model.fit(x_A, y_A, eval_set=[(x_B, y_B)],early_stopping_rounds=50, verbose=-1)
        return lgbm_model.best_score_['valid_0']['rmse']

    # study = optuna.create_study(direction='minimize',
    #         sampler=optuna.samplers.TPESampler(multivariate=True)) # SOURCE - https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.TPESampler.html
    # study.optimize(objective, n_trials=100)
    # df_results = study.trials_dataframe()



# CONSTRUCTION
# SOURCE - https://machinelearningapplied.com/hyperparameter-search-with-optuna-part-2-xgboost-classification-and-ensembling/
# SOURCE Retrieve the best model - https://stackoverflow.com/questions/62144904/python-how-to-retrive-the-best-model-from-optuna-lightgbm-study#answer-63365355
# SOURCE Retrieve the best model - https://github.com/optuna/optuna/issues/1169
# SOURCE GMM sampler - https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.TPESampler.html
# SOURCE Pruner - https://optuna.readthedocs.io/en/latest/tutorial/10_key_features/005_visualization.html
