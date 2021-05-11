#%%
# ANCHOR packages
import os, subprocess, gc
from google.cloud import storage
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.special import erfinv
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

import optuna

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from scipy import sparse as sp
from tqdm import tqdm

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
    subprocess.call('pip install ssh-Colab', shell=True)    
    subprocess.call('pip install google-colab', shell=True)
    import sshColab

project = "strategic-howl-305522"
bucket_name = "gcs-station-168"           
storage_client = storage.Client(project=project)
bucket = storage_client.bucket(bucket_name)

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

train_df = pd.read_csv('/kaggle/input/tabular-playground-series-apr-2021/train.csv')
test_df = pd.read_csv('/kaggle/input/tabular-playground-series-apr-2021/test.csv')

train_label = train_df['Survived']
train_id = train_df['PassengerId']
test_id = test_df['PassengerId']
del train_df['Survived'], train_df['PassengerId']
del test_df['PassengerId']

train_rows = train_df.shape[0]


# %%
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
    remote_blob_path='tps-apr-2021-label/6imputed_df.pkl', 
    local_file_name='6imputed_df.pkl') 
imputed_df = pickle.load(open('/kaggle/working/6imputed_df.pkl', 'rb'))

cols_to_drop = ['Age', 'Fare', 'Embarked_enc'] + \
    [col for col in data.columns.tolist() if col.startswith('Cab')] + \
    [col for col in data.columns.tolist() if col.startswith('Tick')]   

df = data.drop(cols_to_drop, axis=1).join(imputed_df)
df['Embarked_enc_imp'] = df['Embarked_enc_imp'].astype(int)

df = df.rename(lambda x: x.replace('_imp', ''), axis=1)
df = df.rename(lambda x: x.replace('_enc', ''), axis=1)

# Alternatively
# df.columns = list(map(lambda x: x.replace('_imp', ''), df.columns.tolist()))
# df.columns = list(map(lambda x: x.replace('_enc', ''), df.columns.tolist()))

df['Age'] = rank_gauss(df['Age'].values)
df['Fare'] = rank_gauss(df['Fare'].values)

del imputed_df
del data
gc.collect()

# train_label, train_rows, df
floatCols = df.select_dtypes(include='float').columns.tolist()
classCols = df.select_dtypes(include=['int8', 'int64']).columns.tolist()

from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score


#%% Toy model passed test.
# ANCHOR toy model
# params = {
#     'objective': 'binary:logistic',
#     # 'num_class': 2, # when you use binary classification, comment this out.
#     'n_estimators': 700,
#     'booster': 'gbtree',
#     'verbosity': 0,
#     'average': 'macro',
#     # 'eval_metric': 'auc',
#     # 'tree_method': 'gpu_hist',
#     'use_label_encoder': False # <-------        
# }
# start_time = timer()
# x_train = df.iloc[:train_rows, :].values
# y_train = train_label.iloc[:train_rows].values
# y_oof = np.zeros((x_train.shape[0],2))
# acc_scores = []
# xgb_classifier = XGBClassifier(**params)
# rkf = RepeatedStratifiedKFold(n_splits=2, n_repeats=1)
# for i, (train_index, valid_index) in enumerate(rkf.split(x_train, y_train)):
#     X_A, X_B = x_train[train_index, :], x_train[valid_index, :]
#     y_A, y_B = y_train[train_index], y_train[valid_index]
#     xgb_classifier.fit(
#         X_A, y_A, eval_set=[(X_B, y_B)], early_stopping_rounds=50
#     )
#     y_oof[valid_index] = xgb_classifier.predict_proba(X_B)
#     acc_score = accuracy_score(y_B, np.where(y_oof[valid_index]>0.5, 1, 0).argmax(axis=1))
#     acc_scores.append(acc_score)
#     print(f"===== {acc_score} =====\n")
# print(np.mean(acc_scores))
# timer(start_time)

#%%
N_SPLITS = 5
N_REPEATS = 1
EARLY_STOPPING_ROUNDS = 100
RANDOM_SEED = 2021
N_TRIALS = 1500
TIMEOUT = 3 * 60 * 60 # <------------

params = {
    'objective': 'binary:logistic', # NOTE don't specify "n_classes" in binary:logistic case.
    'n_estimators': 800,
    'booster': 'gbtree',
    'verbosity': 0,
    'average': 'macro',
    'eval_metric': 'auc', 
    'tree_method': 'gpu_hist', 
    'use_label_encoder': False # <------- do it yourself beforehand to aovid warning message. See label encoding in "10.D GCS_XGboost_imputation_AgeFareEmbarked.py."       
}

def objective(trial, x_train, y_train, params=params):
    start_time = timer()
    temp_map = {
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.005, 0.05),
        "min_child_weight": trial.suggest_loguniform("min_child_weight", 5, 1000),
        "subsample": trial.suggest_loguniform("subsample", 0.4, 0.8),
        "colsample_bytree": trial.suggest_loguniform("colsample_bytree", 0.2, 0.8),
        "alpha": trial.suggest_loguniform("alpha", 0.01, 10.0),
        "lambda": trial.suggest_loguniform("lambda", 1e-8, 10.0),
        "gamma": trial.suggest_loguniform("gamma", 1e-8, 10.0)
    }
    params.update(temp_map)
    # x_train = df.iloc[:train_rows, :].values
    # y_train = train_label.iloc[:train_rows].values
    y_oof = np.zeros((x_train.shape[0]))
    acc_scores = []    
    # pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation_0-logloss")
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation_0-auc")
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_SEED)
    for i, (train_index, valid_index) in enumerate(rskf.split(x_train, y_train)):
        X_A, X_B = x_train[train_index, :], x_train[valid_index, :]
        y_A, y_B = y_train[train_index], y_train[valid_index]
        xgb_classifier = XGBClassifier(**params)
        xgb_classifier.fit(
            X_A, y_A, eval_set=[(X_B, y_B)], 
            early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=0,
            callbacks=[pruning_callback]
        )
        best_iteration = xgb_classifier.get_booster().best_ntree_limit               # new
        y_oof[valid_index] = xgb_classifier.predict(X_B, ntree_limit=best_iteration) # new iteration_range=[0,best_iteration]      
        acc_score = accuracy_score(y_B, y_oof[valid_index])
        acc_scores.append(acc_score)
        # print(f"===== {i} fold : acc {acc_score} =====")
    trial.set_user_attr(key="best_booster", value=xgb_classifier) # NOTE update the best model in the optuna's table.
    res = np.mean(acc_scores) 
    # print(f"===== {res} =====")
    timer(start_time)
    return res 

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
study.optimize(lambda trial: objective(trial, 
                                       df.iloc[:train_rows, :].values, 
                                       train_label.iloc[:train_rows].values, params), 
               n_trials=N_TRIALS, 
               timeout=TIMEOUT, 
               n_jobs=1, 
               callbacks=[save_best]
)

hp = study.best_params
for key, value in hp.items():
    print(f"{key:>20s} : {value}")
print(f"{'best objective value':>20s} : {study.best_value}")  
best_model=study.user_attrs["best_booster"]

file = '9xgboost1500itModel_with_ntree.pkl'
os.chdir('/kaggle/working')
pickle.dump(best_model, open(f'/kaggle/working/{file}', 'wb'))
sshColab.upload_to_gcs(project, bucket_name, f'tps-apr-2021-label/{file}', f'/kaggle/working/{file}')

# ANCHOR Screenshot . plot feature importance - https://i.postimg.cc/SNmz7GLr/2021-04-28-at-22-58-26.png
from xgboost import plot_importance
# method 1
plot_importance(best_model)
# method 2
feature_important = best_model.get_booster().get_score(importance_type='weight')
tmp = pd.Series(list(feature_important.keys())).map(lambda x: x[1:]).astype(int).tolist()
cols = [df.columns.tolist()[i] for i in tmp]
keys = cols
values = list(feature_important.values())
edata = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=True)
edata.plot(kind='barh')


#%%

# ANCHOR load trained xgboost model
file_to_load = '9xgboost1500itModel.pkl'
sshColab.download_to_colab(project, bucket_name, 
    destination_directory = '/kaggle/working', 
    remote_blob_path=f'tps-apr-2021-label/{file_to_load}', 
    local_file_name=file_to_load) 
xgb_model = pickle.load(open(f'/kaggle/working/{file_to_load}', 'rb'))

# SECTION inference
source_blob_name = f"tps-apr-2021-label/9xgboost1500itModel_May-09-2021-at-22-11_with_ntree.pkl" # (5) No prefix slash
destination_file_name = f"/kaggle/working/9xgboost1500itModel_May-09-2021-at-22-11_with_ntree.pkl" # (6)
blob = bucket.blob(source_blob_name)
blob.download_to_filename(destination_file_name)
best_model = pickle.load(open(destination_file_name, 'rb'))  


MODE = 'CV' # 'DEBUG' 'CV', 'PROD'

if MODE=='DEBUG':
    N_ESTIMATORS = 1
    N_SPLITS = 2
    N_REPEATS = 2
    SEED = 2021
    EARLY_STOPPING_ROUNDS = 1
    VERBOSE = 100
    TIMEOUT = 1 * 60

if MODE=='CV':
    N_ESTIMATORS = 1000
    N_SPLITS = 5 #5 3, 5, 10
    N_REPEATS = 2 #6 3, 5, 10, 30
    SEED = 2021
    EARLY_STOPPING_ROUNDS = 100
    VERBOSE = 100
    N_TRIALS = 1000
    TIMEOUT = 3 * 60 * 60

def xgb_inference(model, X, Y, X_test):
    x = X.values
    y = Y.values
    x_tst = X_test.values
    x_tst = np.ascontiguousarray(x_tst)
    # y_oof = np.zeros(x.shape[0])
    y_tst = np.zeros((x_tst.shape[0], len(np.unique(y))))
    acc_scores = []
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, 
                                   n_repeats=N_REPEATS, 
                                   random_state=SEED)
    params = model.get_params()
    for i, (train_index, valid_index) in enumerate(rskf.split(x, y)):
        print(i)
        X_A, X_B = x[train_index, :], x[valid_index, :]
        y_A, y_B = y[train_index], y[valid_index]
        xgb_model = XGBClassifier(**params)
        X_A, X_B = np.ascontiguousarray(X_A), np.ascontiguousarray(X_B)
        y_A, y_B = np.ascontiguousarray(y_A), np.ascontiguousarray(y_B)
        xgb_model.fit(
            X_A, y_A, eval_set=[(X_B, y_B)],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose=0)
        best_iteration = xgb_model.get_booster().best_ntree_limit               # new
        # y_oof[valid_index] = xgb_model.predict(X_B, ntree_limit=best_iteration) # new         
        tmp = xgb_model.predict(X_B, iteration_range=[0,best_iteration])
        acc_score = accuracy_score(y_B, tmp)
        acc_scores.append(acc_score)
        y_tst += model.predict_proba(x_tst, iteration_range=[0,best_iteration])
    y_tst /= N_SPLITS * N_REPEATS
    return y_tst, np.mean(acc_scores)    

X = df.iloc[:train_rows, :]
Y = train_label.iloc[:train_rows]
X_test = df.iloc[train_rows:, :]
Y_tst, mean_acc = xgb_inference(best_model, X, Y, X_test)
submission.Survived = np.where(Y_tst[:,-1] > 0.5, 1, 0)
submission.to_csv(f"/kaggle/working/my_xgb_submission_cv{mean_acc:.6f}_May-09-2021-at-22-11_with_ntree.csv", index=False)



# !SECTION inference




# ------------------------ training with pseudo label ------------------------ #
# SECTION training with pseudo label
file_to_load = '10-tps-apr-simple-ensemble-submission.csv'
sshColab.download_to_colab(project, bucket_name, 
    destination_directory = '/kaggle/working', 
    remote_blob_path=f'tps-apr-2021-label/{file_to_load}', 
    local_file_name=file_to_load) 
test_pseudo_label = pd.read_csv((f'/kaggle/working/{file_to_load}'))

# ANCHOR merge pseudo label into data frame
TARGET = 'Survived'
df.loc[train_rows:, TARGET] = test_pseudo_label.loc[:, TARGET].values # NOTE IF you encounter "only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices", convert the RHS into np.numpy by "".values".
df.loc[:train_rows-1, TARGET] = train_label.values

target_df = df[TARGET].copy()
features_df = df.drop(TARGET, axis=1)

# The block below is exactly the same as the preceeding block except the input
# arguments would target_df and features_df.

TARGET = 'Survived'
# ANCHOR pseudo label
df.loc[train_rows:, TARGET] = test_pseudo_label.loc[:, TARGET].values # NOTE IF you encounter "only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices", convert the RHS into np.numpy by "".values".
df.loc[:train_rows-1, TARGET] = train_label.values

target_df = df[TARGET].copy()
features_df = df.drop(TARGET, axis=1)

# #%% ANCHOR grouping columns
# def check_col_nunique(df):
#     tab = dict()
#     for col in df.columns:
#         tab[col] = df[col].nunique()
#     return pd.Series(tab)
# CARDINALITY_THRESHOLD = 10
# cardinality_ss = check_col_nunique(df.drop(TARGET, axis=1))
# mask = np.array(cardinality_ss) > CARDINALITY_THRESHOLD
# label_cols = [col for col in cardinality_ss[mask].index.tolist() if df[col].dtype != 'float']
# onehot_cols = [col for col in df.columns.tolist() if col not in label_cols and df[col].dtype != 'float']
# float_cols = [col for col in df.columns.tolist() if df[col].dtype == 'float' and not col in ['Survived']]


N_SPLITS = 5
N_REPEATS = 1
EARLY_STOPPING_ROUNDS = 50
N_TRIALS = 1500
TIMEOUT = 3 * 60 * 60 # <------------

params = {
    'objective': 'binary:logistic', # NOTE don't specify "n_classes" in binary:logistic case.
    'n_estimators': 700,
    'booster': 'gbtree',
    'verbosity': 0,
    'average': 'macro',
    'eval_metric': 'auc', 
    'tree_method': 'gpu_hist', 
    'use_label_encoder': False # <-------        
}

def objective(trial, x_train, y_train, params=params):
    start_time = timer()
    temp_map = {
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.005, 0.05),
        "min_child_weight": trial.suggest_loguniform("min_child_weight", 50, 1000),
        "subsample": trial.suggest_loguniform("subsample", 0.4, 0.8),
        "colsample_bytree": trial.suggest_loguniform("colsample_bytree", 0.2, 0.8),
        "alpha": trial.suggest_loguniform("alpha", 0.01, 10.0),
        "lambda": trial.suggest_loguniform("lambda", 1e-8, 10.0),
        "gamma": trial.suggest_loguniform("lambda", 1e-8, 10.0)
    }
    params.update(temp_map)
    # x_train = df.iloc[:train_rows, :].values
    # y_train = train_label.iloc[:train_rows].values
    y_oof = np.zeros((x_train.shape[0],2))
    acc_scores = []    
    xgb_classifier = XGBClassifier(**params)
    # pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation_0-logloss")
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation_0-auc")
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS)
    for i, (train_index, valid_index) in enumerate(rskf.split(x_train, y_train)):
        X_A, X_B = x_train[train_index, :], x_train[valid_index, :]
        y_A, y_B = y_train[train_index], y_train[valid_index]
        xgb_classifier.fit(
            X_A, y_A, eval_set=[(X_B, y_B)], 
            early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=0,
            callbacks=[pruning_callback]
        )
        y_oof[valid_index] = xgb_classifier.predict_proba(X_B)
        acc_score = accuracy_score(y_B, np.where(y_oof[valid_index]>0.5, 1, 0).argmax(axis=1))
        acc_scores.append(acc_score)
        print(f"===== {i} fold : acc {acc_score} =====")
    trial.set_user_attr(key="best_booster", value=xgb_classifier) # NOTE update the best model in the optuna's table.
    res = np.mean(acc_scores) 
    print(f"***** {res} *****")
    timer(start_time)
    return res 

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
study.optimize(lambda trial: objective(trial, 
                                       features_df.values, 
                                       target_df.values, params), 
               n_trials=N_TRIALS, 
               timeout=TIMEOUT, 
               n_jobs=1, 
               callbacks=[save_best]
)

hp = study.best_params
for key, value in hp.items():
    print(f"{key:>20s} : {value}")
print(f"{'best objective value':>20s} : {study.best_value}")  
best_model=study.user_attrs["best_booster"]

file = '10xgboost1500itModelPseudoLabeling.pkl'
os.chdir('/kaggle/working')
pickle.dump(best_model, open(f'/kaggle/working/{file}', 'wb'))
sshColab.upload_to_gcs(project, bucket_name, f'tps-apr-2021-label/{file}', f'/kaggle/working/{file}')

# %%

#            max_depth : 11
#        learning_rate : 0.022689456527447947
#     min_child_weight : 50.420222266881
#            subsample : 0.7305219821323933
#     colsample_bytree : 0.7925388902164281
#                alpha : 0.03796124616238057
#               lambda : 8.592501699788991e-08
# best objective value : 0.88709

# ANCHOR Resuem Optuna Study example in the current case
study.optimize(lambda trial: objective(trial, 
                                       features_df.values, 
                                       target_df.values, params), 
               n_trials=N_TRIALS, 
               timeout=TIMEOUT, 
               n_jobs=1, 
               callbacks=[save_best]
)
# !SECTION training with pseudo label