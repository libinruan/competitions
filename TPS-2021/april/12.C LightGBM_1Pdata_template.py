#%%
# SECTION DATA

# ANCHOR packages
import subprocess
import pandas as pd
import numpy as np
import random
import os
import pickle
from datetime import datetime
from google.cloud import storage
import sshColab
import gc

from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.inspection import permutation_importance

import optuna

# import catboost as ctb
import lightgbm as lgb
from xgboost import XGBClassifier, XGBRegressor

import graphviz
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter('ignore')
pd.set_option('max_columns', 100)

# ANCHOR COLAB
COLAB = True
if COLAB:
    # import sshColab
    os.chdir('/root/.kaggle/')
    json_file = 'gcs-colab.json'
    subprocess.call(f'chmod 600 /root/.kaggle/{json_file}', shell=True)        
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = f'/root/.kaggle/{json_file}' 
    subprocess.call('echo $GOOGLE_APPLICATION_CREDENTIALS', shell=True)    
else:
    subprocess.call('pip install ssh-Colab', shell=True)    
    subprocess.call('pip install google-colab', shell=True)
    import sshColab

# ANCHOR GCS
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

def current_datetime(tz="America/New_York"):
    import datetime
    import pytz
    utc_now = pytz.utc.localize(datetime.datetime.utcnow())
    pst_now = utc_now.astimezone(pytz.timezone("America/New_York"))
    return pst_now.strftime("%b-%d-%Y-at-%H-%M")
        
def rank_gauss(x):
    N = x.shape[0]
    temp = x.argsort()
    rank_x = temp.argsort() / N
    rank_x -= rank_x.mean()
    rank_x *= 2
    efi_x = erfinv(rank_x)
    efi_x -= efi_x.mean()
    return efi_x

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)    

# from pathlib import Path
# LOG_PATH = Path("./log/")
# LOG_PATH.mkdir(parents=True, exist_ok=True)
# def score_log(df: pd.DataFrame, seed: int, num_fold: int, model_name: str, cv: float):
#     score_dict = {'date': datetime.now(), 'seed': seed, 'fold': num_fold, 'model': model_name, 'cv': cv}
#     # noinspection PyTypeChecker
#     df = pd.concat([df, pd.DataFrame.from_dict([score_dict])])
#     df.to_csv(LOG_PATH / f"model_score_{model_name}.csv", index=False)
#     return df    

# #L GCS upload
# project = "strategic-howl-305522" # (1)
# bucket_name = "gcs-station-168" # (2)          
# storage_client = storage.Client(project=project)
# bucket = storage_client.bucket(bucket_name)
# destination_blob_name = "tps-apr-2021-label/test.pkl" # (3) No prefix slash.
# source_file_name = "/kaggle/working/test.pkl" # (4)
# pickle.dump(all_df, open(source_file_name, 'wb'))
# blob = bucket.blob(destination_blob_name)
# blob.upload_from_filename(source_file_name)

# ANCHOR LOAD RAW DATA
TARGET = 'Survived'
os.chdir('/kaggle/working/')
train_df = pd.read_csv('../input/tabular-playground-series-apr-2021/train.csv')
test_df = pd.read_csv('../input/tabular-playground-series-apr-2021/test.csv')
submission = pd.read_csv('../input/tabular-playground-series-apr-2021/sample_submission.csv')
train_label = train_df['Survived']

#L GCS download
source_blob_name = "tps-apr-2021-label/voting_submission_from_3_best.csv" # (5) No prefix slash
destination_file_name = "/kaggle/working/voting_submission_from_3_best.csv" # (6)
blob = bucket.blob(source_blob_name)
blob.download_to_filename(destination_file_name)
# all_df = pickle.load(open(destination_file_name, 'rb'))  
test_df[TARGET] = pd.read_csv("voting_submission_from_3_best.csv")[TARGET]

all_df = pd.concat([train_df, test_df]).reset_index(drop=True)
train_rows = train_df.shape[0]

#L GCS下传
source_blob_name = "tps-apr-2021-label/14_all_df.pkl" # (5) No prefix slash
destination_file_name = "/kaggle/working/14_all_df.pkl" # (6)
blob = bucket.blob(source_blob_name)
blob.download_to_filename(destination_file_name)
all_df = pickle.load(open(destination_file_name, 'rb'))  


"""
WARN [Bug] LightGBMError: bin size 257 cannot run on GPU - https://github.com/microsoft/LightGBM/issues/3339
"""
# Since we have applied categrical encoding on feature 'Name', we no longer
# treat it as a categorical variable. Why? See the link above.
cat_features = all_df.select_dtypes(exclude='float').columns.tolist()
cat_features = list(set(cat_features).difference(set(['Name', 'Survived'])))
# NOTE we need to tell LGBM which are categorical features by index rather than
# column name.
cat_list = []
for c in cat_features:
    cat_list.append(all_df.columns.get_loc(c))

# !SECTION DATA

# %%
# SECTION BASIC MODEL
MODE = 'DEBUG' # 'DEBUG' 'CV'

if MODE=='DEBUG':
    N_ESTIMATORS = 2000
    N_SPLITS = 2
    N_REPEATS = 1
    SEED = 2021
    EARLY_STOPPING_ROUNDS = 100
    VERBOSE = 100
    N_TRIALS = 2
    TIMEOUT = 1 * 60

"""
NOTE If you want to use GPU, there exists constraints on the number of bins for categorical features.
Check this for detailed solution - https://github.com/microsoft/LightGBM/issues/3339#issuecomment-714485265
"""
params = { # check offical API doc - https://lightgbm.readthedocs.io/en/latest/Parameters.html
    'objective': 'binary', # cross-entropy
    'metric': 'binary_logloss', # 'rmse', 'auc'
    'cat_feature': cat_list,
    'device': 'gpu',       # comment it out if running on CPU 
    'gpu_platform_id': 0,  # comment it out if running on CPU
    'gpu_device_id': 0,    # comment it out if running on CPU 
    'random_state': SEED,
    'bagging_seed': SEED,
    'feature_fraction_seed': SEED,
    'n_estimators': N_ESTIMATORS, # <--- TECH accuracy trio (1)
    # 'class_weight': 'balanced', # NOTE Use either this or in binary case 'scale_pos_weight' will result in poor estimates of individual class probabilities.
    # 'early_stopping_rounds': EARLY_STOPPING_ROUNDS, # <--- TECH accuracy trio (3)
    # 'learning_rate': 0.02, # <--- TECH accuracy trio (2)
    # 'min_child_samples': 150,
    # 'reg_alpha': 3e-5,
    # 'reg_lambda': 9e-2,
    # 'num_leaves': 20,
    # 'max_depth': 16,
    # 'colsample_bytree': 0.8,
    # 'subsample': 0.8, # NOTE similar to feature_fraction. subsample has aliases like bagging and sub_row.
    # 'subsample_freq': 2,
    # 'max_bin': 240,
    # 'max_drop': 50 # by default
}

X = all_df.drop(['Survived'], axis=1).iloc[:train_rows] 
Y = train_label.iloc[:train_rows]

# var_params = { # api doc - https://lightgbm.readthedocs.io/en/latest/Parameters.html#max_depth
#     'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2),
#     'max_depth': trial.suggest_int('max_depth', 6, 127), # default: -1 (no limit)
#     'num_leaves': trial.suggest_int('num_leaves', 31, 255), # default: 31. Total num of leaves in one tree.
#     'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0), # default: 0. lambda_l1.
#     'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0), # default: 0. lambda_l2.
#     'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 0.9), # feature fraction.
#     'min_child_samples': trial.suggest_int('min_child_samples', 1, 300), # min_data_in_leaf.
#     'subsample_freq': trial.suggest_int('subsample_freq', 1, 10), # NOTE definition - With subsample (or bagging_fraction)  you can specify the percentage of rows used per tree building iteration. 
#     'subsample': trial.suggest_float('subsample', 0.3, 0.9), # https://lightgbm.readthedocs.io/en/latest/Parameters.html
#     'max_bin': trial.suggest_int('max_bin', 128, 1024), # default: 255. smaller more power to deal with overfitting
#     'min_data_per_group': trial.suggest_int('min_data_per_group', 50, 200), # default: 100
#     'cat_smooth': trial.suggest_int('cat_smooth', 10, 100),
#     'cat_l2': trial.suggest_int('cat_l2', 1, 20) # L2 regularization in categorical split
# }

X_A, X_B, Y_A, Y_B = train_test_split(
    X, Y, test_size = 0.33, random_state = 42
)

start_time = timer()
pre_model = lgb.LGBMClassifier(**params)
pre_model.fit(
        X_A, Y_A,
        eval_set=[(X_B, Y_B)], # TECH for plotting learning curve. If multiple evaluation datasets or multiple evaluation metrics are provided, then early stopping will use the last in the list. 
        early_stopping_rounds=100, # NOTE early stopping of 10 iterations means if the result doesn’t improve in the next 10 iterations, stop training.
        verbose=500
    )
pre_model.fit(X, Y)    
timer(start_time)

# screenshot : GPU 35 seconds vs CPU 41 seconds
# !SECTION BASIC MODEL


# %%
# SECTION OPTUNA

MODE = 'CV'

if MODE=='DEBUG':
    N_ESTIMATORS = 1000 # 1000-700 FOR XGBOOST or OUT OF MEMORY
    N_SPLITS = 2 # 10
    N_REPEATS = 1
    N_ITERS = 1 # 30
    SEED = 2021
    EARLY_STOPPING_ROUNDS = 1
    VERBOSE = False
    N_TRIALS = 500
    TIMEOUT = 3 * 60 * 60

if MODE=='CV':
    N_ESTIMATORS = 1000 # 1000-700 FOR XGBOOST or OUT OF MEMORY
    N_SPLITS = 5 # 10
    N_REPEATS = 6
    N_ITERS = 1 # 30
    SEED = 2021
    EARLY_STOPPING_ROUNDS = 100
    VERBOSE = False
    N_TRIALS = 500
    TIMEOUT = 4 * 60 * 60

params = { # check offical API doc - https://lightgbm.readthedocs.io/en/latest/Parameters.html
    'objective': 'binary', # cross-entropy
    'metric': 'auc', # 'rmse', 'auc'
    'cat_feature': cat_list,
    'device': 'gpu',       # comment it out if running on CPU 
    'gpu_platform_id': 0,  # comment it out if running on CPU
    'gpu_device_id': 0,    # comment it out if running on CPU 
    'random_state': SEED,
    'bagging_seed': SEED,
    'feature_fraction_seed': SEED,
    'n_estimators': N_ESTIMATORS, # <--- TECH accuracy trio (1)
    # 'class_weight': 'balanced', # NOTE Use either this or in binary case 'scale_pos_weight' will result in poor estimates of individual class probabilities.
    # 'early_stopping_rounds': EARLY_STOPPING_ROUNDS, # <--- TECH accuracy trio (3)
}

def objective(trial, x_train, y_train, params=params):
    # x_train, y_train: ndarray
    # start_time = timer()    
    temp_map = { # api doc - https://lightgbm.readthedocs.io/en/latest/Parameters.html#max_depth
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2),
        'max_depth': trial.suggest_int('max_depth', 6, 127), # default: -1 (no limit)
        # 'num_leaves': trial.suggest_int('num_leaves', 31, 255), # default: 31. Total num of leaves in one tree.
        'num_leaves': trial.suggest_categorical('num_leaves', [31, 63, 127, 255]), # default: 31
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0), # default: 0. lambda_l1.
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0), # default: 0. lambda_l2.
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 0.9), # feature fraction.
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 300), # min_data_in_leaf.
        'subsample_freq': trial.suggest_int('subsample_freq', 1, 10), # NOTE definition - With subsample (or bagging_fraction)  you can specify the percentage of rows used per tree building iteration. 
        'subsample': trial.suggest_float('subsample', 0.3, 0.9), # https://lightgbm.readthedocs.io/en/latest/Parameters.html
        # 'max_bin': trial.suggest_int('max_bin', 128, 1024), # default: 255. smaller more power to deal with overfitting
        'max_bin': trial.suggest_categorical('max_bin', [15, 31, 63, 127, 255]), # default: 255. smaller more power to deal with overfitting
        'min_data_per_group': trial.suggest_int('min_data_per_group', 50, 200), # default: 100
        'cat_smooth': trial.suggest_int('cat_smooth', 10, 100),
        'cat_l2': trial.suggest_int('cat_l2', 1, 20) # L2 regularization in categorical split
    }
    params.update(temp_map)

    y_oof = np.zeros(x_train.shape[0])
    acc_scores = []    
    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "auc") # depends on the choice of eval_metric; "validation_0-logloss"
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    
    for i, (train_index, valid_index) in enumerate(rskf.split(x_train, y_train)):
        X_A, X_B = x_train[train_index, :], x_train[valid_index, :]
        y_A, y_B = y_train[train_index], y_train[valid_index]
        lgbmClassifier = lgb.LGBMClassifier(**params)
        lgbmClassifier.fit(
            X_A, y_A, eval_set=[(X_B, y_B)], 
            early_stopping_rounds=EARLY_STOPPING_ROUNDS, 
            verbose=VERBOSE,
            callbacks=[pruning_callback])
        y_oof[valid_index] = lgbmClassifier.predict(X_B, num_iteration=lgbmClassifier.best_iteration_) 
        acc_scores.append(accuracy_score(y_B, y_oof[valid_index]))
        
    trial.set_user_attr(key="best_booster", value=lgbmClassifier) # NOTE update the best model in the optuna's table.
    res = np.mean(acc_scores) 
    
    # timer(start_time)
    return res 

def save_best(study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="best_booster", value=trial.user_attrs["best_booster"]) # SOURCE retrieve the best number of estimators https://github.com/optuna/optuna/issues/1169    

if MODE=='CV':
    
    X_input = all_df.drop('Survived', axis=1).iloc[:train_rows, :].values
    Y_input = train_label.iloc[:train_rows].values

    study = optuna.create_study(
        direction = "maximize", 
        sampler = optuna.samplers.TPESampler(seed=2021, multivariate=True),
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)
    )
    study.optimize(lambda trial: objective(trial, X_input, Y_input, params), 
                   n_trials=N_TRIALS, 
                   timeout=TIMEOUT, 
                   callbacks=[save_best],
                   n_jobs=1
    )
    hp = study.best_params
    for key, value in hp.items():
        print(f"{key:>20s} : {value}")
    print(f"{'best objective value':>20s} : {study.best_value}")  
    best_model=study.user_attrs["best_booster"]

    #L GCS上传
    curdt = current_datetime()
    project = "strategic-howl-305522" # (1)
    bucket_name = "gcs-station-168" # (2)          
    storage_client = storage.Client(project=project)
    bucket = storage_client.bucket(bucket_name)
    destination_blob_name = f"tps-apr-2021-label/13_1P_LGBM{curdt}.pkl" # (3) No prefix slash.
    source_file_name = f"/kaggle/working/13_1P_LGBM{curdt}.pkl" # (4)
    pickle.dump(best_model, open(source_file_name, 'wb'))
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)

# !SECTION OPTUNA

# SECTION Inference
source_blob_name = f"tps-apr-2021-label/13_1P_LGBMMay-09-2021-at-12-41.pkl" # (5) No prefix slash
destination_file_name = f"/kaggle/working/13_1P_LGBMMay-09-2021-at-12-41.pkl" # (6)
blob = bucket.blob(source_blob_name)
blob.download_to_filename(destination_file_name)
best_model = pickle.load(open(destination_file_name, 'rb'))  

MODE = 'CV' # 'DEBUG' 'CV', 'PROD'

if MODE=='CV':
    N_ESTIMATORS = 1000
    N_SPLITS = 5 # 3, 5, 10
    N_REPEATS = 6 # 3, 5, 10, 30
    SEED = 2021
    EARLY_STOPPING_ROUNDS = 100
    VERBOSE = 100
    N_TRIALS = 500
    TIMEOUT = 3 * 60 * 60
    TEST_SIZE = 0.1

def lgb_inference(params, X, Y, X_test):
    """
    KFold inference performs better in leaderboard, not in local CV. Be mindful!
    Use this not the one after this code block.
    """
    x = X.values
    y = Y.values
    x_tst = X_test.values
    y_tst = np.zeros((x_tst.shape[0], len(np.unique(y))))
    acc_scores = []
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, 
                                   random_state=SEED)
    for i, (train_index, valid_index) in enumerate(rskf.split(x, y)):
        X_A, X_B = x[train_index, :], x[valid_index, :]
        y_A, y_B = y[train_index], y[valid_index]
        print(i)
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_A, y_A, eval_set=[(X_B, y_B)],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose=0)
        acc_score = accuracy_score(y_B, model.predict(X_B, num_iteration=model.best_iteration_))
        acc_scores.append(acc_score)
        y_tst += model.predict_proba(x_tst, num_iteration=model.best_iteration_)
    y_tst /= N_SPLITS * N_REPEATS
    return y_tst, np.mean(acc_scores)    

X = all_df.drop('Survived', axis=1).iloc[:train_rows, :]
Y = train_label.iloc[:train_rows]
X_test = all_df.drop('Survived', axis=1).iloc[train_rows:, :]
Y_tst, mean_acc = lgb_inference(best_model.get_params(), X, Y, X_test)
submission.Survived = np.where(Y_tst[:,-1] > 0.5, 1, 0)
submission.to_csv(f"/kaggle/working/submission_cv_lgb_{mean_acc:.6f}_{curdt}.csv", index=False)
print(f"/kaggle/working/submission_cv_lgb_{mean_acc:.6f}_{curdt}.csv, {mean_acc}")

#L GCS上传
# curdt = current_datetime()
project = "strategic-howl-305522" # (1)
bucket_name = "gcs-station-168" # (2)          
storage_client = storage.Client(project=project)
bucket = storage_client.bucket(bucket_name)
destination_blob_name = f"tps-apr-2021-label/submission_cv_lgb_{mean_acc:.6f}_{curdt}.csv" # (3) No prefix slash.
source_file_name = f"/kaggle/working/submission_cv_lgb_{mean_acc:.6f}_{curdt}.csv" # (4)
# pickle.dump(best_model, open(source_file_name, 'wb'))
blob = bucket.blob(destination_blob_name)
blob.upload_from_filename(source_file_name)
# !SECTION inference 
# %%
