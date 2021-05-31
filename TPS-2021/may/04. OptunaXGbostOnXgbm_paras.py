#%%
"""
class weighted condition. optimal hyperparameter.
CV: 1.10079
PB: 1.09666
"""

# from pytorch_tabnet.tab_model import TabNetClassifier
import torch

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import gc

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold
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

import optuna
from xgboost import XGBClassifier
import xgboost as xgb

from tqdm.notebook import tqdm
from IPython.display import Image , display

import warnings
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)


# Time Functions
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


# GCS API and Credentials        
from google.cloud import storage

project = "strategic-howl-305522"
bucket_name = "gcs-station-168"           
storage_client = storage.Client(project=project)
bucket = storage_client.bucket(bucket_name)

train = pd.read_csv("/kaggle/input/tabular-playground-series-may-2021/train.csv", index_col = 'id')
test = pd.read_csv("/kaggle/input/tabular-playground-series-may-2021/test.csv", index_col = 'id')
raw_features = test.columns.tolist()

# 

N_SPLITS = 7
N_REPEATS = 3
EARLY_STOPPING_ROUNDS = 10
RANDOM_SEED = 2021 
N_TRIALS = 300
TIMEOUT = 4 * 60 * 60 

# ANCHOR 
# To assign all out of bounds test value an identical value
def objective(trial, train, test, raw_features):    
    # start_time = timer()
    CLIP_FEATURES = False # trial.suggest_categorical("clip", ["True", "False"])

    df_all_X = pd.concat([train.drop('target', axis=1), test], axis=0)
    if CLIP_FEATURES == False:
        le = LabelEncoder()
        df_all_X = df_all_X.apply(le.fit_transform)
    else:    
        missing_integer = df_all_X.max().max() + 15 # we will replace any unseen test value with this one.
        for col in test.columns: # inspect across all test set's columns
            if not all(test[col].isin(train[col].value_counts().index.tolist())): # see if there's out of bounds value
                for i in set(test[col]).difference(set(train[col])):
                    test[col].replace(i, missing_integer, inplace=True) # replace the oob value with a dummy value
                train[f'{col}_{missing_integer}'] = 0 # generate a boolean column to mark all those missing values in test set
                test[f'{col}_{missing_integer}'] = 1 # generate a boolean column to mark all those missing values in test set
                test[f'{col}_{missing_integer}'].where(test[col]==missing_integer, 0, inplace=True) 
        df_all_X = pd.concat([train.drop('target', axis=1), test], axis=0)

    # Gropu low frequency into one value            
    GROUP_LOW_FREQUENCY = False
    GROUP_LOW_FREQUENCY_THRESHOLD = 0 # trial.suggest_discrete_uniform("threshold", 0, 50, 1)

    if GROUP_LOW_FREQUENCY:
        for col in raw_features:
            value_counts_SS = df_all_X[col].value_counts()
            low_freq_values = value_counts_SS.index[value_counts_SS < GROUP_LOW_FREQUENCY_THRESHOLD]
            if len(low_freq_values) > 0:
                df_all_X[f'{col}_low_freq'] = 0
                for i in low_freq_values.tolist():
                    df_all_X[f'{col}_low_freq'].iloc[df_all_X[col]==i] = 1

    Xtrn, Xtst = df_all_X.iloc[:len(train)], df_all_X.iloc[len(train):]
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(train['target']))       

    # ANCHOR CONSTRUCTION
    class0 = 0.8023844111083878
    class1 = 0.4973760913650416
    class2 = 0.8940055025348296
    class3 = 0.8641162667601383   

    class_weights = [class0, class1, class2, class3]
    losses = []
    y_oof = np.zeros((Xtrn.shape[0], len(np.unique(y))))
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "val-mlogloss")
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_SEED)

    temp_map = {
            # "learning_rate": trial.suggest_loguniform("learning_rate", 0.005, 0.05),
            "colsample_bytree": trial.suggest_loguniform("colsample_bytree", 0.1, 0.8),
            "subsample": trial.suggest_loguniform("subsample", 0.1, 0.8),
            "alpha": trial.suggest_loguniform("alpha", 0.01, 10.0),
            "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
            "gamma": trial.suggest_loguniform("lambda", 1e-8, 1.0),
            "min_child_weight": trial.suggest_loguniform("min_child_weight", 3, 100),        
            'max_depth': trial.suggest_int('max_depth', 3, 12)
        }

# ANCHOR CONSTRUCTION

# from sklearn.model_selection import train_test_split
# le = LabelEncoder()
# y = le.fit_transform(train['target'])
# X_A, X_B, y_A, y_B = train_test_split(train.drop('target', axis=1), y, test_size=0.33, random_state=42)
# dtrain = xgb.DMatrix(X_A, label=y_A)
# dtest = xgb.DMatrix(X_B, label=y_B)
# params = {
#     'objective': "multi:softprob",
#     'eval_metric': 'mlogloss',
#     'n_estimators': 10000,
#     'booster': 'gbtree',   
#     'tree_method': 'gpu_hist',     
#     'num_class': 4
# }
# xgb_model = xgb.train(params, 
#     dtrain=dtrain, 
#     evals=[(dtest, 'val'), (dtrain, 'train')], 
#     verbose_eval=False)
# tmp = xgb_model.predict(xgb.DMatrix(X_B))

    for i, (train_index, valid_index) in enumerate(rskf.split(Xtrn, y)):
        X_A, X_B = Xtrn.iloc[train_index, :], Xtrn.iloc[valid_index, :]
        y_A, y_B = y.iloc[train_index], y.iloc[valid_index]    
        # sample_weight_fold = [class_weights[j] for j in y_A]
        params = {
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'n_estimators': 10000,
            'booster': 'gbtree',
            'verbosity': 0,
            'tree_method': 'gpu_hist',
            'num_class': 4
        }
        dtrain = xgb.DMatrix(X_A, label=y_A, weight=[class_weights[j] for j in y_A])
        dtest = xgb.DMatrix(X_B, label=y_B) #, weight=[class_weights[j] for j in y_B])
        dtestX = xgb.DMatrix(X_B)
        params.update(temp_map)
        # learning api https://tinyurl.com/yz8bqyfd
        xgb_model = xgb.train(params, 
            dtrain=dtrain, 
            evals=[(dtest, 'val'), (dtrain, 'train')], 
        #     sample_weight=sample_weight_fold,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            callbacks=[pruning_callback],            
            verbose_eval=False)
        # xgb_classifier = XGBClassifier(**params)
        # xgb_classifier.fit(
        #     X_A, y_A, eval_set=[(X_B, y_B)],
        #     sample_weight=sample_weight_fold,
        #     early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        #     callbacks=[pruning_callback]
        # )
        # tmp = xgb_classifier.predict_proba(X_B)
        tmp = xgb_model.predict(dtestX)
        y_oof[valid_index, :] = tmp / N_REPEATS
        loss = log_loss(y_B, tmp)
        losses.append(loss)
        # print(f'loss: {loss}')
    mean_running_loss = np.mean(losses)
    # print(f'average running loss: {mean_running_loss}')
    # oof_loss = log_loss(y, y_oof)
    # print(f'average repeat oof loss: {oof_loss}')
    # trial.set_user_attr(key="best_booster", value=xgb_model) 
    # timer(start_time)
    xgb_model.__del__() # release memory https://tinyurl.com/ydw9nebm

    return mean_running_loss


def save_best(study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="best_booster", value=trial.user_attrs["best_booster"])

study = optuna.create_study(
    direction = "minimize", 
    sampler = optuna.samplers.TPESampler(seed=2021, multivariate=True),
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=3))

study.optimize(lambda trial: objective(trial, train, test, raw_features), 
               n_trials=N_TRIALS, 
               timeout=TIMEOUT, 
            #    callbacks=[save_best],
               n_jobs=1)

# best iteration : https://tinyurl.com/ygb6kftr
hp = study.best_params
for key, value in hp.items():
    print(f"{key:>20s} : {value}")
print(f"{'best objective value':>20s} : {study.best_value}")  
# best_model=study.user_attrs["best_booster"]  


# %%
# ANCHOR prediction based on optuna optimal parameter    
# best objective value : 1.1458917625733058         
optimal_params = {
    "colsample_bytree" : 0.6724510803617302,
           "subsample" : 0.7804697500290203,
               "alpha" : 0.01592181571979387,
              "lambda" : 2.6312462651552615e-08,
    "min_child_weight" : 26.682631038211774,
           "max_depth" : 12
}
params = {
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'n_estimators': 10000,
    'booster': 'gbtree',
    'verbosity': 0,
    'tree_method': 'gpu_hist',
    'num_class': 4
}
params.update(optimal_params)
df_all_X = pd.concat([train.drop('target', axis=1), test], axis=0)
le = LabelEncoder()
df_all_X = df_all_X.apply(le.fit_transform)
Xtrn, Xtst = df_all_X.iloc[:len(train)], df_all_X.iloc[len(train):]
le = LabelEncoder()
y = pd.Series(le.fit_transform(train['target'])) 
losses = []
y_oof = np.zeros((Xtst.shape[0], len(np.unique(y))))
y_val = np.zeros((Xtrn.shape[0], len(np.unique(y))))

N_SPLITS = 7
N_REPEATS = 3
EARLY_STOPPING_ROUNDS = 10
RANDOM_SEED = 2021 

rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_SEED)

dtestX = xgb.DMatrix(Xtst)
for i, (train_index, valid_index) in enumerate(rskf.split(Xtrn, y)):
    X_A, X_B = Xtrn.iloc[train_index, :], Xtrn.iloc[valid_index, :]
    y_A, y_B = y.iloc[train_index], y.iloc[valid_index]    
    dtrain = xgb.DMatrix(X_A, label=y_A)
    dval = xgb.DMatrix(X_B, label=y_B) #, weight=[class_weights[j] for j in y_B])
    dvalX = xgb.DMatrix(X_B)
    # learning api https://tinyurl.com/yz8bqyfd
    xgb_model = xgb.train(params, 
        dtrain=dtrain, 
        evals=[(dval, 'val'), (dtrain, 'train')], 
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,      
        verbose_eval=False)
    # xgb_classifier = XGBClassifier(**params)
    # xgb_classifier.fit(
    #     X_A, y_A, eval_set=[(X_B, y_B)],
    #     sample_weight=sample_weight_fold,
    #     early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    #     callbacks=[pruning_callback]
    # )
    # tmp = xgb_classifier.predict_proba(X_B)
    tmp = xgb_model.predict(dvalX)
    y_val[valid_index, :] += tmp / N_REPEATS
    loss = log_loss(y_B, tmp)
    losses.append(loss)
    y_oof +=  xgb_model.predict(dtestX) / (N_REPEATS * N_SPLITS)

log_loss(y, y_val)

sub = pd.read_csv("/kaggle/input/tabular-playground-series-may-2021/sample_submission.csv")

predictions_df = pd.DataFrame(y_oof, columns = ["Class_1", "Class_2", "Class_3", "Class_4"])
predictions_df['id'] = sub['id']
csv_filename = f"/kaggle/working/may_model/{dtnow()}-TPS-05-xgbm-solo-weighted_submission.csv"
predictions_df.to_csv(csv_filename, index = False)


# %%
base_class = y
preds = np.argmax(y_val, axis = 1)
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
sns.heatmap(pd.DataFrame(confusion_matrix(base_class, preds)), annot=True, linewidths=.5, fmt="d");
# print("f1: {:0.6f} log loss: {:0.6f}".format(f1_score(base_class, preds, average='macro'), log_loss(y_valid, y_valud_preds_df)))

# screenshot https://i.postimg.cc/sg5VsvV6/2021-05-29-at-20-09-30.png
# NOTE CONCLUSION
# Class weighting is not helpful.

# %%
