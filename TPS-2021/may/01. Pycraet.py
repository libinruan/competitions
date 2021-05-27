
# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import pickle
import datetime
import pytz

import seaborn as sns
import plotly.express as px
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import confusion_matrix

import optuna
import lightgbm as lgb
from xgboost import XGBClassifier, XGBRegressor
from scipy.special import erfinv
import warnings


warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option('display.max_columns', None)

def seed_everything(seed=2021):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

seed_everything()

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

# %%

os.chdir('/kaggle/working')
train = pd.read_csv('../input/tabular-playground-series-may-2021/train.csv')
test = pd.read_csv('../input/tabular-playground-series-may-2021/test.csv')
sample_submission = pd.read_csv('../input/tabular-playground-series-may-2021/sample_submission.csv')

train = train.drop(['id'], axis=1)
test = test.drop(['id'], axis=1)

train['target'] = LabelEncoder().fit_transform(train['target'])



# %% ANCHOR rankgauss

def rank_gauss(x):
    N = x.shape[0]
    temp = x.argsort()
    rank_x = temp.argsort() / N
    rank_x -= rank_x.mean()
    rank_x *= 2
    efi_x = erfinv(rank_x)
    efi_x -= efi_x.mean()
    return efi_x

def get_train_test_split_on_rankgauss_data(train, test):
    X = pd.concat([train.drop('target', axis=1), test])
    X = X.apply(rank_gauss)
    y = train.target
    
    return y, X.iloc[:len(train), :], X.iloc[len(train):, :]

y, Xtrn, Xtst = get_train_test_split_on_rankgauss_data(train, test)

# %% ANCHOR 1. rankgauss debug

def rankgauss_debug():

    X_train, X_val, y_train, y_val = train_test_split(Xtrn, y, test_size=0.2, random_state=2021)

    lgb_params = {
        'objective': 'multiclass',
        'num_class': 4,
        'boosting': 'gbdt',
        'metric': 'multi_logloss'
    }
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_val, y_val, reference=lgb_train)
    model = lgb.train(
        lgb_params, 
        lgb_train, 
        valid_sets = [lgb_train, lgb_valid],
        valid_names = ['train', 'eval'],
        num_boost_round = 1000,
        verbose_eval = 25, # https://tinyurl.com/yhdmtdm8
        early_stopping_rounds = 20
    )
    print(np.round(log_loss(y_val, model.predict(X_val, num_iteration=model.best_iteration)), decimals=6)) # confirmed
    return model

lgbm = rankgauss_debug()



# %% NOTE save model template

def save_trained_classifier(model, title, score, save_directory):
    model_file = dtnow() + '_' + title
    model_path = os.path.join(save_directory, model_file)
    pickle.dump(model, open(model_path, 'wb'))
    with open("/kaggle/working/may_model/model_roster.txt", "a+") as file1:
        file1.write(f'{model_path}, {score}\n')
    print(f"Succssfully saved {model_path}")
    return model_path

save_directory = "/kaggle/working/may_model"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# %% NOTE save model template

score = lgbm.best_score['eval']['multi_logloss']
model_pickle = save_trained_classifier(lgbm, 'lgbm_rankgauss', score, save_directory)  

lgbm_pickle = pickle.load(open(model_pickle, 'rb'))
lgbm_pickle.predict(Xtst)

# %% ANCHOR 2. optuna.integration

startTime = timer()
X_train, X_val, y_train, y_val = train_test_split(Xtrn, y, test_size=0.2, random_state=2021)
lgb_params = {
    'objective': 'multiclass',
    'num_class': 4,
    'boosting': 'gbdt',
    'metric': 'multi_logloss'
}
lgb_train = lgb.Dataset(X_train, y_train)
lgb_valid = lgb.Dataset(X_val, y_val, reference=lgb_train)
tuned_model = optuna.integration.lightgbm.train(
    lgb_params, lgb_train,
    valid_sets = [lgb_train, lgb_valid],
    valid_names = ['train', 'eval'],
    num_boost_round=1000,
    verbose_eval = 25, # https://tinyurl.com/yhdmtdm8    
    early_stopping_rounds=20
)

timer(startTime)

# %%

lgb.plot_importance(tuned_model, figsize=(20, 20))



# %%

score = tuned_model.best_score['eval']['multi_logloss']
model_pickle = save_trained_classifier(tuned_model, 'lgbm_rankgauss_integration', score, save_directory)  

# %%

lgb_params = {
    'objective': 'multiclass',
    'num_class': 4,
    'boosting': 'gbdt',
    'metric': 'multi_logloss'
}

DEBUG = True

if DEBUG == True:
    N_SPLITS = 2 
    N_REPEATS = 1
    SEED = 2021
else:
    N_SPLITS = 7 
    N_REPEATS = 3
    SEED = 2021        

def objective(trial, x_train, y_train, params):
    # x_train, y_train: ndarray
    # start_time = timer()    
    param_update = { # api doc - https://lightgbm.readthedocs.io/en/latest/Parameters.html#max_depth
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
    params.update(param_update)

    y_oof = np.zeros(x_train.shape[0])
    scores = []    
    # pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "auc") # depends on the choice of eval_metric; "validation_0-logloss"
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    
    for i, (train_index, valid_index) in enumerate(rskf.split(x_train, y_train)):

        X_A, X_B = x_train[train_index, :], x_train[valid_index, :]
        y_A, y_B = y_train[train_index], y_train[valid_index]
        lgb_train = lgb.Dataset(X_A, y_A)
        lgb_valid = lgb.Dataset(X_B, y_B, reference=lgb_train)   
        model = lgb.train(
            lgb_params, 
            lgb_train, 
            valid_sets=[lgb_train, lgb_valid],
            valid_names=['train', 'eval'],
            early_stopping_rounds=10
        )             
        # lgbmClassifier = lgb.LGBMClassifier(**params)
        # lgbmClassifier.fit(
        #     X_A, y_A, eval_set=[(X_B, y_B)], 
        #     early_stopping_rounds=EARLY_STOPPING_ROUNDS, 
        #     verbose=VERBOSE,
        #     callbacks=[pruning_callback])
        y_oof[valid_index] = model.predict(X_B, num_iteration=model.best_iteration) 
        scores.append(accuracy_score(y_B, y_oof[valid_index]))
        
    trial.set_user_attr(key="best_booster", value=lgbmClassifier) # NOTE update the best model in the optuna's table.
    res = np.mean(scores) 
    
    # timer(start_time)
    return res 

