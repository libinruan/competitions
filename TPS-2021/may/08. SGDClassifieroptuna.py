# %%
"""
reference: https://www.kaggle.com/napetrov/tps04-svm-with-scikit-learn-intelex

import sys
!{sys.executable} -m pip install scikit-learn-intelex --progress-bar off >> /tmp/pip_sklearnex.log

NOT FULLY DEVELOPMENT!
"""


import pandas as pd
import numpy as np
import random
import os
import pickle
import datetime
import pytz

from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.linear_model import SGDClassifier

from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from category_encoders import OrdinalEncoder
import lightgbm as lgb
from scipy.special import erfinv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC

pd.options.display.max_rows = 100
pd.options.display.max_columns = 100

import warnings
warnings.simplefilter('ignore')

TARGET = 'target'
NUM_CLASSES = 4
SEED = 2021

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
seed_everything(SEED)

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


os.chdir('/kaggle/working')
train = pd.read_csv('../input/tabular-playground-series-may-2021/train.csv')
test = pd.read_csv('../input/tabular-playground-series-may-2021/test.csv')
submission = pd.read_csv('../input/tabular-playground-series-may-2021/sample_submission.csv')

feature_columns = train.iloc[:, 1:-1].columns.values
all_df = pd.concat([train, test]).reset_index(drop=True).drop(['id', 'target'], axis=1)

from sklearn.preprocessing import LabelEncoder
all_df = all_df.apply(LabelEncoder().fit_transform)

def rank_gauss(x):
    N = x.shape[0]
    temp = x.argsort()
    rank_x = temp.argsort() / N
    rank_x -= rank_x.mean()
    rank_x *= 2
    efi_x = erfinv(rank_x)
    efi_x -= efi_x.mean()
    return efi_x

all_df = all_df.apply(rank_gauss)    
for col in feature_columns:
    all_df[col] = StandardScaler().fit_transform(all_df[col].values.reshape(-1,1))

# experiment
Xtrn, Xtst = all_df.iloc[:len(train)], all_df.iloc[len(train):]
y = train['target']
RANDOM_SEED = 2021

# X_train, X_test, y_train, y_test = train_test_split(Xtrn, y, test_size = 0.20, random_state = RANDOM_SEED)
# svc_kernel_rbf = SVC(kernel='rbf', random_state=0, C=0.779481782160288, gamma=0.10264575666119422 )
# svc_kernel_rbf.fit(X_train, y_train)
# y_pred = svc_kernel_rbf.predict(X_test)
# accuracy_score(y_pred, y_test)

def objective(trial):

sgd = SGDClassifier(loss='log', alpha=0.001, early_stopping=True, n_jobs=-1)
# %%

sgd.fit(Xtrn, y)
sgd.predict_proba(Xtrn)

n_splits = 10
n_repeats = 1
n_trials = 500
iterations=100000
random_state = 33

folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)  
logloss_all = []
_predict = np.zeros((Xtrn.shape[0]))
_probas = np.zeros((Xtrn.shape[0]))
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(Xtrn,  y)):
    print("Fold --> " + str(n_fold+1) + "/" + str(n_splits))
    train_X, train_y = Xtrn.iloc[train_idx].copy(), y.iloc[train_idx]
    valid_X, valid_y = Xtrn.iloc[valid_idx].copy(), y.iloc[valid_idx]
    params = {
        "kernel": 'rbf',
        "random_state": 2021,
        "C": trial.suggest_float("C", 1e-3, 10),
        "gamma": "auto"
    }
    model = SVC(**params)
    model.fit(train_X, train_y)
    _predict = model.predict(valid_X)
    logloss_of_fold = log_loss(list(valid_y), _predict)
    logloss_all.append(logloss_of_fold)
    _probas[valid_idx] += _proba / n_repeats
    print(f"logloss of validation for repeat, split {(n_fold // n_splits) + 1} fold {(n_fold % n_splits) + 1} --> {logloss_of_fold}")
# print(f"{'average repeat logloss -->':>28} {log_loss(list(y_train), _probas)}")    
print(f"{'average split logloss -->':>28} {np.mean(logloss_all)}")    