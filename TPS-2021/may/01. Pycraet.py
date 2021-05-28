
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

# %% ANCHOR 3. experiment focal loss

# "Focal loss implementation for LightGBM" : https://maxhalford.github.io/blog/lightgbm-focal-loss/
# "Multi-Class classification using Focal Loss and LihtGBM" : https://towardsdatascience.com/multi-class-classification-using-focal-loss-and-lightgbm-a6a6dec28872

from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, accuracy_score,confusion_matrix, roc_auc_score
import numpy as np

import itertools
import matplotlib.pyplot as plt
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

X, y = make_classification(n_classes=3,
                           n_samples=200000, 
                           n_features=2,
                           n_informative=2,
                           n_redundant =0,
                           n_clusters_per_class=1,
                           weights=[.005, .01, .985], 
                           flip_y=.01, 
                           random_state=42)

le = preprocessing.LabelEncoder()
y_label = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_label, test_size=0.30, random_state=42)    

classes =[]
labeles=np.unique(y_label)
for v in labeles:
    classes.append('Class '+ str(v))
print(classes)

# %%

import lightgbm as lgb
clf = lgb.LGBMClassifier()
clf.fit(X_train, y_train, verbose=0)

clf = lgb.LGBMClassifier()
clf.fit(X_train, y_train)

y_test_pred = clf.predict(X_test)
pred_accuracy_score = accuracy_score(y_test, y_test_pred)
pred_recall_score = recall_score(y_test, y_test_pred, average='macro')
print('Prediction accuracy', pred_accuracy_score,' recall ', pred_recall_score)

cnf_matrix = confusion_matrix(y_test, y_test_pred, labels=labeles)
plot_confusion_matrix(cnf_matrix, classes=classes,normalize=True,  title='Confusion matrix')

# NOTE load function in the same folder
# %cd /kaggle/working/TPS-2021/may/
# %load OneVsRestLightGBMWithCustomizedLoss.py
# %load FocalLoss.py
# # reference: https://tinyurl.com/yk4te2ov

# %%

from OneVsRestLightGBMWithCustomizedLoss import *
from FocalLoss import FocalLoss

# Instantiate Focal loss
# loss = FocalLoss(alpha=0.75, gamma=2.0)
loss = FocalLoss(alpha=0.75, gamma=2.0)

# Not using early stopping
clf2 = OneVsRestLightGBMWithCustomizedLoss(loss=loss)
# clf2.fit(X_train, y_train)

# Using early stopping
fit_params = {'eval_set': [(X_test, y_test)]}
clf2.fit(X_train, y_train, **fit_params)

y_test_pred2 = clf2.predict(X_test)
pred_accuracy_score2 = accuracy_score(y_test, y_test_pred2)
pred_recall_score2 = recall_score(y_test, y_test_pred2, average='macro')
print('prediction accuracy', pred_accuracy_score2,' recall ', pred_recall_score2)

cnf_matrix2 = confusion_matrix(y_test, y_test_pred2, labels=labeles)
plot_confusion_matrix(cnf_matrix2, classes=classes,normalize=True,  title='Confusion matrix')


# %% ANCHOR 4. Optuna + LightGBM 
from tqdm import tqdm
DEBUG = True

if DEBUG == True:
    N_SPLITS = 2 
    N_REPEATS = 1
    SEED = 2021
    N_TRIALS = 5
    TIMEOUT = 1 * 60
else:
    N_SPLITS = 7
    N_REPEATS = 6
    SEED = 2021    
    N_TRIALS = 100
    TIMEOUT = 6 * 60 * 60        

# core parameters : https://lightgbm.readthedocs.io/en/latest/Parameters.html
params = {
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'num_class': 4,
    'boosting': 'gbdt',
    'metric': 'multi_logloss'
    # 'num_threads': 4
    # 'device': 'gpu'
}
exp_counter = 0
def objective(trial, X_train, y_train, params):
    # x_train, y_train: ndarray
    start_time = timer()    
    global exp_counter
    exp_counter += 1
    param_update = { # api doc - https://lightgbm.readthedocs.io/en/latest/Parameters.html#max_depth
        'learning_rate': 0.06, # trial.suggest_float('learning_rate', 1e-4, 1e-2),
        'max_depth': trial.suggest_int('max_depth', 1, 127), # default: -1 (no limit)
        'num_leaves': trial.suggest_int('num_leaves', 15, 255), # default: 31. Total num of leaves in one tree.
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-7, 1.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-7, 1.0, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.1, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 1000)        
        # 'num_leaves': trial.suggest_categorical('num_leaves', [31, 63, 127, 255]), # default: 31
        # 'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0), # default: 0. lambda_l1.
        # 'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0), # default: 0. lambda_l2.
        # 'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 0.9), # feature fraction.
        # 'min_child_samples': trial.suggest_int('min_child_samples', 1, 300), # min_data_in_leaf.
        # 'subsample_freq': trial.suggest_int('subsample_freq', 1, 10), # NOTE definition - With subsample (or bagging_fraction)  you can specify the percentage of rows used per tree building iteration. 
        # 'subsample': trial.suggest_float('subsample', 0.3, 0.9), # https://lightgbm.readthedocs.io/en/latest/Parameters.html
        # # 'max_bin': trial.suggest_int('max_bin', 128, 1024), # default: 255. smaller more power to deal with overfitting
        # 'max_bin': trial.suggest_categorical('max_bin', [15, 31, 63, 127, 255]), # default: 255. smaller more power to deal with overfitting
        # 'min_data_per_group': trial.suggest_int('min_data_per_group', 50, 200), # default: 100
        # 'cat_smooth': trial.suggest_int('cat_smooth', 10, 100),
        # 'cat_l2': trial.suggest_int('cat_l2', 1, 20) # L2 regularization in categorical split
    }
    params.update(param_update)

    losses = []    
    # pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "auc") # depends on the choice of eval_metric; "validation_0-logloss"
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "multi_logloss")
    for i, (train_index, valid_index) in tqdm(enumerate(rskf.split(X_train, y_train))):
        print(f"{exp_counter} - {i}")
        X_A, X_B = X_train.iloc[train_index, :], X_train.iloc[valid_index, :]
        y_A, y_B = y_train.iloc[train_index], y_train.iloc[valid_index]
        lgb_train = lgb.Dataset(X_A, y_A)
        lgb_valid = lgb.Dataset(X_B, y_B, reference=lgb_train)   
        lgbm_model = lgb.train(
            params, 
            lgb_train, 
            valid_sets=[lgb_train, lgb_valid],
            valid_names=['train', 'valid_0'],
            num_boost_round=10000,
            verbose_eval = -1, # https://tinyurl.com/yhdmtdm8    
            early_stopping_rounds=20,
            callbacks=[pruning_callback]
        )             
        # lgbmClassifier = lgb.LGBMClassifier(**params)
        # lgbmClassifier.fit(
        #     X_A, y_A, eval_set=[(X_B, y_B)], 
        #     early_stopping_rounds=EARLY_STOPPING_ROUNDS, 
        #     verbose=VERBOSE,
        #     callbacks=[pruning_callback])
        y_oof = lgbm_model.predict(X_B) # not needed, num_iteration=lgbm_model.best_iteration
        losses.append(log_loss(y_B, y_oof))

    trial.set_user_attr(key="best_booster", value=lgbm_model) 
    res = np.mean(losses) 

    timer(start_time)
    return res 

def save_best(study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="best_booster", value=trial.user_attrs["best_booster"])

study = optuna.create_study(
    direction = "minimize", 
    sampler = optuna.samplers.TPESampler(seed=2021, multivariate=True),
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=3))

study.optimize(lambda trial: objective(trial, Xtrn, y, params), 
               n_trials=N_TRIALS, 
               timeout=TIMEOUT, 
               callbacks=[save_best],
               n_jobs=1)

# best iteration : https://tinyurl.com/ygb6kftr
hp = study.best_params
for key, value in hp.items():
    print(f"{key:>20s} : {value}")
print(f"{'best objective value':>20s} : {study.best_value}")  
best_model=study.user_attrs["best_booster"]               

score = study.best_value
model_pickle = save_trained_classifier(best_model, 'lgbm_1min', score, save_directory)  

lgbm_pickle = pickle.load(open(model_pickle, 'rb'))
lgbm_pickle.predict(Xtst)

# %% ANCHOR 10 FOLD RANKGAUSS VALLINA LGBM
N_SPLITS = 8 
N_REPEATS = 5
SEED = 2021   

rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
lgb_params = {
    'objective': 'multiclass',
    'num_class': 4,
    'boosting': 'gbdt',
    'metric': 'multi_logloss'
}
losses = []    
for i, (train_index, valid_index) in enumerate(rskf.split(Xtrn, y)):

    X_A, X_B = Xtrn.iloc[train_index, :], Xtrn.iloc[valid_index, :]
    y_A, y_B = y.iloc[train_index], y.iloc[valid_index]
    lgb_train = lgb.Dataset(X_A, y_A)
    lgb_valid = lgb.Dataset(X_B, y_B, reference=lgb_train)   
    lgbm_model = lgb.train(
        params, 
        lgb_train, 
        valid_sets=[lgb_train, lgb_valid],
        valid_names=['train', 'valid_0'],
        num_boost_round=10000,
        verbose_eval = 50, # https://tinyurl.com/yhdmtdm8    
        early_stopping_rounds=20,
        # callbacks=[pruning_callback]
    )             
y_oof = lgbm_model.predict(X_B) # not needed, num_iteration=lgbm_model.best_iteration
losses.append(log_loss(y_B, y_oof))
print(np.mean(losses))
# %%
score = np.mean(losses)
model_pickle = save_trained_classifier(best_model, 'lgbm_8f5r_integration_vallina', score, save_directory)  

model_pickle = '/kaggle/working/may_model/202105271625_lgbm_1min'
lgbm_pickle = pickle.load(open(model_pickle, 'rb'))
lgbm_pickle.predict(Xtst)

