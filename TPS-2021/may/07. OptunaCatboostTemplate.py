# %%

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
pd.options.display.max_columns = 100

# Importing Catboost and sklearn libraries.
from catboost import Pool, CatBoostClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import log_loss

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

# Reading datasets.
train = pd.read_csv("/kaggle/input/tabular-playground-series-may-2021/train.csv")
test = pd.read_csv("/kaggle/input/tabular-playground-series-may-2021/test.csv")
submission = pd.read_csv("/kaggle/input/tabular-playground-series-may-2021/sample_submission.csv")

# Saving all features to all_features list.
all_features=[]
for i in range(0,50):
    all_features.append("feature_" + str(i))

# NOTE convert to string. Catboost seems to only work on string targets (not verify it yet)    
# Converting all features to string.
for column in all_features:
    train[column] = train[column].astype(str)
    test[column] = test[column].astype(str)
    
    
# Removing records which are not appear in test set from train set.
train_have_test_dont_have = {}
for feature in all_features:
    value = list(set(list(train[feature].unique())) - set(list(test[feature].unique())))
    train_have_test_dont_have.update({feature:value})

for feature,value in train_have_test_dont_have.items():
    train = train[~(train[feature].isin(value))]
train.reset_index(inplace=True)
print(train.shape)

# %% experiment
n_splits = 10
n_repeats = 1
n_trials = 500
iterations=100000
random_state = 33

folds = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)  
df = train.copy() # 
columns=all_features # 
categoric_columns = all_features #
X_train = df[columns]
y_train = df['target']
logloss_all = []
_proba = np.zeros((X_train.shape[0], y_train.nunique()))
_probas = np.zeros((X_train.shape[0], y_train.nunique()))
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X_train,  y_train)):
    print("Fold --> " + str(n_fold+1) + "/" + str(n_splits))
    train_X, train_y = X_train.iloc[train_idx].copy(), y_train.iloc[train_idx]
    valid_X, valid_y = X_train.iloc[valid_idx].copy(), y_train.iloc[valid_idx]
    dataset = Pool(train_X, train_y, categoric_columns)
    evalset = Pool(valid_X, valid_y, categoric_columns)
    model = CatBoostClassifier(
        task_type="GPU",
        depth=4,
        iterations=iterations,
        od_wait=1000,
        od_type='Iter',
        learning_rate=0.02,
        use_best_model=True,
        loss_function='MultiClass',
        verbose = False
        )
    model.fit(dataset, plot=False, verbose=500, eval_set=evalset)
    _proba = model.predict_proba(valid_X[all_features])
    logloss_of_fold = log_loss(list(valid_y),_proba)
    logloss_all.append(logloss_of_fold)
    _probas[valid_idx, :] += _proba / n_repeats
    print(f"logloss of validation for repeat, split {(n_fold // n_splits) + 1} fold {(n_fold % n_splits) + 1} --> {logloss_of_fold}")
# print(f"{'average repeat logloss -->':>28} {log_loss(list(y_train), _probas)}")    
print(f"{'average split logloss -->':>28} {np.mean(logloss_all)}")    
# print(f"{'best score: ':>15}{model.get_best_score()['validation']['MultiClass']}")

# Result: screenshot https://i.postimg.cc/J794zwHG/2021-05-30-at-15-19-41.png CV: 1.09045 random seed: 2021
# https://i.postimg.cc/yYjYKL9G/2021-05-30-at-15-44-41.png random seed: 33

pickle.dump(model, open(f'/kaggle/working/may_model/{dtnow()}-catboost-solo-10f1r-plain-CV1d09048.pkl', 'wb'))


# %% 
import pickle
model_solo = pickle.load(open('/kaggle/working/may_model/202105301551-catboost-solo-10f1r-plain-CV1d09048.pkl', 'rb'))
import optuna # Start 1503 PM 0530 2021.

n_splits = 10
n_repeats = 1
n_trials = 200
iterations= 100000
random_state = 2021
n_warmup_steps = 0

def objective(trial):
    params = {'iterations': iterations,
              'od_wait': 1000,
              'od_type': 'Iter',
              'learning_rate': 0.02, # trial.suggest_float("learning_rate", 5e-3, 0.05, log=True), # Bad (1e-4, 5e-2) -> Good (5e-3, 5e-2)
              'max_leaves': trial.suggest_int("max_leaves", 3, 16), # Bad (10, 32) -> Good (10, 20)
              'depth': trial.suggest_int("depth", 3, 16),
              'l2_leaf_reg': 0.514, # trial.suggest_float("l2_leaf_reg", 0.001, 25, log=True),
              'bagging_temperature': 9.7107, # trial.suggest_float("bagging_temperature", 2, 10),
              'auto_class_weights': 'None', #trial.suggest_categorical('auto_class_weights', ['None','Balanced','SqrtBalanced']),
              'min_data_in_leaf': trial.suggest_int("min_data_in_leaf", 1, 15),
              'use_best_model': True,
              'task_type': 'GPU', 
              'verbose': False,
              'loss_function': 'MultiClass',              
              'border_count': 128, #trial.suggest_categorical('border_count', [32, 64, 128]), # Bad [32, 64, 128, 254] -> Good [32, 64, 128]            
            #   "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1), # leads to error in multiclass classification.
              'grow_policy': 'Lossguide', # if you enable seach on max_leaves, you needs to use it.
            #   'bootstrap_type': 'Bayesian',
             }
    folds = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)  
    df = train.copy() # <---
    columns=all_features # <---
    categoric_columns = all_features #
    X_train = df[columns]
    y_train = df['target']
    logloss_all = []
    _proba = np.zeros((X_train.shape[0], y_train.nunique()))
    _probas = np.zeros((X_train.shape[0], y_train.nunique()))             
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X_train,  y_train)):
        print("Fold --> " + str(n_fold+1) + "/" + str(n_splits * n_repeats))
        train_X, train_y = X_train.iloc[train_idx].copy(), y_train.iloc[train_idx]
        valid_X, valid_y = X_train.iloc[valid_idx].copy(), y_train.iloc[valid_idx]
        dataset = Pool(train_X, train_y, categoric_columns)
        evalset = Pool(valid_X, valid_y, categoric_columns)
        model = CatBoostClassifier(
            **params
            )
        model.fit(dataset, plot=False, verbose=500, eval_set=evalset)
        _proba = model.predict_proba(valid_X[all_features])
        logloss_of_fold = log_loss(list(valid_y),_proba)
        logloss_all.append(logloss_of_fold)
        _probas[valid_idx, :] += _proba / n_repeats
        # print(f"logloss of validation for repeat, split {(n_fold // n_splits) + 1} fold {(n_fold % n_splits) + 1} --> {logloss_of_fold}")
    # avg_repeat_loss = log_loss(list(y_train), _probas)
    # print(f"{'average repeat logloss -->':>28} {avg_repeat_loss}")    
    # print(f"{'average split logloss -->':>28} {np.mean(logloss_all)}")   
    return np.mean(logloss_all) # avg_repeat_loss
study = optuna.create_study(direction="minimize",
    sampler = optuna.samplers.TPESampler(seed=random_state, multivariate=True),
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=n_warmup_steps)
)
study.enqueue_trial(model_solo.get_all_params()) # here we use the vallina solo model's parameter as our starting point.
study.optimize(objective, timeout= 4 * 60 * 60, n_trials = n_trials)    

# print(study.best_trial)
print("Number of finished trials: {}".format(len(study.trials)))
print("Best trial:")
trial = study.best_trial
print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value)) 

import pickle
name = f'{dtnow()}-catboost-10f1r-4hr-OptunaParamDict'
pickle.dump(trial.params, open(f'/kaggle/working/may_model/{name}.pkl', 'wb'))
best_params = pickle.load(open(f'/kaggle/working/may_model/{name}.pkl', 'rb'))
# submission.to_csv(name, index=False)
# gcs_folder = 'tps-may-2021-label/'
# local_filename = f'{name}'
# local_folder = './'
# blob = bucket.blob(f'{gcs_folder}{local_filename}')
# blob.upload_from_filename(f'{local_folder}{local_filename}') 
j# %%

# %%


import pickle
model_solo = pickle.load(open('/kaggle/working/may_model/202105301551-catboost-solo-10f1r-plain-CV1d09048.pkl', 'rb'))
import optuna # Start 1503 PM 0530 2021.

n_splits = 6
n_repeats = 1
n_trials = 80
iterations= 100000
random_state = 2021
n_warmup_steps = 0

def objective(trial):
    params = {'iterations': iterations,
              'od_wait': 100,
              'od_type': 'Iter',
              'learning_rate': 0.02, # trial.suggest_float("learning_rate", 5e-3, 0.05, log=True), # Bad (1e-4, 5e-2) -> Good (5e-3, 5e-2)
              'max_leaves': trial.suggest_int("max_leaves", 3, 16), # Bad (10, 32) -> Good (10, 20)
              'depth': trial.suggest_int("depth", 3, 16),
              'l2_leaf_reg': 0.514, # trial.suggest_float("l2_leaf_reg", 0.001, 25, log=True),
              'bagging_temperature': 9.7107, # trial.suggest_float("bagging_temperature", 2, 10),
              'auto_class_weights': 'None', #trial.suggest_categorical('auto_class_weights', ['None','Balanced','SqrtBalanced']),
              'min_data_in_leaf': trial.suggest_int("min_data_in_leaf", 1, 15),
              'use_best_model': True,
              'task_type': 'GPU', 
              'verbose': False,
              'loss_function': 'MultiClass',              
              'border_count': 128, #trial.suggest_categorical('border_count', [32, 64, 128]), # Bad [32, 64, 128, 254] -> Good [32, 64, 128]            
            #   "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1), # leads to error in multiclass classification.
              'grow_policy': 'Lossguide', # if you enable seach on max_leaves, you needs to use it.
            #   'bootstrap_type': 'Bayesian',
             }
    folds = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)  
    df = train.copy() # <---
    columns=all_features # <---
    categoric_columns = all_features #
    X_train = df[columns]
    y_train = df['target']
    logloss_all = []
    _proba = np.zeros((X_train.shape[0], y_train.nunique()))
    _probas = np.zeros((X_train.shape[0], y_train.nunique()))             
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X_train,  y_train)):
        print("Fold --> " + str(n_fold+1) + "/" + str(n_splits * n_repeats))
        train_X, train_y = X_train.iloc[train_idx].copy(), y_train.iloc[train_idx]
        valid_X, valid_y = X_train.iloc[valid_idx].copy(), y_train.iloc[valid_idx]
        dataset = Pool(train_X, train_y, categoric_columns)
        evalset = Pool(valid_X, valid_y, categoric_columns)
        model = CatBoostClassifier(
            **params
            )
        model.fit(dataset, plot=False, verbose=500, eval_set=evalset)
        _proba = model.predict_proba(valid_X[all_features])
        logloss_of_fold = log_loss(list(valid_y),_proba)
        logloss_all.append(logloss_of_fold)
        _probas[valid_idx, :] += _proba / n_repeats
        # print(f"logloss of validation for repeat, split {(n_fold // n_splits) + 1} fold {(n_fold % n_splits) + 1} --> {logloss_of_fold}")
    # avg_repeat_loss = log_loss(list(y_train), _probas)
    # print(f"{'average repeat logloss -->':>28} {avg_repeat_loss}")    
    # print(f"{'average split logloss -->':>28} {np.mean(logloss_all)}")   
    return np.mean(logloss_all) # avg_repeat_loss
study = optuna.create_study(direction="minimize",
    sampler = optuna.samplers.TPESampler(seed=random_state, multivariate=True),
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=n_warmup_steps)
)
study.enqueue_trial(best_params) # here we use the vallina solo model's parameter as our starting point.
study.optimize(objective, timeout= 2 * 60 * 60, n_trials = n_trials)    

# print(study.best_trial)
print("Number of finished trials: {}".format(len(study.trials)))
print("Best trial:")
trial = study.best_trial
print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value)) 

import pickle
name = f'{dtnow()}-catboost-10f1r-4hr-OptunaParamDict'
pickle.dump(trial.params, open(f'/kaggle/working/may_model/{name}.pkl', 'wb'))
# best_params = pickle.load(open(f'/kaggle/working/may_model/{name}.pkl', 'rb'))
# %%
