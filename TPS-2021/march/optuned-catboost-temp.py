# trainable.Lipin.
import os
import pandas as pd
from sklearn.metrics import roc_auc_score
import optuna
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedStratifiedKFold
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
from datetime import datetime

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('Time taken : %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

RunningInCOLAB = 'COLAB_GPU' in os.environ or os.environ.get('PWD') == '/kaggle/working'
if RunningInCOLAB:
    os.chdir('/kaggle/working')
    train = pd.read_csv('../input/tabular-playground-series-mar-2021/train.csv', index_col=0)
    test = pd.read_csv('../input/tabular-playground-series-mar-2021/test.csv', index_col=0)
    sample_submission = pd.read_csv('../input/tabular-playground-series-mar-2021/sample_submission.csv')
else:
    os.chdir('G:\kagglePlayground')
    train = pd.read_csv('train.csv', index_col=0)
    test = pd.read_csv('test.csv', index_col=0)
    sample_submission = pd.read_csv('sample_submission.csv') 

# %% [markdown]
# # CatBoost Optuna

# %% [markdown]
# ## Load data

# %% [code]
import pandas as pd

# %% [code]
train = pd.read_csv('../input/tabular-playground-series-mar-2021/train.csv')
test = pd.read_csv('../input/tabular-playground-series-mar-2021/test.csv')
train.head()

# %% [code]
cat_cols = [x for x in train.columns if x.startswith('cat')]
cat_cols

# %% [code]
# from cuml.preprocessing.TargetEncoder import TargetEncoder
# from categorical_transform import CategoricalTransform,IntegerCategoricalTransform
# ct = IntegerCategoricalTransform(cat_cols)
# x_train = ct.fit_transform(train)
# x_test = ct.transform(test)

# %% [code]
x_train = train.drop(columns=['id','target'])
y_train = train['target']

# %% [markdown]
# # Optuna optimization

# %% [code]


def objective(trial):
    params = {'iterations': 10000,
              'depth': trial.suggest_int("depth", 4, 16),
              'l2_leaf_reg': trial.suggest_float("l2_leaf_reg", 0.0001, 25, log=True),
              'bagging_temperature': trial.suggest_float("bagging_temperature", 0, 10),
              'auto_class_weights': trial.suggest_categorical('auto_class_weights', [None,'Balanced','SqrtBalanced']),
              'grow_policy': 'Lossguide',
              'early_stopping_rounds': 200,
              'eval_metric': 'AUC',
              'bootstrap_type': 'Bayesian',
              'use_best_model': True,
              'task_type': 'GPU', 
              'cat_features': cat_cols,
              'verbose': True,
              'border_count': 254              
             }
    #'grow_policy': trial.suggest_categorical('grow_policy',['SymmetricTree','Depthwise','Lossguide']),              
    #if params['grow_policy'] in ['Depthwise','Lossguide']:
    #    params['min_data_in_leaf'] = trial.suggest_int("min_data_in_leaf", 1, 5000, log=True)
    #if params['grow_policy'] in ['Lossguide']:
    #    params['max_leaves'] = trial.suggest_int("max_leaves", 1, 64)
    
    cbc = CatBoostClassifier(**params)
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    roc_test = []
    for train_index, test_index in kf.split(x_train, y_train):
        x_train_fold, x_test_fold = x_train.loc[train_index], x_train.loc[test_index]
        y_train_fold, y_test_fold = y_train.loc[train_index], y_train.loc[test_index]
        cbc.fit(x_train_fold, y_train_fold, eval_set=(x_test_fold, y_test_fold))    
        proba = cbc.predict_proba(x_test_fold)[:,1]
        roc_test.append(roc_auc_score(y_test_fold, proba))
    return np.mean(roc_test)

# %% [code]
# start_time = timer(None)
# study = optuna.create_study(direction="maximize")
# study.optimize(objective, timeout=5*60*60)
# # print(study.best_trial)
# print("Number of finished trials: {}".format(len(study.trials)))
# print("Best trial:")
# trial = study.best_trial
# print("  Value: {}".format(trial.value))
# print("  Params: ")
# for key, value in trial.params.items():
#     print("    {}: {}".format(key, value))    
# timer(start_time)

# %% [code]
# study.best_params

# %% [code]
# study.best_value

# %% [code]
# len(study.trials)

# %% [code]
# from optuna.visualization import plot_optimization_history, plot_param_importances
# plot_optimization_history(study)

# %% [code]
# plot_param_importances(study)

# %% [code]
# screenshot: https://i.postimg.cc/9M0sKtPb/2021-03-30-at-18-36-46.png
# 12 trials best model.
# ------------------------------- Run the best ------------------------------- #
params_optuned = {'iterations': 10000,
          'depth': 11, # trial.suggest_int("depth", 4, 16),
          'l2_leaf_reg': 0.000689801155369707, # trial.suggest_float("l2_leaf_reg", 0.0001, 25, log=True),
          'bagging_temperature': 9.800730196440876, # trial.suggest_float("bagging_temperature", 0, 10),
          'auto_class_weights': 'Balanced', # trial.suggest_categorical('auto_class_weights', [None,'Balanced','SqrtBalanced']),
          'grow_policy': 'Lossguide',
          'early_stopping_rounds': 200,
          'eval_metric': 'AUC',
          'bootstrap_type': 'Bayesian',
          'use_best_model': True,
          'task_type': 'GPU', 
          'cat_features': cat_cols,
          'verbose': False,
          'border_count': 254              
         }

def run_rskf(train, target, test, clf, params):
    train_preds = np.zeros((train.shape[0], 2)) # repeated two.
    test_preds = 0
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1337)
    for fold, (train_index, val_index) in enumerate(rskf.split(train, target)):
        print("-> Fold {}".format(fold + 1))
        start_time = timer(None)
        x_train, x_valid = train.iloc[train_index], train.iloc[val_index]
        y_train, y_valid = target.iloc[train_index], target.iloc[val_index]
    
        model = clf(**params)
        model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)])
    
        train_oof_preds = model.predict_proba(x_valid)[:,1]
        train_preds[val_index, fold//5] = train_oof_preds # we use double slashes to separate one repeatition from another.
        test_oof_preds = model.predict_proba(test)[:,1]
        test_preds += test_oof_preds / 10
        print("ROC AUC Score = {}".format(roc_auc_score(y_valid, train_oof_preds)))
        if fold in [4, 9]:
            print("=> Overall ROC AUC Score = {}".format(roc_auc_score(target, train_preds[:, fold//5])))
        timer(start_time)
    return model, train_preds.mean(axis=1), test_preds

x_train = train.drop(columns=['id','target'])
y_train = train['target']
x_test = test.drop('id', axis=1)
clf_cb = CatBoostClassifier  
model_cb, oof_preds_cb, test_preds_cb = run_rskf(x_train, y_train, x_test, clf_cb, params_optuned)

np.save('./oof_preds_cb.npy', oof_preds_cb)
np.save('./test_preds_cb.npy', test_preds_cb)