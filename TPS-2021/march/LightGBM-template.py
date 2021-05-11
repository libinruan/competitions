"""
oof-lgbm-svyatoslavsokolov-0
tst-lgbm-svyatoslavsokolov-0
gpu
"""
#%%
import pandas as pd
import numpy as np
import os
import time

class Timer:
    def __enter__(self):
        self.start=time.time()
        return self
    def __exit__(self, *args):
        self.end=time.time()
        self.hour, temp = divmod((self.end - self.start), 3600)
        self.min, self.second = divmod(temp, 60)
        self.hour, self.min, self.second = int(self.hour), int(self.min), round(self.second, 2)
        return self

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
# %%

X = train.drop(['target'], axis= 1)
y = train['target']
temp = pd.concat([X, test])

from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
cat_columns = temp.select_dtypes(include='object').columns.values.tolist() # ['cat0', ...]

with Timer() as t:
    for col in cat_columns:
        label.fit(temp[col])
        temp.loc[:, col] = label.transform(temp[col])
print(f'{t.hour} hours {t.min} minutes {t.second} seconds')   

X = temp.iloc[:len(X)].copy()
test = temp.iloc[len(X):].copy()

lgbm_test_pred = np.zeros(len(test))
n_splits = 10

from sklearn.model_selection import StratifiedKFold  
kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
auc = []
oof = np.zeros(X.shape[0])

import lightgbm
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
#%%
with Timer() as t:
    for trn_idx, val_idx in kf.split(X, y):
        with Timer() as it:
            x_train = X.iloc[trn_idx]
            x_valid = X.iloc[val_idx]
            y_train = y.iloc[trn_idx]
            y_valid = y.iloc[val_idx]

            # identify the indices of label encoded columns
            cat_col_idx = [x_train.columns.get_loc(col) for col in x_train.select_dtypes('int')]

            # parameters after Optuna
            lgbm_parameters = {
                'cat_feature': cat_col_idx, # integer # <--------------------
                'metric': 'auc',
                'random_state': 8862537, 
                'n_estimators': 20000,
                'reg_alpha': 0.000721024661208569,
                'reg_lambda': 47.79748127808107,
                'colsample_bytree': 0.24493010466517195,
                'subsample': 0.12246675404710294,
                'learning_rate': 0.013933182980403087,
                # 'max_bin': 15, # 15 maybe too small
                'max_depth': 21,
                'num_leaves': 90,
                'is_unbalance': True, 
                'min_child_samples': 144,
                'cat_smooth': 63,
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0        
            }

            lgbm_model = LGBMClassifier(**lgbm_parameters)
            lgbm_model.fit(
                x_train, 
                y_train, 
                eval_set = ((x_valid, y_valid)),
                verbose = 1000, 
                early_stopping_rounds = 200) # <----------------------------------------  

            auc.append(
                roc_auc_score(
                    y_valid, 
                    lgbm_model.predict_proba(x_valid)[:, 1])) 
            off[val_idx] = lgbm_model.predict_proba(x_valid)[:, 1]
            lgbm_test_pred += lgbm_model.predict_proba(test)[:, 1] / n_splits
        print(f'{it.hour} hours {it.min} minutes {it.second} seconds')

    print(f'AUC: {np.mean(auc)}')
print(f'{t.hour} hours {t.min} minutes {t.second} seconds')

np.save('oof-lgbm-svyatoslavsokolov-0', oof)
np.save('tst-lgbm-svyatoslavsokolov-0', lgbm_test_pred)

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (6, 5)
lightgbm.plot_importance(lgbm_model,max_num_features=16, height=.9)