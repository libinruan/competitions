"""
oof-onielg-lgbm
tst-onielg-lgbm
oof-onielg-catb
tst-onielg-catb
"""
#%%
# Tabular Playground Series - Mar 2021
# The below code creates a blend of a tuned LGBM Model and a CatBoost Model to obtain a good LB Score.

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import PIL
import json
import gc

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

RunningInCOLAB = os.environ.get('HOME') == '/root' or os.environ.get('PWD') == '/kaggle/working'
if RunningInCOLAB:
    os.chdir('/kaggle/working')
    train = pd.read_csv('../input/tabular-playground-series-mar-2021/train.csv', index_col=0)
    test = pd.read_csv('../input/tabular-playground-series-mar-2021/test.csv', index_col=0)
    sample_sub = pd.read_csv('../input/tabular-playground-series-mar-2021/sample_submission.csv')
else:
    os.chdir('G:\kagglePlayground')
    train = pd.read_csv('train.csv', index_col=0)
    test = pd.read_csv('test.csv', index_col=0)
    sample_submission = pd.read_csv('sample_submission.csv') 

cat_vars = [col for col in train.columns if col.startswith('cat')] 
cont_vars = [col for col in train.columns if col.startswith('cont')] 

# LABEL ENCODE
def encode_LE(col,train,test):
    df_comb = pd.concat([train[col], test[col]], axis=0)
    df_comb, _ = df_comb.factorize(sort=True) # regardless of what sort values woulkd be. I believe.
    
    train[col] = df_comb[:len(train)].astype('int16')
    test[col] = df_comb[len(train):].astype('int16')
    del df_comb; 
    gc.collect()
    print(col, ', ', end='')

# FREQ ENCODE
def encode_FE(df1, df2, cols):
    for col in cols:
        df = pd.concat([df1[col],df2[col]])
        col_dict = df.value_counts(dropna=True, normalize=True).to_dict()
        col_dict[-1] = -1
        colname = col+'_FE'
        df1[colname] = df1[col].map(col_dict)
        df1[colname] = df1[colname].astype('float32')
        
        df2[colname] = df2[col].map(col_dict)
        df2[colname] = df2[colname].astype('float32')
        print(colname, ', ', end='')

# Label Encode
encode_LE('cat0',train,test)
encode_LE('cat11',train,test)
encode_LE('cat12',train,test)
encode_LE('cat13',train,test)
encode_LE('cat14',train,test)
encode_LE('cat15',train,test)
encode_LE('cat16',train,test)
encode_LE('cat17',train,test)
encode_LE('cat18',train,test)

# Frequency Encode
encode_FE(train,test,['cat1','cat2','cat3','cat4','cat5','cat6','cat7','cat8','cat9','cat10'])

train[cat_vars] = train[cat_vars].astype('category')
test[cat_vars] = test[cat_vars].astype('category')

usecols = cat_vars + cont_vars
dep_var = 'target'

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import train_test_split
X = train[usecols]
y = train[dep_var]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

from scipy.stats import uniform as sp_uniform
param_test ={'num_leaves': [31,40,50], 
             'max_depth': [10,15,20,25], 
             'n_estimators': [1000,1500,2000,3000],
             'learning_rate': [0.1,0.15,0.2,0.012,0.01],
             'reg_alpha': [0, 1e-1, 1, 2],
             'bagging_fraction': [0.7,0.8, 0.65],
             'feature_fraction': [0.7,0.8, 0.65],
             'params': ['gbdt'],
             'reg_lambda': [0, 1e-1, 1, 5]}

import lightgbm as lgb
clf = lgb.LGBMClassifier(
          n_estimators=1000,
          early_stopping_rounds=500, 
          verbose_eval=1000)

n_HP_points_to_test = 30
gs = RandomizedSearchCV(
    estimator=clf, param_distributions=param_test, 
    n_iter=n_HP_points_to_test,    
    cv=5,
    refit=True,
    random_state=2021,
    verbose=True)

# gs.fit(train_X[usecols], train_y,
#        eval_set = [(val_X[usecols], val_y)], verbose = 50, 
#         early_stopping_rounds=500, eval_metric='auc')
# print('Best score reached: {} with params: {} '.format(gs.best_score_, gs.best_params_))

params = {}
params["objective"] = "binary"
params["boosting"] = "gbdt"
params['metric']= "AUC",
params["max_depth"] = 20
params["min_data_in_leaf"] = 1
params["min_child_samples"] = 45
params["reg_alpha"] =  4.20
params["reg_lambda"] = 6.34
params["learning_rate"] = 0.01
params["bagging_fraction"] = 0.65
params["feature_fraction"] = 0.65
params["reg_lambda"] = 0.1
params["reg_alpha"] = 0
params["num_leaves"] = 223 #50
params["n_estimators"] = 6000
params["cat_smooth"] = 74
params["nthread"] =  4
params["verbosity"] = -1
params['early_stopping_rounds'] = 500
params['device'] = 'gpu'
params['gpu_platform_id'] = 0
params['gpu_device_id'] = 0 


# ### Train the Models - Cross Validation

import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold

cv_scores = []
pred_test_full = 0
ooflgb = np.zeros(train.shape[0])
predictionslgb= np.zeros(test.shape[0])

N_SPLITS = 5

fold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=2021)

#%%
num_rounds = 6000
for dev_index, val_index in fold.split(train[usecols], train[dep_var]):    
    dev_X, val_X = train[usecols].iloc[dev_index,:].copy(), train[usecols].iloc[val_index,:].copy()
    dev_y, val_y = train[dep_var].iloc[dev_index].copy(), train[dep_var].iloc[val_index].copy()         
    lgtrain = lgb.Dataset(dev_X, label=dev_y)
    lgtest = lgb.Dataset(val_X, label=val_y)
    model = lgb.train(
        params, 
        lgtrain, 
        num_rounds, 
        valid_sets=[lgtest], 
        early_stopping_rounds=200, 
        verbose_eval=1000)
    pred_val  = model.predict(val_X, num_iteration=model.best_iteration)
    pred_test = model.predict(test[usecols], num_iteration=model.best_iteration)
    ooflgb[val_index] = pred_val
    predictionslgb += pred_test / N_SPLITS


np.save('oof-onielg-lgbm', ooflgb)
np.save('tst-onielg-lgbm', predictionslgb)

#%%
# --------------------------------- Catboost --------------------------------- #

from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score
from catboost import CatBoostClassifier

categorical_features_indices = np.where(X.dtypes =='category')[0]
categorical_features_indices
oofcat = np.zeros(X.shape[0])
y_pred_totcb = np.zeros(test.shape[0]) 

from sklearn.model_selection import KFold,StratifiedKFold
fold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=2021)
# fold = KFold(n_splits=N_SPLITS,shuffle=False,random_state=2021)

for i, (train_index, test_index) in enumerate(fold.split(X,y)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    m = CatBoostClassifier(
            n_estimators=6000,
            random_state=2021,
            eval_metric='AUC',
            max_depth=6,
            learning_rate=0.01,
            od_wait=50,
            l2_leaf_reg=10,
            task_type="GPU",
            cat_features=categorical_features_indices,
            bagging_temperature=0.80,
            random_strength=100,
            use_best_model=True)
    m.fit(X_train, 
        y_train,eval_set = [(X_test, y_test)], 
        early_stopping_rounds = 200, 
        verbose = 1000)
    oofcat[test_index] = m.predict_proba(X_test)[:,-1]
    p = m.predict_proba(test[usecols])[:,-1]
    y_pred_totcb += p / N_SPLITS


#%%
np.save('oof-onielg-catb', oofcat)
np.save('tst-onielg-catb', y_pred_totcb)    
    
# y_pred_totcb = y_pred_totcb/5 

# sample_sub['target'] = pd.DataFrame(predictionslgb).rank(pct=True)
# sample_sub.to_csv('submission.csv',index=False)

#sample_sub['target'] = y_pred_totcb
sample_sub['target'] = pd.DataFrame(y_pred_totcb).rank(pct=True) * 0.60 + pd.DataFrame(predictionslgb).rank(pct=True) * 0.40
sample_sub.to_csv('submission.csv',index=False)

# ### How did the model do...

# y_pred = sample_sub['target']
plt.figure(figsize=(8,4))
plt.hist(oofcat[np.where(y == 0)], bins=100, alpha=0.75, label='neg class')
plt.hist(oofcat[np.where(y == 1)], bins=100, alpha=0.75, label='pos class')
plt.legend()
plt.show()

# y_pred = sample_sub['target']
plt.figure(figsize=(8,4))
plt.hist(ooflgb[np.where(y == 0)], bins=100, alpha=0.75, label='neg class')
plt.hist(ooflgb[np.where(y == 1)], bins=100, alpha=0.75, label='pos class')
plt.legend()
plt.show()

# ### Understanding the predictions

from sklearn.model_selection import train_test_split

row_to_show = 5
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
data_for_prediction = val_X.iloc[row_to_show]  
data_for_prediction_array = data_for_prediction.values.reshape(1, -1)

m.predict_proba(data_for_prediction_array)

import shap
explainer = shap.TreeExplainer(m)
shap_values = explainer.shap_values(data_for_prediction_array)

shap.initjs()
shap.force_plot(explainer.expected_value, shap_values, data_for_prediction_array)
