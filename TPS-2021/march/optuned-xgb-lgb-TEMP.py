# %% [markdown]
# NOTE The code contains the following information:  
# - repeated stratified K fold

# %% [code]
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [markdown]
# ## Load libraries

# %% [code]
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import warnings
warnings.filterwarnings("ignore")

# %% [markdown]
# ## Read data

# %% [code]
train = pd.read_csv('../input/tabular-playground-series-mar-2021/train.csv')
test = pd.read_csv('../input/tabular-playground-series-mar-2021/test.csv')
sub_xgb = pd.read_csv('../input/tabular-playground-series-mar-2021/sample_submission.csv')
sub_lgb = pd.read_csv('../input/tabular-playground-series-mar-2021/sample_submission.csv')
sub_two = pd.read_csv('../input/tabular-playground-series-mar-2021/sample_submission.csv')

# %% [markdown]
# ## Feature engineering

# %% [code]
target = train['target'].values

# %% [code]
columns = test.columns[1:]
columns

# %% [code]
cont_cols = [col for col in columns if 'cont' in col] # dataframe
cat_cols = [col for col in columns if 'cat' in col] # dataframe

def label_encode(train_df, test_df, column):
    le = LabelEncoder()
    new_feature = "{}_le".format(column)
    le.fit(train_df[column].unique().tolist() + test_df[column].unique().tolist())
    train_df[new_feature] = le.transform(train_df[column])
    test_df[new_feature] = le.transform(test_df[column])
    return new_feature

le_cols = []
for feature in cat_cols:
    le_cols.append(label_encode(train, test, feature))
    
columns = cont_cols + le_cols # remove old cat features and append the derived labeled encoded features.

# %% [markdown]
# ## Build and Run

# %% [code]
# ANCHOR run_rskf
def run_rskf(train, target, clf, params):
    train_preds = np.zeros((train.shape[0], 2)) # repeated two.
    test_preds = 0
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1337)
    for fold, (train_index, val_index) in enumerate(rskf.split(train, target)):
        print("-> Fold {}".format(fold + 1))
       
        x_train, x_valid = train.iloc[train_index][columns], train.iloc[val_index][columns]
        y_train, y_valid = target[train_index], target[val_index]
    
        model = clf(**params)
        model.fit(x_train, y_train,
                    eval_set=[(x_valid, y_valid)], 
                    verbose=0,
                    early_stopping_rounds=500)
    
        train_oof_preds = model.predict_proba(x_valid)[:,1]
        train_preds[val_index, fold//5] = train_oof_preds # we use double slashes to separate one repeatition from another.
        test_oof_preds = model.predict_proba(test[columns])[:,1]
        test_preds += test_oof_preds / 10
        print("ROC AUC Score = {}".format(roc_auc_score(y_valid, train_oof_preds)))
        if fold in [4, 9]:
            print("=> Overall ROC AUC Score = {}".format(roc_auc_score(target, train_preds[:, fold//5])))
    return model, train_preds.mean(axis=1), test_preds



# %% [markdown]
# ## Xgboost

# %% [code]
params_xgb = {'seed':2021,
            'n_estimators':10000,
            'verbosity':1,
            'objective': 'binary:logistic',
            'eval_metric':"auc",
            'tree_method':"gpu_hist",
            'use_label_encoder':False,
            'gpu_id':0,
            'alpha':7.105038963844129,
            'colsample_bytree':0.25505629740052566,
            'gamma':0.4999381950212869,
            'reg_lambda':1.7256912198205319,
            'learning_rate':0.011823142071967673,
            'max_bin':338,
            'max_depth':8,
            'min_child_weight':2.286836198630466,
            'subsample':0.618417952155855}

clf_xgb = XGBClassifier

# %% [code]
model_xgb, oof_preds_xgb, test_preds_xgb = run_rskf(train, target, clf_xgb , params_xgb)

# %% [markdown]
# ## Lightgbm

# %% [code]
params_lgb = {
            'cat_smooth':89.2699690675538,
            'colsample_bytree':0.2557260109926193,
            'learning_rate':0.00918685483594994,
            'max_bin':788,
            'max_depth':81,
            'metric':"auc",
            'min_child_samples':292,
            'min_data_per_group':177,
            'n_estimators':16000,
            'n_jobs':-1,
            'num_leaves':171,
            'reg_alpha':0.7115353581785044,
            'reg_lambda':5.658115293998945,
            'subsample':0.9262904583735796,
            'subsample_freq':1,
            'verbose':-1
            }

clf_lgb = LGBMClassifier

# %% [code]
model_lgb, oof_preds_lgb, test_preds_lgb = run_rskf(train, target, clf_lgb , params_lgb)

# %% [markdown]
# ## Submit

# %% [code]
sub_xgb['target'] = test_preds_xgb
sub_xgb.to_csv('submission_xgb.csv', index=False)

sub_lgb['target'] = test_preds_lgb
sub_lgb.to_csv('submission_lgb.csv', index=False)

sub_two['target'] = (test_preds_xgb + test_preds_lgb)/2
sub_two.to_csv('submission_two.csv', index=False)