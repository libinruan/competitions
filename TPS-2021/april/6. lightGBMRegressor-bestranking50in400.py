# SOURCE "TPS Apr 2021 single LGBMRegressor" by hiro5299834 BIZEN
# LINK https://www.kaggle.com/hiro5299834/tps-apr-2021-single-lgbmregressor
#%%
import pandas as pd
import numpy as np
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import lightgbm as lgb
from matplotlib import pyplot
from datetime import datetime, timedelta
import itertools

def label_encoder(c):
    lc = LabelEncoder()
    return lc.fit_transform(c)

train_df = pd.read_csv('/kaggle/input/tabular-playground-series-apr-2021/train.csv')
test_df = pd.read_csv('/kaggle/input/tabular-playground-series-apr-2021/test.csv')
submission = pd.read_csv('/kaggle/input/tabular-playground-series-apr-2021/sample_submission.csv')

all_df = pd.concat([train_df, test_df])    

#%%
# SECTION Feature engineering
# Age fillna with mean age for each class
# NOTE HIRO changed it from average age to median age.
age_map = all_df[['Age', 'Pclass']].dropna().groupby('Pclass').median().to_dict() # TECH turn pd.Series to dict object.
all_df.Age = all_df.Age.fillna(all_df.Pclass.map(age_map['Age']))

# Cabin, fillna with 'X' and take first letter
all_df.Cabin = all_df.Cabin.fillna('X').map(lambda x: x[0].strip())

# Ticket, fillna with 'X', split string and take first split 
all_df.Ticket = all_df.Ticket.fillna('X').map(lambda x: str(x).split()[0] if len(str(x).split()) > 1 else 'X')

# Fare, fillna with mean value
all_df.Fare = all_df.Fare.fillna(all_df.Fare.mean())

# Embarked, fillna with 'X' value
all_df.Embarked = all_df.Embarked.fillna('X')

# Name, take only surnames
all_df.Name = all_df.Name.map(lambda x: x.split(',')[0])

# TECH deal with high cardinality categorical features.
label_cols = ['Name', 'Ticket']
onehot_cols = ['Pclass', 'Sex', 'Cabin', 'Embarked']
numerical_cols = ['Age', 'SibSp', 'Parch', 'Fare', 'Survived']

onehot_encoded_df = pd.get_dummies(all_df[onehot_cols])
label_encoded_df = all_df[label_cols].apply(label_encoder)
numerical_df = all_df[numerical_cols]

all_df = pd.concat([numerical_df, label_encoded_df, onehot_encoded_df], axis=1)
# !SECTION Feature engineering

# SECTION lightGBM parameter
# Re-split all data
X = all_df[:train_df.shape[0]]
y = X.pop('Survived')
X_test = all_df[train_df.shape[0]:].drop(columns=['Survived'])

# TODO Add early_stopping_rounds into Optuna. 
# (U) LINK https://towardsdatascience.com/selecting-optimal-parameters-for-xgboost-model-training-c7cd9ed5e45e
# Find the optimal params kernel here: https://www.kaggle.com/jmargni/tps-apr-2021-lightgbm-optuna
# SOURCE lightGBM api doc http://devdoc.net/bigdata/LightGBM-doc-2.2.2/Parameters.html#num_iterations
params = {
    'metric': 'binary_logloss',
    'n_estimators': 10000, # <--- TECH accuracy trio (1)
    'objective': 'binary',
    'random_state': 2021,
    'learning_rate': 0.02, # <--- TECH accuracy trio (2)
    'min_child_samples': 150,
    'reg_alpha': 3e-5,
    'reg_lambda': 9e-2,
    'num_leaves': 20,
    'max_depth': 16,
    # 'early_stopping_rounds': 100 # <--- TECH accuracy trio (3)
    'colsample_bytree': 0.8,
    'subsample': 0.8, # NOTE similar to feature_fraction. subsample has aliases like bagging and sub_row.
    'subsample_freq': 2,
    'max_bin': 240,
    # 'max_drop': 50 # by default
    'device': 'gpu',      # comment it out if running on CPU
    'gpu_platform_id': 0, # comment it out if running on CPU
    'gpu_device_id': 0    # comment it out if running on CPU 
}
# !SECTION lightGBM parameter

# %%
# SECTION validating the model
oof = np.zeros(X.shape[0])
preds = 0

# NOTE HIRO changed to StratifiedKFold from vallina KFold.
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2021)
start_time = time.time()
for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    print(f"===== FOLD {fold} =====")

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

    # NOTE LGMBRegressor https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html#lightgbm.LGBMRegressor.fit
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)], # TECH for plotting learning curve. If multiple evaluation datasets or multiple evaluation metrics are provided, then early stopping will use the last in the list. 
        early_stopping_rounds=100, # NOTE early stopping of 10 iterations means if the result doesn’t improve in the next 10 iterations, stop training.
        verbose=500
    )
    
    oof[valid_idx] = model.predict(X_valid)
    preds += model.predict(X_test, num_iteration=model.best_iteration_) / skf.n_splits # TECH num_iterations has aliases such as num_rounds, n_estimators, num_trees and n_iters.
    
    acc_score = accuracy_score(y_valid, np.where(oof[valid_idx]>0.5, 1, 0))
    print(f"===== ACCURACY SCORE {acc_score} =====\n")
    
acc_score = accuracy_score(y, np.where(oof>0.5, 1, 0))
print(f"===== ACCURACY SCORE {acc_score} =====")
elapsed_time = time.time() - start_time  
print ("Time elapsed: ", timedelta(seconds=elapsed_time))


# Feature importance of last CV, just to get ideas where try to improve
lgb.plot_importance(model)
submission['Survived'] = np.where(preds>0.5, 1, 0)
submission.to_csv('submission.csv', index=False)
# !SECTION validating the model

# # SECTION Learning curve (the missing part)
# results = model.evals_result()
# epochs = len(results['validation_0']['error'])
# x_axis = range(0, epochs)
# # plot log loss
# fig, ax = pyplot.subplots()
# ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
# ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
# ax.legend()
# pyplot.ylabel('Log Loss')
# pyplot.title('XGBoost Log Loss')
# pyplot.show()
# # plot classification error
# fig, ax = pyplot.subplots()
# ax.plot(x_axis, results['validation_0']['error'], label='Train')
# ax.plot(x_axis, results['validation_1']['error'], label='Test')
# ax.legend()
# pyplot.ylabel('Classification Error')
# pyplot.title('XGBoost Classification Error')
# pyplot.show()
# # !SECTION Learning curve (the missing part)

# %%
