#%%
import subprocess
import pandas as pd
import numpy as np
import random
import os
import pickle
from datetime import datetime
from google.cloud import storage
import sshColab
import gc

from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.inspection import permutation_importance

import optuna

# import catboost as ctb
import lightgbm as lgb
from xgboost import XGBClassifier, XGBRegressor

import graphviz
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter('ignore')

pd.set_option('max_columns', 100)

COLAB = True

if COLAB:
    # import sshColab
    os.chdir('/root/.kaggle/')
    json_file = 'gcs-colab.json'
    subprocess.call(f'chmod 600 /root/.kaggle/{json_file}', shell=True)        
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = f'/root/.kaggle/{json_file}' 
    subprocess.call('echo $GOOGLE_APPLICATION_CREDENTIALS', shell=True)    
else:
    subprocess.call('pip install ssh-Colab', shell=True)    
    subprocess.call('pip install google-colab', shell=True)
    import sshColab

project = "strategic-howl-305522"
bucket_name = "gcs-station-168"           
storage_client = storage.Client(project=project)
bucket = storage_client.bucket(bucket_name)

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('Time taken : %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

def current_datetime(tz="America/New_York"):
    import datetime
    import pytz
    utc_now = pytz.utc.localize(datetime.datetime.utcnow())
    pst_now = utc_now.astimezone(pytz.timezone("America/New_York"))
    return pst_now.strftime("%b-%d-%Y-at-%H-%M")
        
def rank_gauss(x):
    N = x.shape[0]
    temp = x.argsort()
    rank_x = temp.argsort() / N
    rank_x -= rank_x.mean()
    rank_x *= 2
    efi_x = erfinv(rank_x)
    efi_x -= efi_x.mean()
    return efi_x

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)    

# #L GCS upload
# project = "strategic-howl-305522" # (1)
# bucket_name = "gcs-station-168" # (2)          
# storage_client = storage.Client(project=project)
# bucket = storage_client.bucket(bucket_name)
# destination_blob_name = "tps-apr-2021-label/test.pkl" # (3) No prefix slash.
# source_file_name = "/kaggle/working/test.pkl" # (4)
# pickle.dump(all_df, open(source_file_name, 'wb'))
# blob = bucket.blob(destination_blob_name)
# blob.upload_from_filename(source_file_name)

# ANCHOR RAW DATA

TARGET = 'Survived'
os.chdir('/kaggle/working/')
train_df = pd.read_csv('../input/tabular-playground-series-apr-2021/train.csv')
test_df = pd.read_csv('../input/tabular-playground-series-apr-2021/test.csv')
submission = pd.read_csv('../input/tabular-playground-series-apr-2021/sample_submission.csv')
train_label = train_df['Survived']

#L GCS download
source_blob_name = "tps-apr-2021-label/voting_submission_from_3_best.csv" # (5) No prefix slash
destination_file_name = "/kaggle/working/voting_submission_from_3_best.csv" # (6)
blob = bucket.blob(source_blob_name)
blob.download_to_filename(destination_file_name)
# all_df = pickle.load(open(destination_file_name, 'rb'))  
test_df[TARGET] = pd.read_csv("voting_submission_from_3_best.csv")[TARGET]

all_df = pd.concat([train_df, test_df]).reset_index(drop=True)
train_rows = train_df.shape[0]


# ANCHOR DATA ZONE

all_df = pd.concat([train_df, test_df]).reset_index(drop=True)

# Age fillna with mean age for each class
all_df['Age'] = all_df['Age'].fillna(all_df['Age'].mean())

# Cabin, fillna with 'X' and take first letter
all_df['Cabin'] = all_df['Cabin'].fillna('X').map(lambda x: x[0].strip())

# Ticket, fillna with 'X', split string and take first split 
all_df['Ticket'] = all_df['Ticket'].fillna('X').map(lambda x:str(x).split()[0] if len(str(x).split()) > 1 else 'X')

# Fare, fillna with mean value
fare_map = all_df[['Fare', 'Pclass']].dropna().groupby('Pclass').median().to_dict() # 
all_df['Fare'] = all_df['Fare'].fillna(all_df['Pclass'].map(fare_map['Fare'])) # 
all_df['Fare'] = np.log1p(all_df['Fare'])

# Embarked, fillna with 'X' value
all_df['Embarked'] = all_df['Embarked'].fillna('X')

# Name, take only surnames
all_df['Name'] = all_df['Name'].map(lambda x: x.split(',')[0])

label_cols = ['Name', 'Ticket', 'Sex']
onehot_cols = ['Cabin', 'Embarked']
numerical_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

label_cols = ['Name', 'Ticket', 'Sex']
onehot_cols = ['Cabin', 'Embarked']
numerical_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

def label_encoder(c):
    le = LabelEncoder()
    return le.fit_transform(c)

scaler = StandardScaler()

TARGET = 'Survived'
onehot_encoded_df = pd.get_dummies(all_df[onehot_cols])
label_encoded_df = all_df[label_cols].apply(label_encoder)
numerical_df = pd.DataFrame(scaler.fit_transform(all_df[numerical_cols]), columns=numerical_cols)
target_df = all_df[TARGET]

all_df = pd.concat([numerical_df, label_encoded_df, onehot_encoded_df, target_df], axis=1)

# #L GCS上传
# project = "strategic-howl-305522" # (1)
# bucket_name = "gcs-station-168" # (2)          
# storage_client = storage.Client(project=project)
# bucket = storage_client.bucket(bucket_name)
# destination_blob_name = "tps-apr-2021-label/14_all_df.pkl" # (3) No prefix slash.
# source_file_name = "/kaggle/working/14_all_df.pkl" # (4)
# pickle.dump(all_df, open(source_file_name, 'wb'))
# blob = bucket.blob(destination_blob_name)
# blob.upload_from_filename(source_file_name)

# !SECTION 1ST PLACE DATA

#L GCS下传
source_blob_name = "tps-apr-2021-label/14_all_df.pkl" # (5) No prefix slash
destination_file_name = "/kaggle/working/14_all_df.pkl" # (6)
blob = bucket.blob(source_blob_name)
blob.download_to_filename(destination_file_name)
all_df = pickle.load(open(destination_file_name, 'rb'))  

#%%
# SECTION Tuning Xgboost on 1P dataset.
MODE = 'CV' # 'DEBUG' 'CV', 'PROD'

if MODE=='DEBUG':
    N_ESTIMATORS = 1
    N_SPLITS = 2
    N_REPEATS = 2
    SEED = 2021
    EARLY_STOPPING_ROUNDS = 1
    VERBOSE = 100
    TIMEOUT = 1 * 60

if MODE=='CV':
    N_ESTIMATORS = 1000
    N_SPLITS = 5 # 3, 5, 10
    N_REPEATS = 6 # 3, 5, 10, 30
    SEED = 2021
    EARLY_STOPPING_ROUNDS = 100
    VERBOSE = 100
    N_TRIALS = 500
    TIMEOUT = 3 * 60 * 60

params = {
    'objective': 'binary:logistic', # NOTE don't specify "n_classes" in binary:logistic case.
    'n_estimators': N_ESTIMATORS,
    'booster': 'gbtree',
    'verbosity': 0,
    'average': 'macro', # micro vs macro https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin
    'eval_metric': 'auc', 
    'tree_method': 'gpu_hist', 
    'use_label_encoder': False # <-------        
}

def objective(trial, x_train, y_train, params=params):
    # x_train, y_train: ndarray
    start_time = timer()    
    temp_map = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        "learning_rate": trial.suggest_loguniform("learning_rate", 5e-3, 5e-2),
        "min_child_weight": trial.suggest_loguniform("min_child_weight", 1, 300), # 5, 100
        "subsample": trial.suggest_loguniform("subsample", 0.4, 0.8),
        "colsample_bytree": trial.suggest_loguniform("colsample_bytree", 0.2, 0.8),
        "alpha": trial.suggest_loguniform("alpha", 0.01, 10.0),
        "lambda": trial.suggest_loguniform("lambda", 1e-8, 10.0),
        "gamma": trial.suggest_loguniform("lambda", 1e-8, 10.0)
    }
    params.update(temp_map)

    y_oof = np.zeros(x_train.shape[0])
    acc_scores = []    
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation_0-auc") # depends on the choice of eval_metric; "validation_0-logloss"
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    
    for i, (train_index, valid_index) in enumerate(rskf.split(x_train, y_train)):
        X_A, X_B = x_train[train_index, :], x_train[valid_index, :]
        y_A, y_B = y_train[train_index], y_train[valid_index]
        xgb_classifier = XGBClassifier(**params)
        xgb_classifier.fit(
            X_A, y_A, eval_set=[(X_B, y_B)], 
            early_stopping_rounds=EARLY_STOPPING_ROUNDS, 
            verbose=0,
            callbacks=[pruning_callback])
        best_iteration = xgb_classifier.get_booster().best_ntree_limit               # new
        y_oof[valid_index] = xgb_classifier.predict(X_B, ntree_limit=best_iteration) # new
        acc_scores.append(accuracy_score(y_B, y_oof[valid_index]))
        
    trial.set_user_attr(key="best_booster", value=xgb_classifier) # NOTE update the best model in the optuna's table.
    res = np.mean(acc_scores) 
    
    timer(start_time)
    return res 

def save_best(study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="best_booster", value=trial.user_attrs["best_booster"]) # SOURCE retrieve the best number of estimators https://github.com/optuna/optuna/issues/1169    

if MODE=='CV':
    
    X_input = all_df.drop('Survived', axis=1).iloc[:train_rows, :].values
    Y_input = train_label.iloc[:train_rows].values

    study = optuna.create_study(
        direction = "maximize", 
        sampler = optuna.samplers.TPESampler(seed=2021, multivariate=True),
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)
    )
    study.optimize(lambda trial: objective(trial, X_input, Y_input, params), 
                   n_trials=N_TRIALS, 
                   timeout=TIMEOUT, 
                   callbacks=[save_best],
                   n_jobs=1
    )
    hp = study.best_params
    for key, value in hp.items():
        print(f"{key:>20s} : {value}")
    print(f"{'best objective value':>20s} : {study.best_value}")  
    best_model=study.user_attrs["best_booster"]
    
    #L GCS上传
    curdt = current_datetime()
    project = "strategic-howl-305522" # (1)
    bucket_name = "gcs-station-168" # (2)          
    storage_client = storage.Client(project=project)
    bucket = storage_client.bucket(bucket_name)
    destination_blob_name = f"tps-apr-2021-label/13_1P_Xgboost{curdt}.pkl" # (3) No prefix slash.
    source_file_name = f"/kaggle/working/13_1P_Xgboost{curdt}.pkl" # (4)
    pickle.dump(best_model, open(source_file_name, 'wb'))
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)

"""N_SPLITS = 5, N_REPEATS = 6.
After considering best iteration for inference, we have:
           max_depth : 3
       learning_rate : 0.018207166629674813
    min_child_weight : 33.66715820475523
           subsample : 0.7801994724633023
    colsample_bytree : 0.4433999442802635
               alpha : 0.12986213675052546
              lambda : 0.0001173024800197149
best objective value : 0.7827116666666668 <--- local, kfold inference

A little differece in min_child_weight.
"""

"""N_SPLITS = 5, N_REPEATS = 6.
           max_depth : 3
       learning_rate : 0.018207166629674813
    min_child_weight : 31.70203516098613
           subsample : 0.7801994724633023
    colsample_bytree : 0.4433999442802635
               alpha : 0.12986213675052546
              lambda : 0.0001173024800197149
best objective value : 0.7827016666666667 <---Local  --> Priv: 0.79921, Pub0.80168
"""

# !SECTION Tuning Xgboost on 1P dataset.

# #L GCS下传
# source_blob_name = "tps-apr-2021-label/test.pkl" # (5) No prefix slash
# destination_file_name = "/kaggle/working/test2.pkl" # (6)
# blob = bucket.blob(source_blob_name)
# blob.download_to_filename(destination_file_name)
# all_df = pickle.load(open(destination_file_name, 'rb'))

# SECTION Inference with 1P's Xgboost model
#L GCS下传
source_blob_name = f"tps-apr-2021-label/13_1P_Xgboost{curdt}.pkl" # (5) No prefix slash
destination_file_name = f"/kaggle/working/13_1P_Xgboost{curdt}.pkl" # (6)
blob = bucket.blob(source_blob_name)
blob.download_to_filename(destination_file_name)
best_model = pickle.load(open(destination_file_name, 'rb'))  

MODE = 'CV' # 'DEBUG' 'CV', 'PROD'

if MODE=='CV':
    N_ESTIMATORS = 1000
    N_SPLITS = 5 # 3, 5, 10
    N_REPEATS = 6 # 3, 5, 10, 30
    SEED = 2021
    EARLY_STOPPING_ROUNDS = 100
    VERBOSE = 100
    N_TRIALS = 500
    TIMEOUT = 3 * 60 * 60
    TEST_SIZE = 0.1

def xgb_inference(model, X, Y, X_test):
    """
    KFold inference performs better in leaderboard, not in local CV. Be mindful!
    Use this not the one after this code block.
    """
    x = X.values
    y = Y.values
    x_tst = X_test.values
    y_tst = np.zeros((x_tst.shape[0], len(np.unique(y))))
    acc_scores = []
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, 
                                   n_repeats=N_REPEATS, 
                                   random_state=SEED)
    for i, (train_index, valid_index) in enumerate(rskf.split(x, y)):
        X_A, X_B = x[train_index, :], x[valid_index, :]
        y_A, y_B = y[train_index], y[valid_index]
        print(i)
        model.fit(
            X_A, y_A, eval_set=[(X_B, y_B)],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose=0)
        acc_score = accuracy_score(y_B, model.predict(X_B)) # Xgboost automatically fills in best iteration (you may check again in its source code and Li's stackoverflow question). So we don't need to assign best iteration as we do for lgbm.
        acc_scores.append(acc_score)
        y_tst += model.predict_proba(x_tst)
    y_tst /= N_SPLITS * N_REPEATS
    return y_tst, np.mean(acc_scores)    


X = all_df.drop('Survived', axis=1).iloc[:train_rows, :]
Y = train_label.iloc[:train_rows]
X_test = all_df.drop('Survived', axis=1).iloc[train_rows:, :]
Y_tst, mean_acc = xgb_inference(best_model, X, Y, X_test)
submission.Survived = np.where(Y_tst[:,-1] > 0.5, 1, 0)
submission.to_csv(f"/kaggle/working/submission_cv{mean_acc:.6f}_{curdt}.csv", index=False)

"""
The practice below (Non-KFold) is not as good as the previous one.
"""
def xgb_inference1(model, X, Y, X_test): 
    x = X.values
    y = Y.values
    x_tst = X_test.values
    y_tst = np.zeros((x_tst.shape[0], len(np.unique(y))))
    acc_scores = []
    X_A, X_B, y_A, y_B = train_test_split(x, y, test_size=TEST_SIZE, 
                                          shuffle=True, 
                                          random_state=SEED)
    model.fit(
        X_A, y_A, eval_set=[(X_B, y_B)],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose=0)
    best_iteration = model.get_booster().best_ntree_limit
    acc_score = accuracy_score(y_B, model.predict(X_B, ntree_limit=best_iteration))
    acc_scores.append(acc_score)
    y_tst = model.predict_proba(x_tst, ntree_limit=best_iteration)
    return y_tst, np.mean(acc_scores)       

X = all_df.drop('Survived', axis=1).iloc[:train_rows, :]
Y = train_label.iloc[:train_rows]
X_test = all_df.drop('Survived', axis=1).iloc[train_rows:, :]
Y_tst, mean_acc = xgb_inference1(best_model, X, Y, X_test)
submission.Survived = np.where(Y_tst[:,-1] > 0.5, 1, 0)
submission.to_csv(f"/kaggle/working/submission_oneshot_cv{mean_acc:.6f}_{curdt}.csv", index=False)

""" 
Note: This section uses simple strategy to process raw data, no advanced imputation. 
Note: 1st place winner solution's dataset.      CV        private  public 
                  Use KFold in inference phase: 0.782712, 0.79919, 0.80172 (good. lower CV, but higher private and public)
No KFold but a single split in inference phase: 0.783700, 0.79852, 0.80051 (bad. higher CV, but lower private and public)
"""

# !SECTION Inference with 1P's Xgboost model

# SECTION 1P's Xgboost feature importances
# ANCHOR FEATURE IMPORTANCE
feature_important = best_model.get_booster().get_score(importance_type='weight')
tmp = pd.Series(list(feature_important.keys())).map(lambda x: x[1:]).astype(int).tolist() # remove 'f' to get column indices.
cols = [df.columns.tolist()[i] for i in tmp] # rearrange columns
keys = cols
values = list(feature_important.values())
edata = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=True)
edata.plot(kind='barh') # screenshot https://i.postimg.cc/vHwz5Rmv/2021-05-07-at-15-33-17.png

# ANCHOR PERMUTATION IMPORTANCE https://mljar.com/blog/feature-importance-xgboost/
from sklearn.inspection import permutation_importance
perm_importance = permutation_importance(best_model, X, Y)
sorted_idx = perm_importance.importances_mean.argsort()
plt.barh(np.array(all_df.drop('Survived', axis=1).columns.tolist())[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance") # screenshot https://i.postimg.cc/sf6ndG6M/2021-05-07-at-15-32-54.png

# ANCHOR PERFMUATION IMPORATNCE IS GOOD AS LONG AS FEATURES ARE NOT HIGHLY CORRELATED. https://mljar.com/blog/feature-importance-xgboost/
def correlation_heatmap(train):
    correlations = train.corr()
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f', cmap="YlGnBu",
                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70}
                )
    plt.tight_layout()
    plt.show();
    
correlation_heatmap(X)

# ANCHOR import shap
import shap
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X, plot_type="bar") # sreccnshot https://i.postimg.cc/wvzYRfn4/2021-05-07-at-15-44-00.png

# !SECTION 1P's Xgboost feature importances

# SECTION Tuning Xgboost on my imputed data (not good model compared with 1P's model. see below)
#L GCS下传 xgbclf
source_blob_name = "tps-apr-2021-label/13xgboostMay-05-2021-at-23-28.pkl" # (5) No prefix slash
destination_file_name = "/kaggle/working/13xgboostMay-05-2021-at-23-28.pkl" # (6)
blob = bucket.blob(source_blob_name)
blob.download_to_filename(destination_file_name)
xgbclf = pickle.load(open(destination_file_name, 'rb')) 

#L GCS下传 train_df, test_df, train_label, train_id, test_id, train_rows
source_blob_name = "tps-apr-2021-label/13DfLabelIDRows.pkl" # (5) No prefix slash
destination_file_name = "/kaggle/working/13DfLabelIDRows.pkl" # (6)
blob = bucket.blob(source_blob_name)
blob.download_to_filename(destination_file_name)
train_df, test_df, train_label, train_id, test_id, train_rows = pickle.load(open(destination_file_name, 'rb')) 

#L GCS下传 df
source_blob_name = "tps-apr-2021-label/13df.pkl" # (5) No prefix slash
destination_file_name = "/kaggle/working/13df.pkl " # (6)
blob = bucket.blob(source_blob_name)
blob.download_to_filename(destination_file_name)
df = pickle.load(open(destination_file_name, 'rb')) 

MODE = 'CV' # 'DEBUG' 'CV', 'PROD'

if MODE=='DEBUG':
    N_ESTIMATORS = 1
    N_SPLITS = 2
    N_REPEATS = 2
    SEED = 2021
    EARLY_STOPPING_ROUNDS = 1
    VERBOSE = 100
    TIMEOUT = 1 * 60

if MODE=='CV':
    N_ESTIMATORS = 1000
    N_SPLITS = 5 # 3, 5, 10
    N_REPEATS = 6 # 3, 5, 10, 30
    SEED = 2021
    EARLY_STOPPING_ROUNDS = 100
    VERBOSE = 100
    N_TRIALS = 1000
    TIMEOUT = 3 * 60 * 60

params = {
    'objective': 'binary:logistic', # NOTE don't specify "n_classes" in binary:logistic case.
    'n_estimators': N_ESTIMATORS,
    'booster': 'gbtree',
    'verbosity': 0,
    'average': 'macro', # micro vs macro https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin
    'eval_metric': 'auc', 
    'tree_method': 'gpu_hist', 
    'use_label_encoder': False # <-------        
}

def objective(trial, x_train, y_train, params=params):
    # x_train, y_train: ndarray
    start_time = timer()    
    temp_map = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        "learning_rate": trial.suggest_loguniform("learning_rate", 5e-3, 5e-2),
        "min_child_weight": trial.suggest_loguniform("min_child_weight", 5, 100),
        "subsample": trial.suggest_loguniform("subsample", 0.4, 0.8),
        "colsample_bytree": trial.suggest_loguniform("colsample_bytree", 0.2, 0.8),
        "alpha": trial.suggest_loguniform("alpha", 0.01, 10.0),
        "lambda": trial.suggest_loguniform("lambda", 1e-8, 10.0),
        "gamma": trial.suggest_loguniform("lambda", 1e-8, 10.0)
    }
    params.update(temp_map)

    y_oof = np.zeros(x_train.shape[0])
    acc_scores = []    
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation_0-auc") # depends on the choice of eval_metric; "validation_0-logloss"
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    
    for i, (train_index, valid_index) in enumerate(rskf.split(x_train, y_train)):
        X_A, X_B = x_train[train_index, :], x_train[valid_index, :]
        y_A, y_B = y_train[train_index], y_train[valid_index]
        xgb_classifier = XGBClassifier(**params)
        xgb_classifier.fit(
            X_A, y_A, eval_set=[(X_B, y_B)], 
            early_stopping_rounds=EARLY_STOPPING_ROUNDS, 
            verbose=0,
            callbacks=[pruning_callback])
        y_oof[valid_index] = xgb_classifier.predict(X_B)
        acc_scores.append(accuracy_score(y_B, y_oof[valid_index]))
        
    trial.set_user_attr(key="best_booster", value=xgb_classifier) # NOTE update the best model in the optuna's table.
    res = np.mean(acc_scores) 
    
    timer(start_time)
    return res 

def save_best(study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="best_booster", value=trial.user_attrs["best_booster"]) # SOURCE retrieve the best number of estimators https://github.com/optuna/optuna/issues/1169    

def current_datetime(tz="America/New_York"):
    import datetime
    import pytz
    utc_now = pytz.utc.localize(datetime.datetime.utcnow())
    pst_now = utc_now.astimezone(pytz.timezone("America/New_York"))
    return pst_now.strftime("%b-%d-%Y-at-%H-%M")

if MODE=='CV':
    
    X_input = df.iloc[:train_rows, :].values
    Y_input = train_label.iloc[:train_rows].values

    study = optuna.create_study(
        direction = "maximize", 
        sampler = optuna.samplers.TPESampler(seed=2021, multivariate=True),
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)
    )
    study.optimize(lambda trial: objective(trial, X_input, Y_input, params), 
                   n_trials=N_TRIALS, 
                   timeout=TIMEOUT, 
                   callbacks=[save_best],
                   n_jobs=1
    )
    hp = study.best_params
    for key, value in hp.items():
        print(f"{key:>20s} : {value}")
    print(f"{'best objective value':>20s} : {study.best_value}")  
    best_model=study.user_attrs["best_booster"]
    
    #L GCS上传
    curdt = current_datetime()
    project = "strategic-howl-305522" # (1)
    bucket_name = "gcs-station-168" # (2)          
    storage_client = storage.Client(project=project)
    bucket = storage_client.bucket(bucket_name)
    destination_blob_name = f"tps-apr-2021-label/13_my_Xgboost{curdt}.pkl" # (3) No prefix slash.
    source_file_name = f"/kaggle/working/13_my_Xgboost{curdt}.pkl" # (4)
    pickle.dump(best_model, open(source_file_name, 'wb'))
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)

    #L GCS下传
source_blob_name = f"tps-apr-2021-label/13_my_Xgboost{curdt}.pkl" # (5) No prefix slash
destination_file_name = f"/kaggle/working/13_my_Xgboost{curdt}.pkl" # (6)
blob = bucket.blob(source_blob_name)
blob.download_to_filename(destination_file_name)
best_model = pickle.load(open(destination_file_name, 'rb'))  

MODE = 'CV' # 'DEBUG' 'CV', 'PROD'

if MODE=='DEBUG':
    N_ESTIMATORS = 1
    N_SPLITS = 2
    N_REPEATS = 2
    SEED = 2021
    EARLY_STOPPING_ROUNDS = 1
    VERBOSE = 100
    TIMEOUT = 1 * 60

if MODE=='CV':
    N_ESTIMATORS = 1000
    N_SPLITS = 5 # 3, 5, 10
    N_REPEATS = 6 # 3, 5, 10, 30
    SEED = 2021
    EARLY_STOPPING_ROUNDS = 100
    VERBOSE = 100
    N_TRIALS = 1000
    TIMEOUT = 3 * 60 * 60

def xgb_inference(model, X, Y, X_test):
    x = X.values
    y = Y.values
    x_tst = X_test.values
    y_tst = np.zeros((x_tst.shape[0], len(np.unique(y))))
    acc_scores = []
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, 
                                   n_repeats=N_REPEATS, 
                                   random_state=SEED)
    for i, (train_index, valid_index) in enumerate(rskf.split(x, y)):
        X_A, X_B = x[train_index, :], x[valid_index, :]
        y_A, y_B = y[train_index], y[valid_index]
        model.fit(
            X_A, y_A, eval_set=[(X_B, y_B)],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose=0)
        acc_score = accuracy_score(y_B, model.predict(X_B))
        acc_scores.append(acc_score)
        y_tst += model.predict_proba(x_tst)
    y_tst /= N_SPLITS * N_REPEATS
    return y_tst, np.mean(acc_scores)    

X = df.iloc[:train_rows, :]
Y = train_label.iloc[:train_rows]
X_test = df.iloc[train_rows:, :]
Y_tst, mean_acc = xgb_inference(best_model, X, Y, X_test)
submission.Survived = np.where(Y_tst[:,-1] > 0.5, 1, 0)
submission.to_csv(f"/kaggle/working/my_xgb_submission_cv{mean_acc:.6f}.csv", index=False)

# GCS upload
project = "strategic-howl-305522" # (1)
bucket_name = "gcs-station-168" # (2)          
storage_client = storage.Client(project=project)
bucket = storage_client.bucket(bucket_name)
destination_blob_name = f"tps-apr-2021-label/my_xgb_submission_cv{mean_acc:.6f}.csv" # (3) No prefix slash.
source_file_name = f"/kaggle/working/my_xgb_submission_cv{mean_acc:.6f}.csv" # (4)
# pickle.dump(best_model, open(source_file_name, 'wb'))
blob = bucket.blob(destination_blob_name)
blob.upload_from_filename(source_file_name)

"""
13_my_XgboostMay-08-2021-at-01-45.pkl
           max_depth : 7
       learning_rate : 0.010266267475345451
    min_child_weight : 51.15578915995479
           subsample : 0.6358592643365711
    colsample_bytree : 0.7997864635122331
               alpha : 0.22432657083910007
              lambda : 1.5054250976430413e-06
best objective value : 0.7844066666666667 LC <-- --> Priv: 0.79145, Pub: 0.79409

13xgboostMay-05-2021-at-13-55.pkl
           max_depth : 9
       learning_rate : 0.025813617122660452
    min_child_weight : 50.60951510989578
           subsample : 0.793455994642076
    colsample_bytree : 0.793643019233612
               alpha : 0.10857898922910038
              lambda : 0.02340486141678862
best objective value : 0.7851600000000001
"""
# !SECTION comparison xgboost on imputed dataset   


