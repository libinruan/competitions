#%%
# SECTION 1ST PLACE DATA
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

#L GCS上传
project = "strategic-howl-305522" # (1)
bucket_name = "gcs-station-168" # (2)          
storage_client = storage.Client(project=project)
bucket = storage_client.bucket(bucket_name)
destination_blob_name = "tps-apr-2021-label/14_all_df.pkl" # (3) No prefix slash.
source_file_name = "/kaggle/working/14_all_df.pkl" # (4)
pickle.dump(all_df, open(source_file_name, 'wb'))
blob = bucket.blob(destination_blob_name)
blob.upload_from_filename(source_file_name)

# !SECTION 1ST PLACE DATA

# SECTION CROSS_VAL_SCORE METHOD 1-A
#L GCS下传
source_blob_name = "tps-apr-2021-label/14_all_df.pkl" # (5) No prefix slash
destination_file_name = "/kaggle/working/14_all_df.pkl" # (6)
blob = bucket.blob(source_blob_name)
blob.download_to_filename(destination_file_name)
all_df = pickle.load(open(destination_file_name, 'rb'))  

MODE = 'DEBUG' # 'DEBUG' 'CV', 'PROD'

if MODE=='DEBUG':
    N_ESTIMATORS = 1
    N_SPLITS = 2
    N_REPEATS = 1
    N_ITERS = 2
    SEED = 2021
    EARLY_STOPPING_ROUNDS = 1
    VERBOSE = 100
    TIMEOUT = 1 * 60

if MODE=='CV':
    N_ESTIMATORS = 1000
    N_SPLITS = 5 # 10
    N_REPEATS = 1
    N_ITERS = 2 # 30
    SEED = 2021
    EARLY_STOPPING_ROUNDS = 100
    VERBOSE = 100
    N_TRIALS = 1500
    TIMEOUT = 3 * 60 * 60

X = all_df.drop('Survived', axis=1).iloc[:train_rows] 
Y = train_label.iloc[:train_rows]

# Model 1-a. cross_val_score 
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc'
}
model = XGBClassifier(**params)
kfold = StratifiedKFold(n_splits=N_SPLITS, random_state=SEED)
results = cross_val_score(model, X, Y, scoring='accuracy', cv=kfold) # only a single metric is permitted. model is cloned not relay across folds.
print(f'Accuracy: {results.mean()*100:.4f}% ({results.std()*100:.3f})')

# !SECTION CROSS_VAL_SCORE

# SECTION SKLEARN API MODEL 1-B PREDICT, NOT PREDICT_PROBA

x_train = all_df.drop('Survived', axis=1).iloc[:train_rows].values 
y_train = train_label.iloc[:train_rows].values
y_oof = np.zeros(x_train.shape[0])
acc_scores = []
kfold = StratifiedKFold(n_splits=N_SPLITS, random_state=SEED)
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc'
}
model = XGBClassifier(**params)
for i, (train_index, valid_index) in enumerate(kfold.split(x_train, y_train)):
    X_A, X_B = x_train[train_index, :], x_train[valid_index, :]
    y_A, y_B = y_train[train_index], y_train[valid_index]
    model.fit(X_A, y_A, eval_set=[(X_B, y_B)], verbose=0)
    y_oof[valid_index] = model.predict(X_B)
    acc_scores.append(accuracy_score(y_B, y_oof[valid_index]))

print(f'Accuracy (manual): {np.mean(acc_scores)*100:.4f}% ({np.std(acc_scores)*100:.3f})')
# the same as cross_val_score

# !SECTION SKLEARN API MODEL 1-B

# SECTION SKLEARN API MODEL 1-C PREDICT_PROBA
# Model 1-c. Sklearn API with predict_proba.
x_train = all_df.drop('Survived', axis=1).iloc[:train_rows].values 
y_train = train_label.iloc[:train_rows].values
y_oof = np.zeros(x_train.shape[0])
acc_scores = []
kfold = StratifiedKFold(n_splits=N_SPLITS, random_state=SEED)
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc'
}
model = XGBClassifier(**params)
for i, (train_index, valid_index) in enumerate(kfold.split(x_train, y_train)):
    X_A, X_B = x_train[train_index, :], x_train[valid_index, :]
    y_A, y_B = y_train[train_index], y_train[valid_index]
    model.fit(X_A, y_A, eval_set=[(X_B, y_B)])
    y_oof[valid_index] = np.where(model.predict_proba(X_B)[:,-1] > 0.5, 1, 0)
    acc_scores.append(accuracy_score(y_B, y_oof[valid_index]))

print(f'Accuracy (manual): {np.mean(acc_scores)*100:.4f}% ({np.std(acc_scores)*100:.3f})')
# the same as cross_val_score
# !SECTION SKLEARN API MODEL 1-C PREDICT_PROBA

# SECTION MODEL 1-D LEARNING API USING BINARY:LOGISTIC 
# Model 1-d. Learning API using using binary:logistic
import xgboost as xgb
x_train = all_df.drop('Survived', axis=1).iloc[:train_rows].values 
y_train = train_label.iloc[:train_rows].values
y_oof = np.zeros((x_train.shape[0]))
acc_scores = []
kfold = StratifiedKFold(n_splits=N_SPLITS, random_state=SEED)
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc'
}
for i, (train_index, valid_index) in enumerate(kfold.split(x_train, y_train)):
    model = xgb
    X_A, X_B = x_train[train_index, :], x_train[valid_index, :]
    y_A, y_B = y_train[train_index], y_train[valid_index]
    dtrain = xgb.DMatrix(X_A, label=y_A)
    dvalid = xgb.DMatrix(X_B, label=y_B)
    evallist = [(dvalid, 'eval')]
    trainedModel = model.train(params, dtrain=dtrain, evals=evallist, num_boost_round=100, verbose_eval=False) # verbose_eval=False
    y_oof[valid_index] = trainedModel.predict(xgb.DMatrix(X_B))
    acc_scores.append(accuracy_score(y_B, np.where(y_oof[valid_index]>0.5, 1, 0)))

print(f'Accuracy (manual): {np.mean(acc_scores)*100:.4f}% ({np.std(acc_scores)*100:.3f})')
# Accuracy (manual): 77.5640% (0.018) the same as 1-a ~ 1-c.
# !SECTION LEARNING API USING BINARY:LOGISTIC 

"""
MODEL 1-A ~ 1-D ALL FOUR GENEATE IDENTICAL RESULTS!!
"""

# SECTION MODEL 1-E MULTI:SOFTPROB
# construction binary classes using multi:softprob (softmax similar)
import xgboost as xgb
x_train = all_df.drop('Survived', axis=1).iloc[:train_rows].values 
y_train = train_label.iloc[:train_rows].values
y_oof = np.zeros((x_train.shape[0], train_label.nunique()))
acc_scores = []
kfold = StratifiedKFold(n_splits=N_SPLITS, random_state=SEED)
params = {
    'objective': 'multi:softprob', # output probability
    'eval_metric': 'auc',
    'num_class': 2 # you need this parameter when working on multiple classes.
}
for i, (train_index, valid_index) in enumerate(kfold.split(x_train, y_train)):
    model = xgb
    X_A, X_B = x_train[train_index, :], x_train[valid_index, :]
    y_A, y_B = y_train[train_index], y_train[valid_index]
    dtrain = xgb.DMatrix(X_A, label=y_A)
    dvalid = xgb.DMatrix(X_B, label=y_B)
    evallist = [(dvalid, 'eval')]
    trainedModel = model.train(params, dtrain=dtrain, evals=evallist, num_boost_round=100, verbose_eval=False)
    y_oof[valid_index] = trainedModel.predict(xgb.DMatrix(X_B))
    acc_scores.append(accuracy_score(y_B, np.argmax(y_oof[valid_index], axis=1)))
y_oof
print(f'Accuracy (manual): {np.mean(acc_scores)*100:.4f}% ({np.std(acc_scores)*100:.3f})')
# !SECTION MODEL 1-E MULTI:SOFTPROB

# SECTION MODEL 1-F MULTI:SOFTMAX
# construction binary classes using multi:softprob (softmax similar)
import xgboost as xgb
x_train = all_df.drop('Survived', axis=1).iloc[:train_rows].values 
y_train = train_label.iloc[:train_rows].values
y_oof = np.zeros(x_train.shape[0])
acc_scores = []
kfold = StratifiedKFold(n_splits=N_SPLITS, random_state=SEED)
params = {
    'objective': 'multi:softmax', # output probability
    'eval_metric': 'auc',
    'num_class': 2 # you need this parameter when working on multiple classes.
}
for i, (train_index, valid_index) in enumerate(kfold.split(x_train, y_train)):
    model = xgb
    X_A, X_B = x_train[train_index, :], x_train[valid_index, :]
    y_A, y_B = y_train[train_index], y_train[valid_index]
    dtrain = xgb.DMatrix(X_A, label=y_A)
    dvalid = xgb.DMatrix(X_B, label=y_B)
    evallist = [(dvalid, 'eval')]
    trainedModel = model.train(params, dtrain=dtrain, evals=evallist, num_boost_round=100, verbose_eval=False)
    y_oof[valid_index] = trainedModel.predict(xgb.DMatrix(X_B))
    acc_scores.append(accuracy_score(y_B, y_oof[valid_index]))
y_oof
print(f'Accuracy (manual): {np.mean(acc_scores)*100:.4f}% ({np.std(acc_scores)*100:.3f})')
# !SECTION MODEL 1-F MULTI:SOFTMAX

"""
MODEL 1-E AND 1-F SHARED THE SAME RESULTS, BUT DIFFERENT FROM THOSE GENERATED BY 1-A, 1-B, 1-C, AND 1-D.
Accuracy (manual): 77.4750% (0.029)
"""
