#%%
import subprocess, os, datetime, pickle, gc
from google.cloud import storage
from datetime import datetime
import numpy as np
import pandas as pd
import warnings
from scipy.special import erfinv
from matplotlib import pyplot as plt
import optuna
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

warnings.simplefilter('ignore')
plt.style.use('fivethirtyeight')
pd.set_option('max_columns', 100)

COLAB = True

if COLAB:
    COLAB = True
    import sshColab
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

train_df = pd.read_csv('/kaggle/input/tabular-playground-series-apr-2021/train.csv')
test_df = pd.read_csv('/kaggle/input/tabular-playground-series-apr-2021/test.csv')

train_label = train_df['Survived']
train_id = train_df['PassengerId']
test_id = test_df['PassengerId']
del train_df['Survived'], train_df['PassengerId']
del test_df['PassengerId']

train_rows = train_df.shape[0]

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('Time taken : %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


file_to_load = '11dataframe_xgboost_based_trim.pkl'
sshColab.download_to_colab(project, bucket_name, 
    destination_directory = '/kaggle/working', 
    remote_blob_path=f'tps-apr-2021-label/{file_to_load}', 
    local_file_name=file_to_load) 
df = pickle.load(open(f'/kaggle/working/{file_to_load}', 'rb'))

file_to_load = '11cols_tuple.pkl'
sshColab.download_to_colab(project, bucket_name, 
    destination_directory = '/kaggle/working', 
    remote_blob_path=f'tps-apr-2021-label/{file_to_load}', 
    local_file_name=file_to_load) 
cat_cols, num_cols = pickle.load(open(f'/kaggle/working/{file_to_load}', 'rb'))

def feature_distribution():
    plt.figure(figsize=(16, 32))
    for i, col in enumerate(df.columns.tolist()):
        ax = plt.subplot(10, 2, i + 1)
        ax.set_title(col)
        df[col].hist(bins=50)
feature_distribution()

df.info()
# %%
# SECTION No pseudolabelling baseline

N_SPLITS = 5
N_REPEATS = 1
EARLY_STOPPING_ROUNDS = 50
N_TRIALS = 1500
TIMEOUT = 3 * 60 * 60 # <------------

params = {
    'objective': 'binary:logistic', # NOTE don't specify "n_classes" in binary:logistic case.
    'n_estimators': 700,
    'booster': 'gbtree',
    'verbosity': 0,
    'average': 'macro',
    'eval_metric': 'auc', 
    'tree_method': 'gpu_hist', 
    'use_label_encoder': False # <------- do it yourself beforehand!       
}

def objective(trial, x_train, y_train, params=params):
    start_time = timer()
    temp_map = {
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.005, 0.05),
        "min_child_weight": trial.suggest_loguniform("min_child_weight", 50, 1000),
        "subsample": trial.suggest_loguniform("subsample", 0.4, 0.8),
        "colsample_bytree": trial.suggest_loguniform("colsample_bytree", 0.2, 0.8),
        "alpha": trial.suggest_loguniform("alpha", 0.01, 10.0),
        "lambda": trial.suggest_loguniform("lambda", 1e-8, 10.0),
        "gamma": trial.suggest_loguniform("lambda", 1e-8, 10.0)
    }
    params.update(temp_map)
    # x_train = df.iloc[:train_rows, :].values
    # y_train = train_label.iloc[:train_rows].values
    y_oof = np.zeros((x_train.shape[0],2))
    acc_scores = []    
    xgb_classifier = XGBClassifier(**params)
    # pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation_0-logloss")
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation_0-auc")
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS)
    for i, (train_index, valid_index) in enumerate(rskf.split(x_train, y_train)):
        X_A, X_B = x_train[train_index, :], x_train[valid_index, :]
        y_A, y_B = y_train[train_index], y_train[valid_index]
        xgb_classifier.fit(
            X_A, y_A, eval_set=[(X_B, y_B)], 
            early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=0,
            callbacks=[pruning_callback]
        )
        y_oof[valid_index] = xgb_classifier.predict_proba(X_B)
        acc_score = accuracy_score(y_B, np.where(y_oof[valid_index]>0.5, 1, 0).argmax(axis=1))
        acc_scores.append(acc_score)
        print(f"===== {i} fold : acc {acc_score} =====")
    trial.set_user_attr(key="best_booster", value=xgb_classifier) # NOTE update the best model in the optuna's table.
    res = np.mean(acc_scores) 
    print(f"***** {res} *****")
    timer(start_time)
    return res 

def save_best(study, trial):
    if study.best_trial.number == trial.number:
        # Set the best booster as a trial attribute; accessible via study.trials_dataframe.
        study.set_user_attr(key="best_booster", value=trial.user_attrs["best_booster"])
        # SOURCE retrieve the best number of estimators https://github.com/optuna/optuna/issues/1169    

study = optuna.create_study(
    direction = "maximize", 
    sampler = optuna.samplers.TPESampler(seed=2021, multivariate=True),
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)
)
study.optimize(lambda trial: objective(trial, 
                                       df.iloc[:train_rows, :].values, 
                                       train_label.iloc[:train_rows].values, params), 
               n_trials=N_TRIALS, 
               timeout=TIMEOUT, 
               n_jobs=1, 
               callbacks=[save_best]
)

hp = study.best_params
for key, value in hp.items():
    print(f"{key:>20s} : {value}")
print(f"{'best objective value':>20s} : {study.best_value}")  
best_model=study.user_attrs["best_booster"]

file = '11xgb_OnImputedDf_NoPseudoLabel.pkl'
os.chdir('/kaggle/working')
pickle.dump(best_model, open(f'/kaggle/working/{file}', 'wb'))
sshColab.upload_to_gcs(project, bucket_name, f'tps-apr-2021-label/{file}', f'/kaggle/working/{file}')

"""
           max_depth : 11
       learning_rate : 0.02139025492127512
    min_child_weight : 53.72875543695861
           subsample : 0.7865988900180777
    colsample_bytree : 0.7564617384692738
               alpha : 1.5277757465673043
              lambda : 2.6624091861803207
best objective value : 0.78316
"""
# !SECTION No pseudolabelling baseline