# ANCHOR packages
import os, subprocess, sys, gc
from google.cloud import storage
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.special import erfinv
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from scipy import sparse as sp
from tqdm import tqdm

import optuna

INTELEX = True

if INTELEX:
    import site
    sys.path.append(os.path.join(os.path.dirname(site.getsitepackages()[0]), "site-packages")) # from sklearnex import patch_sklearn
    from sklearnex import patch_sklearn
    patch_sklearn()

COLAB = False

if COLAB:
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

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('Time taken : %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

def rank_gauss(x):
    N = x.shape[0]
    temp = x.argsort()
    rank_x = temp.argsort() / N
    rank_x -= rank_x.mean()
    rank_x *= 2
    efi_x = erfinv(rank_x)
    efi_x -= efi_x.mean()
    return efi_x

train_df = pd.read_csv('/kaggle/input/tabular-playground-series-apr-2021/train.csv')
test_df = pd.read_csv('/kaggle/input/tabular-playground-series-apr-2021/test.csv')

train_label = train_df['Survived']
train_id = train_df['PassengerId']
test_id = test_df['PassengerId']
del train_df['Survived'], train_df['PassengerId']
del test_df['PassengerId']

train_rows = train_df.shape[0]

sshColab.download_to_colab(project, bucket_name, 
    destination_directory = '/kaggle/working', 
    remote_blob_path='tps-apr-2021-label/1clean_data.pkl', 
    local_file_name='1clean_data.pkl') 
data = pickle.load(open('/kaggle/working/1clean_data.pkl', 'rb'))

sshColab.download_to_colab(project, bucket_name, 
    destination_directory = '/kaggle/working', 
    remote_blob_path='tps-apr-2021-label/2missing_code_map.pkl', 
    local_file_name='2missing_code_map.pkl') 
missing_code_map = pickle.load(open('/kaggle/working/2missing_code_map.pkl', 'rb'))

sshColab.download_to_colab(project, bucket_name, 
    destination_directory = '/kaggle/working', 
    remote_blob_path='tps-apr-2021-label/6imputed_df.pkl', 
    local_file_name='6imputed_df.pkl') 
imputed_df = pickle.load(open('/kaggle/working/6imputed_df.pkl', 'rb'))

cols_to_drop = ['Age', 'Fare', 'Embarked_enc'] + \
    [col for col in data.columns.tolist() if col.startswith('Cab')] + \
    [col for col in data.columns.tolist() if col.startswith('Tick')]   

df = data.drop(cols_to_drop, axis=1).join(imputed_df)
df['Embarked_enc_imp'] = df['Embarked_enc_imp'].astype(int)

df = df.rename(lambda x: x.replace('_imp', ''), axis=1)
df = df.rename(lambda x: x.replace('_enc', ''), axis=1)

# Alternatively
# df.columns = list(map(lambda x: x.replace('_imp', ''), df.columns.tolist()))
# df.columns = list(map(lambda x: x.replace('_enc', ''), df.columns.tolist()))

df['Age'] = rank_gauss(df['Age'].values)
df['Fare'] = rank_gauss(df['Fare'].values)

# ANCHOR load pseudo label from GCS
#%% Load pseudo label from https://www.kaggle.com/napetrov/tps04-svm-with-scikit-learn-intelex
file_to_load = '10-tps-apr-simple-ensemble-submission.csv'
sshColab.download_to_colab(project, bucket_name, 
    destination_directory = '/kaggle/working', 
    remote_blob_path=f'tps-apr-2021-label/{file_to_load}', 
    local_file_name=file_to_load) 
test_pseudo_label = pd.read_csv((f'/kaggle/working/{file_to_load}'))

TARGET = 'Survived'

# ANCHOR pseudo label
df.loc[train_rows:, TARGET] = test_pseudo_label.loc[:, TARGET].values # NOTE IF you encounter "only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices", convert the RHS into np.numpy by "".values".
df.loc[:train_rows-1, TARGET] = train_label.values
target_df = df[TARGET].copy()
features_df = df.drop(TARGET, axis=1)

#%% ANCHOR grouping columns
def check_col_nunique(df):
    tab = dict()
    for col in df.columns:
        tab[col] = df[col].nunique()
    return pd.Series(tab)

CARDINALITY_THRESHOLD = 10
cardinality_ss = check_col_nunique(df.drop(TARGET, axis=1))
mask = np.array(cardinality_ss) > CARDINALITY_THRESHOLD
label_cols = [col for col in cardinality_ss[mask].index.tolist() if df[col].dtype != 'float']
onehot_cols = [col for col in df.columns.tolist() if col not in label_cols and df[col].dtype != 'float']
float_cols = [col for col in df.columns.tolist() if df[col].dtype == 'float' and not col in ['Survived']]

#%% ANCHOR data processing
# int_features = df.select_dtypes(include=['int8', 'int64']).columns.tolist()
# num_features = df.select_dtypes(include=['float']).columns.tolist()
onehot_processor = Pipeline(steps=[
    ('oh', OneHotEncoder())
])
float_processor = Pipeline(steps=[
    ('ss', StandardScaler())
])
preprocessor = ColumnTransformer(transformers=[
    ('onehot', onehot_processor, onehot_cols),
    ('normal', float_processor, float_cols)
], remainder='passthrough')

pipeline = Pipeline(steps=[
    ('pp', preprocessor),
    ('ss', StandardScaler())
])
features_ray_scaled = pipeline.fit_transform(features_df)

X = features_ray_scaled
y = target_df.values

# ANCHOR toy model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
N_TRIALS = 300
TIMEOUT = 9 * 60 * 60
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.20, random_state = 12345)
params = {
    'shrinking': True,
    'max_iter': 500,
    'probability': False,
    'kernel': 'rbf',
    'verbose': True
    
}
from sklearn.svm import SVC
def objective(trial, X_train, X_valid, y_train, y_valid, params=params):
    temp_map = {
        'C': trial.suggest_loguniform('C', 1e-2, 1e3),
        # 'kernel': trial.suggest_categorical("kernel", ["rbf"]),
        'gamma': trial.suggest_loguniform('gamma', 1e-2, 1e3)
    }
    params.update(temp_map)
    svc = SVC(**params)
    svc.fit(X_train, y_train)
    return svc.score(X_valid, y_valid)

study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=2021),
                            direction="maximize",
                            pruner=optuna.pruners.MedianPruner())
study.optimize(lambda trial: objective(trial, X_train, X_valid, y_train, y_valid), n_trials=N_TRIALS, timeout=TIMEOUT, show_progress_bar=True)


pickle_to_save = '11svcNoPCA9hrsStudy.pkl'
os.chdir('/kaggle/working')
pickle.dump(study, open(f'{pickle_to_save}', 'wb'))
sshColab.upload_to_gcs(project, bucket_name, f'tps-apr-2021-label/{pickle_to_save}', f'/kaggle/working/{pickle_to_save}')

# ANCHOR result
"""
[LibSVM][I 2021-04-29 09:22:47,054] Trial 111 finished with value: 0.881875 and parameters: {'C': 11.247142842964934, 'gamma': 0.013528911812139915}. Best is trial 67 with value: 0.882425.
"""