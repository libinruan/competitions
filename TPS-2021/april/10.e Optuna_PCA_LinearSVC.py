#%%
# ANCHOR packages
import os, subprocess, sys
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

COLAB = True

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

int_features = df.select_dtypes(include=['int8', 'int64']).columns.tolist()
num_features = df.select_dtypes(include=['float']).columns.tolist()
int_processor = Pipeline(steps=[
    ('oh', OneHotEncoder())
])
float_processor = Pipeline(steps=[
    ('ss', StandardScaler())
])
preprocessor = ColumnTransformer(transformers=[
    ('integer', int_processor, int_features),
    ('float', float_processor, num_features)
])

#%% tps-apr-2021-label/7x_train_pca60_0D806352.pkl
# ANCHOR svm
# gs://gcs-station-168/tps-apr-2021-label/8df_pca400_0d8937398233439501.pkl
# gs://gcs-station-168/tps-apr-2021-label/8df_pca60_0d8063714390260596.pkl
# gs://gcs-station-168/tps-apr-2021-label/8df_pca1000_0d926987292065407.pkl
pickle_to_load = '8df_pca1000_0d926987292065407.pkl'
def GCStoLocal(filename):
    sshColab.download_to_colab(project, bucket_name, 
        destination_directory = '/kaggle/working', 
        remote_blob_path=f'tps-apr-2021-label/{filename}', 
        local_file_name=f'{filename}') 
    return pickle.load(open(f'/kaggle/working/{filename}', 'rb'))
x_pca_transformed = GCStoLocal(pickle_to_load)
# gs://gcs-station-168/tps-apr-2021-label/8df_pca60_0d8063714390260596.pkl

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
# NOTE SVM is suitable for small- or medium-sized datasets.
# NOTE Use soft margin classification to reduce the sensitivity to outliers in SVM.
# Check Hands-On page 154. 

# svm_clf = Pipeline([
#     ('scaler', StandardScaler()),
#     ("linear_svc", LinearSVC(loss='hinge', C=0.001))
# ])

ss = StandardScaler()
x_pca_transformed = ss.fit_transform(x_pca_transformed)


# ANCHOR toy svc model
def toy_svc_model():
    x_train = x_pca_transformed[:train_rows, :]
    y_train = train_label.iloc[:train_rows].values
    y_oof = np.zeros(x_train.shape[0])
    acc_scores = []
    # NOTE SVC api - https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    params = {
        'kernel': 'rbf',
        'gamma': 5,
        'C': 0.001,
        'shrinking': True,
        'probability': False, 
        'cache_size': 200, # NOTE issue with a large cache_size - https://github.com/scikit-learn/scikit-learn/issues/8012
        'max_iter': 1000
    }
    svm_clf = SVC(**params)
    svm_timer = timer()
    y_pred = svm_clf.fit(x_train, y_train)
    timer(svm_timer) # NOTE one minute per round for the LinearSVC class; The SVC class is too slow.

# toy_svc_model()    

# %%
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

N_SPLITS = 3
N_REPEATS = 1
# EARLY_STOPPING_ROUNDS = 50
N_TRIALS = 1500
TIMEOUT = 6 * 60 * 60

params = {
    'loss': 'l2', 
    'dual': False,
    'verbose': 1,
    # 'kernel': 'rbf',
    # 'gamma': 5,
    # 'C': 0.001,
    # 'shrinking': True,
    # 'probability': False, 
    # 'cache_size': 200, # NOTE issue with a large cache_size - https://github.com/scikit-learn/scikit-learn/issues/8012
    'max_iter': 1000
}

def objective(trial, x_train, y_train, params=params):
    
    temp_map = {
        # 'gamma': trial.suggest_loguniform('gamma', 1e-5, 1e-2),
        'C': trial.suggest_loguniform('C', 1e-10, 1e10)
    }
    params.update(temp_map)
    svc_classifier = LinearSVC(**params)
    y_oof = np.zeros(x_train.shape[0])
    acc_scores = []
    # pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation_0-auc")
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS)
    for i, (train_index, valid_index) in enumerate(rskf.split(x_train, y_train)):
        start_time = timer()
        X_A, X_B = x_train[train_index, :], x_train[valid_index, :]
        y_A, y_B = y_train[train_index], y_train[valid_index]
        svc_classifier.fit(
            X_A, y_A
            # eval_set=[(X_B, y_B)], 
            # early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=0,
            # callbacks=[pruning_callback]
        )
        y_oof[valid_index] = svc_classifier.predict(X_B)
        acc_score = accuracy_score(y_B, y_oof[valid_index])
        acc_scores.append(acc_score)
        timer(start_time)
        print(f"===== {i} fold : acc {acc_score} =====")
    # trial.set_user_attr(key="best_booster", value=svc_classifier) # NOTE update the best model in the optuna's table.
    res = np.mean(acc_scores)
    print(f"===== {res} =====")
    return res 
    
# def save_best(study, trial):
#     if study.best_trial.number == trial.number:
#         # Set the best booster as a trial attribute; accessible via study.trials_dataframe.
#         study.set_user_attr(key="best_booster", value=trial.user_attrs["best_booster"])
#         # SOURCE retrieve the best number of estimators https://github.com/optuna/optuna/issues/1169    

study = optuna.create_study(
    direction = "maximize", 
    sampler = optuna.samplers.TPESampler(seed=2021, multivariate=True)
#     pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)
)
study.optimize(lambda trial: objective(trial, 
                                       x_pca_transformed[:train_rows, :], 
                                       train_label.iloc[:train_rows].values, 
                                       params), 
               n_trials=N_TRIALS, 
               timeout=TIMEOUT, 
               n_jobs=1
#                callbacks=[save_best]
)

hp = study.best_params
for key, value in hp.items():
    print(f"{key:>20s} : {value}")
print(f"{'best objective value':>20s} : {study.best_value}")  
# best_model=study.user_attrs["best_booster"]

pickle_to_save = '10linearsvc6hrs.pkl'
os.chdir('/kaggle/working')
pickle.dump(best_model, open(f'{pickle_to_save}', 'wb'))
sshColab.upload_to_gcs(project, bucket_name, f'tps-apr-2021-label/{pickle_to_save}', f'/kaggle/working/{pickle_to_save}')
# %%

