#%%
"""
check the original source first. somethign wrong with my revision.
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from colorama import Fore

from pandas_profiling import ProfileReport
import seaborn as sns
from sklearn import metrics
from scipy import stats
import math

from tqdm.notebook import tqdm
from copy import deepcopy

from sklearn.preprocessing import LabelEncoder

from umap import UMAP
from sklearn.manifold import TSNE

train_df = pd.read_csv('/kaggle/input/tabular-playground-series-may-2021/train.csv')
train_df.columns = [column.lower() for column in train_df.columns]
# train_df = train_df.drop(columns=['passengerid'])

test_df = pd.read_csv('/kaggle/input/tabular-playground-series-may-2021/test.csv')
test_df.columns = [column.lower() for column in test_df.columns]

submission = pd.read_csv('/kaggle/input/tabular-playground-series-may-2021/sample_submission.csv')
submission.head()

train_df.head()

feature_columns = train_df.iloc[:, 1:-1].columns.values
target_column = 'target'
feature_columns


# Time Functions
import datetime
import pytz

def dtnow(tz="America/New_York"):
    utc_now = pytz.utc.localize(datetime.datetime.utcnow())
    pst_now = utc_now.astimezone(pytz.timezone("America/New_York"))
    return pst_now.strftime("%Y%m%d%H%M")

def timer(start_time=None):
    if not start_time:
        start_time = datetime.datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('Time taken : %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

# %%

# SECTION Light Auto ML
"""
reference: https://www.kaggle.com/andreshg/tps-may-a-complete-analysis
title: [TPS-May] 🏄‍♂️ A Complete Analysis
"""
start = timer()
from lightautoml.automl.presets.tabular_presets import TabularAutoML, TabularUtilizedAutoML
from lightautoml.tasks import Task
from sklearn.metrics import log_loss

N_THREADS = 4 # threads cnt for lgbm and linear models
N_FOLDS = 7 # folds cnt for AutoML
RANDOM_STATE = 2021 # fixed random state for various reasons
TEST_SIZE = 0.2 # Test size for metric check
TIMEOUT = 4 * 3600 # Time in seconds for automl run
TARGET_NAME = 'target'

# train_data = pd.read_csv('/kaggle/input/tabular-playground-series-may-2021/train.csv')
# test_data = pd.read_csv('/kaggle/input/tabular-playground-series-may-2021/test.csv')
# train_data[TARGET_NAME] = train_data[TARGET_NAME].str.slice(start=6).astype(int) - 1

train_df = pd.read_csv('/kaggle/input/tabular-playground-series-may-2021/train.csv')
train_df.columns = [column.lower() for column in train_df.columns]
# train_df = train_df.drop(columns=['passengerid'])

test_df = pd.read_csv('/kaggle/input/tabular-playground-series-may-2021/test.csv')
test_df.columns = [column.lower() for column in test_df.columns]

submission = pd.read_csv('/kaggle/input/tabular-playground-series-may-2021/sample_submission.csv')
submission.head()

feature_columns = train_df.iloc[:, 1:-1].columns.values
target_column = 'target'

le = LabelEncoder()
train_df[target_column] = le.fit_transform(train_df[target_column])


task = Task('multiclass',)

roles = {
    'target': target_column,
    'drop': ['id'],
}

automl = TabularUtilizedAutoML(task = task, 
                               timeout = TIMEOUT,
                               cpu_limit = N_THREADS,
                               reader_params = {'n_jobs': N_THREADS},
                               verbose=0
)

oof_pred = automl.fit_predict(train_df, roles = roles)

import pickle
name = f'{dtnow()}-lightAutoML-4hr-model'
oof_pred = automl.fit_predict(train_df, roles = roles)
pickle.dump(automl, open(f'/kaggle/working/may_model/{name}.pkl', 'wb'))

test_pred = automl.predict(test_df)
# print('Prediction for test data:\n{}\nShape = {}'.format(test_pred[:10], test_pred.shape))

print('Check scores...')
print('OOF score: {}'.format(log_loss(train_df[target_column].values, oof_pred.data)))

submission = pd.read_csv('/kaggle/input/tabular-playground-series-may-2021/sample_submission.csv')
submission.iloc[:, 1:] = test_pred.data
submission.to_csv(f'/kaggle/working/may_model/{name}.csv', index = False)

timer(start)
# %%

# SECTION H20 auto ML
"""
reference: https://www.kaggle.com/andreshg/tps-may-a-complete-analysis
title: [TPS-May] 🏄‍♂️ A Complete Analysis
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from colorama import Fore

from pandas_profiling import ProfileReport
import seaborn as sns
from sklearn import metrics
from scipy import stats
import math

from tqdm.notebook import tqdm
from copy import deepcopy

from sklearn.preprocessing import LabelEncoder

from umap import UMAP
from sklearn.manifold import TSNE

train_df = pd.read_csv('/kaggle/input/tabular-playground-series-may-2021/train.csv')
train_df.columns = [column.lower() for column in train_df.columns]
# train_df = train_df.drop(columns=['passengerid'])

test_df = pd.read_csv('/kaggle/input/tabular-playground-series-may-2021/test.csv')
test_df.columns = [column.lower() for column in test_df.columns]

submission = pd.read_csv('/kaggle/input/tabular-playground-series-may-2021/sample_submission.csv')
submission.head()

train_df.head()

feature_columns = train_df.iloc[:, 1:-1].columns.values
target_column = 'target'
feature_columns


# Time Functions
import datetime
import pytz

def dtnow(tz="America/New_York"):
    utc_now = pytz.utc.localize(datetime.datetime.utcnow())
    pst_now = utc_now.astimezone(pytz.timezone("America/New_York"))
    return pst_now.strftime("%Y%m%d%H%M")

def timer(start_time=None):
    if not start_time:
        start_time = datetime.datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('Time taken : %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

from google.cloud import storage

project = "strategic-howl-305522"
bucket_name = "gcs-station-168"           
storage_client = storage.Client(project=project)
bucket = storage_client.bucket(bucket_name)

import h2o
from h2o.automl import H2OAutoML

h2o.init()

train_hf = h2o.H2OFrame(train_df.copy())
test_hf = h2o.H2OFrame(test_df.copy())

train_hf[target_column] = train_hf[target_column].asfactor()

test = pd.DataFrame(columns=['a'])


aml = H2OAutoML(
    seed=2021, 
    max_runtime_secs= 2 * 60 * 60,
    nfolds = 7,
    exclude_algos = ["DeepLearning"]
)

aml.train(
    x=list(feature_columns), 
    y=target_column, 
    training_frame=train_hf
)

import pickle
name = f'{dtnow()}-aml-model.pkl'
pickle.dump(aml, open(f'./{name}', 'wb'))
gcs_folder = 'tps-may-2021-label/'
local_filename = f'{name}'
blob = bucket.blob(f'{gcs_folder}{local_filename}')
blob.upload_from_filename(f'./{local_filename}')  

lb = aml.leaderboard 
lb.head(rows = lb.nrows)

preds = aml.predict(h2o.H2OFrame(test_df[feature_columns].copy()))
preds_df = h2o.as_list(preds)
preds_df

submission[['Class_1', 'Class_2', 'Class_3', 'Class_4']] = preds_df[['Class_1', 'Class_2', 'Class_3', 'Class_4']]
# submission.to_csv('h2o_automl_300s.csv', index=False)
# submission.head()

name = f'{dtnow()}-h2o_automl_2hr_sub.csv'
submission.to_csv(name, index=False)
gcs_folder = 'tps-may-2021-label/'
local_filename = f'{name}'
blob = bucket.blob(f'{gcs_folder}{local_filename}')
blob.upload_from_filename(f'./{local_filename}')  


# SECTION Blender example (Have not been executed)

catboost.rename(columns = {'Class_1' : 'catboost_Class_1'}, inplace = True)
catboost.rename(columns = {'Class_2' : 'catboost_Class_2'}, inplace = True)
catboost.rename(columns = {'Class_3' : 'catboost_Class_3'}, inplace = True)
catboost.rename(columns = {'Class_4' : 'catboost_Class_4'}, inplace = True)
lightautoml.rename(columns = {'Class_1' : 'light_Class_1'}, inplace = True)
lightautoml.rename(columns = {'Class_2' : 'light_Class_2'}, inplace = True)
lightautoml.rename(columns = {'Class_3' : 'light_Class_3'}, inplace = True)
lightautoml.rename(columns = {'Class_4' : 'light_Class_4'}, inplace = True)

lightautoml_and_catboost = pd.merge(left=catboost, right=lightautoml, left_on="id", right_on="id", how="left")

lightautoml_and_catboost["Class_1"] = (lightautoml_and_catboost["light_Class_1"] * 0.80) + (lightautoml_and_catboost["catboost_Class_1"] * 0.20)
lightautoml_and_catboost["Class_2"] = (lightautoml_and_catboost["light_Class_2"] * 0.80) + (lightautoml_and_catboost["catboost_Class_2"] * 0.20)
lightautoml_and_catboost["Class_3"] = (lightautoml_and_catboost["light_Class_3"] * 0.80) + (lightautoml_and_catboost["catboost_Class_3"] * 0.20)
lightautoml_and_catboost["Class_4"] = (lightautoml_and_catboost["light_Class_4"] * 0.80) + (lightautoml_and_catboost["catboost_Class_4"] * 0.20)

lightautoml_and_catboost = lightautoml_and_catboost[["id","Class_1","Class_2","Class_3","Class_4"]]
lightautoml_and_catboost.to_csv("lightautoml_and_catboost.csv", index=False)

print(lightautoml_and_catboost.shape)
lightautoml_and_catboost

