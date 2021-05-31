"""
reference: https://www.kaggle.com/optimo/tabnet-baseline

NOT WORKING YET.
"""

# %%
import pandas as pd
import numpy as np
import random
import os
import pickle
import datetime
import pytz

from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.linear_model import SGDClassifier

from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from category_encoders import OrdinalEncoder
import lightgbm as lgb
from scipy.special import erfinv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC

pd.options.display.max_rows = 100
pd.options.display.max_columns = 100

import warnings
warnings.simplefilter('ignore')

TARGET = 'target'
NUM_CLASSES = 4
SEED = 2021

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
seed_everything(SEED)

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


os.chdir('/kaggle/working')
train = pd.read_csv('../input/tabular-playground-series-may-2021/train.csv')
test = pd.read_csv('../input/tabular-playground-series-may-2021/test.csv')
submission = pd.read_csv('../input/tabular-playground-series-may-2021/sample_submission.csv')

features = train.iloc[:, 1:-1].columns.tolist()
all_df = pd.concat([train, test]).reset_index(drop=True).drop(['id', 'target'], axis=1)

from sklearn.preprocessing import LabelEncoder
all_df = all_df.apply(LabelEncoder().fit_transform)

def rank_gauss_standardscaler(all_df):
    def rank_gauss(x):
        N = x.shape[0]
        temp = x.argsort()
        rank_x = temp.argsort() / N
        rank_x -= rank_x.mean()
        rank_x *= 2
        efi_x = erfinv(rank_x)
        efi_x -= efi_x.mean()
        return efi_x

    all_df = all_df.apply(rank_gauss)    
    for col in feature_columns:
        all_df[col] = StandardScaler().fit_transform(all_df[col].values.reshape(-1,1))
    return all_df
all_df = rank_gauss_standardscaler(all_df)    

X_trn, X_tst = all_df.iloc[:len(train)], all_df.iloc[len(train):]
y_trn = train['target']
RANDOM_SEED = 2021


# SECTION experiment
# %%

# NOTE Qunatile or not

cat_dims = X_trn[features].nunique().tolist()
cat_idxs = [features.index(col) for col in features]

LOG_EMBEDDING_DIM = True
if LOG_EMBEDDING_DIM:
    cat_emb_dims = np.ceil(np.clip((np.array(cat_dims)) / 2, a_min=1, a_max=50)).astype(np.int).tolist()
else:
    cat_emb_dims = np.ceil(np.log(cat_dims)).astype(np.int).tolist()        

X = X_trn[features].values

le = LabelEncoder()
y = le.fit_transform(y_trn) # !!
X_test = X_tst[features].values

from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetClassifier

import torch
N_D = 16
N_A = 16
N_INDEP = 2
N_SHARED = 2
N_STEPS = 1 #2
MASK_TYPE = "sparsemax"
GAMMA = 1.5
BS = 128 #512
MAX_EPOCH =  20 # 20
PRETRAIN = True

clf = TabNetClassifier()
clf.fit(
  X, y,
#   eval_set=[(X_valid, y_valid)]
)


# if PRETRAIN:
#     pretrain_params = dict(n_d=N_D, n_a=N_A, n_steps=N_STEPS,  #0.2,
#                            n_independent=N_INDEP, n_shared=N_SHARED,
#                            cat_idxs=cat_idxs,
#                            cat_dims=cat_dims,
#                            cat_emb_dim=cat_emb_dims,
#                            gamma=GAMMA,
#                            lambda_sparse=0., optimizer_fn=torch.optim.Adam,
#                            optimizer_params=dict(lr=2e-2),
#                            mask_type=MASK_TYPE,
#                            scheduler_params=dict(mode="min",
#                                                  patience=3,
#                                                  min_lr=1e-5,
#                                                  factor=0.5,),
#                            scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,                         
#                            verbose=1,
#                           )

#     # pretrainer = TabNetPretrainer(**pretrain_params)
#     pretrainer = TabNetClassifier(**pretrain_params)

#     pretrainer.fit(X_train=X_test, 
#                    eval_set=[X],
#                    max_epochs=MAX_EPOCH,
#                    patience=25, batch_size=BS, virtual_batch_size=BS, #128,
#                    num_workers=0, drop_last=True,
#                 #    pretraining_ratio=0.1 # 0.5 The bigger your pretraining_ratio the harder it is to reconstruct
#                   )
