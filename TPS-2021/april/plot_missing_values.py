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
import matplotlib.pyplot as plt

def label_encoder(c):
    lc = LabelEncoder()
    return lc.fit_transform(c)

train_df = pd.read_csv('/kaggle/input/tabular-playground-series-apr-2021/train.csv')

# %%
import seaborn as sns

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,5))

sns.heatmap(train_df.isnull(), yticklabels=False, cbar=False, cmap='viridis', ax=ax1)

# %%
