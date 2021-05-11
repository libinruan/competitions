# Experiments on one-hot encoding or label encoding based on the framework of the base file.

# %% [code]
# from scipy.sparse.csgraph import connected_components
# from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from umap import UMAP
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import copy
import seaborn as sns
import time

from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA,FactorAnalysis
from sklearn.manifold import TSNE

from matplotlib.patches import Rectangle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
print(torch.cuda.is_available())
import warnings

def seed_everything(seed=1903):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    


os.chdir('/kaggle/working')
train_df = pd.read_csv('../input/tabular-playground-series-mar-2021/train.csv')
test_df = pd.read_csv('../input/tabular-playground-series-mar-2021/test.csv')
sample_submission = pd.read_csv('../input/tabular-playground-series-mar-2021/sample_submission.csv')

feature_cols = train_df.drop(['id', 'target'], axis=1).columns

# approach 1
numerical_columns = train_df[feature_cols].select_dtypes(include=['int64','float64']).columns
categorical_columns = train_df[feature_cols].select_dtypes(exclude=['int64','float64']).columns
# approach 2
# [s for s in train_df[feature_cols].columns if s.startswith('cat')]
# [s for s in train_df[feature_cols].columns if s.startswith('con')]

# Join train and test datasets in order to obtain the same number of features during categorical conversion
train_indexs = train_df.index
test_indexs = test_df.index

df = pd.concat(objs=[train_df, test_df], axis=0).reset_index(drop=True)
df = df.drop(['id', 'target'], axis=1)

# 
train = train_df.copy() 
test = test_df.copy()

# %% [code]

# ------------------------------------ EDA ----------------------------------- #

def plot_correlation_by_splits(train_df, test_df, numerical_columns):
    background_color = "#f6f5f5"

    fig = plt.figure(figsize=(18, 8), facecolor=background_color)
    gs = fig.add_gridspec(1, 2)
    gs.update(wspace=-0.36, hspace=0.27)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    colors = ["#fbfbfb", "lightgray","#0e4f66"]
    colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

    ax0.set_facecolor(background_color)
    ax0.text(0, 0, 'Train', fontsize=20, fontweight='bold', fontfamily='serif',color='lightgray')

    ax1.set_facecolor(background_color)
    ax1.text(9.5, 11, 'Test', fontsize=20, fontweight='bold', fontfamily='serif',color='lightgray')

    fig.text(0.5,0.5,'Correlation of Features\nFor Train & Test\nDatasets', fontsize=20, fontweight='bold', fontfamily='serif',va='center',ha='center')

    corr = train_df[numerical_columns].corr().abs()
    mask = np.triu(np.ones_like(corr, dtype=np.bool))
    sns.heatmap(corr, ax=ax0, vmin=-1, vmax=1, annot=True, square=True, mask=mask,
                cbar_kws={"orientation": "horizontal"}, cbar=False, cmap=colormap, fmt='.1g',linewidth=3,linecolor=background_color)

    corr = test_df[numerical_columns].corr().abs()
    mask = np.tril(corr)
    sns.heatmap(corr, ax=ax1, vmin=-1, vmax=1, annot=True, square=True, mask=mask,
                cbar_kws={"orientation": "horizontal"}, cbar=False, cmap=colormap, fmt='.1g',linewidth=3,linecolor=background_color)
    ax1.xaxis.tick_top()
    ax1.yaxis.tick_right()

    plt.show()


# plot_correlation_by_splits(train_df, test_df, numerical_columns)
# ANCHOR show correlation of cont. feature with target feature
def plot_correlation_of_continuous_features_with_target(train_df, numerical_columns):
    background_color = "#f6f5f5"

    fig = plt.figure(figsize=(12, 6), facecolor=background_color)
    gs = fig.add_gridspec(1, 1)
    ax0 = fig.add_subplot(gs[0, 0])

    ax0.set_facecolor(background_color)
    ax0.text(-1.1, 0.26, 'Correlation of Continuous Features with Target', fontsize=20, fontweight='bold', fontfamily='serif')
    ax0.text(-1.1, 0.24, 'We see correlation in both directions, with cont5 having the highest positive correlation.' ,fontsize=13, fontweight='light', fontfamily='serif')

    chart_df = pd.DataFrame(train_df[numerical_columns].corrwith(train_df['target']))
    chart_df.columns = ['corr']
    chart_df['positive'] = chart_df['corr'] > 0

    sns.barplot(x=chart_df.index, y=chart_df['corr'], ax=ax0, palette=chart_df.positive.map({True: '#0e4f66', False: 'gray'}), zorder=3,dodge=False)
    ax0.grid(which='major', axis='y', zorder=0, color='gray', linestyle=':', dashes=(1,5))
    ax0.set_ylabel('')


    for s in ["top","right", 'left']:
        ax0.spines[s].set_visible(False)

    plt.show()


# plot_correlation_of_continuous_features_with_target(train_df, numerical_columns)


# ---------------------

from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ANCHOR show cont. feature distribution
def plot_continuous_feature_distribution(train_df):
    background_color = "#f6f5f5"
    fig, ax = plt.subplots(3, 4, figsize=(17, 12), sharex=True)
    fig.set_facecolor('#f6f5f5') 

    for i in range(11): 
        sns.kdeplot(data=train_df[train_df['target']==1], x=f'cont{i}', 
                    fill=True,
                    linewidth=1,
                    color='#0e4f66', alpha=0.8,
                    ax=ax[i%3][i//3])

        sns.kdeplot(data=train_df[train_df['target']==0], x=f'cont{i}', 
                    fill=True,
                    linewidth=1,
                    color='#d0d0d0', alpha=0.8,
                    ax=ax[i%3][i//3])

        ax[i%3][i//3].set_yticks([])
        ax[i%3][i//3].set_ylabel('',visible=False)
        ax[i%3][i//3].set_xlabel('',visible=False)
        ax[i%3][i//3].margins(0.05, 0.2)
        ax[i%3][i//3].set_facecolor(background_color) 
        for s in ["top","right", 'left']:
                ax[i%3][i//3].spines[s].set_visible(False)           

        # bar
        divider = make_axes_locatable(ax[i%3][i//3])
        cax = divider.append_axes("top", size="8%", pad=0)
        cax.get_xaxis().set_visible(False)
        cax.get_yaxis().set_visible(False)
        cax.set_facecolor('black')

        at = AnchoredText(f'cont{i}', loc=10, 
                          prop=dict(backgroundcolor='black',
                                    size=10, color='white', weight='bold'))
        cax.add_artist(at)

    ax[-1][-1].set_visible(False)
    fig.text(0.018, 1.03, 'Continuous Feature Distribution by Target [Train]', fontsize=20, fontweight='bold', fontfamily='serif')

    plt.tight_layout()
    plt.show()

# plot_continuous_feature_distribution(train_df)


# %% [code]
# ------------------------------ TIMING FUNCTION ----------------------------- #

import time
class Timer:
    def __enter__(self):
        self.start=time.time()
        return self
    def __exit__(self, *args):
        self.end=time.time()
        self.hour, temp = divmod((self.end - self.start), 3600)
        self.min, self.second = divmod(temp, 60)
        self.hour, self.min, self.second = int(self.hour), int(self.min), round(self.second, 2)
        return self



# -------------------------- Gradient Boosting Zone -------------------------- #

# Baseline models from [TPS EDA & Model [March 2020]](https://www.kaggle.com/joshuaswords/tps-eda-model-march-2020/data)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from category_encoders import CatBoostEncoder, LeaveOneOutEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report,plot_confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, precision_score
from sklearn.svm import LinearSVC

# Question: When do we choose Label Encoder over One Hot Encoder? [StackOverflow](https://datascience.stackexchange.com/questions/9443/when-to-use-one-hot-encoding-vs-labelencoder-vs-dictvectorizor)
# Question: What's One-hot encoding and what's label encoding? [post](https://www.analyticsvidhya.com/blog/2020/03/one-hot-encoding-vs-label-encoding-using-scikit-learn/)
# Takeaway: it is not an issue, if you are using decision tree like algorithms such as XGboost or Catboost.

def label_encoding_before_data_splits(train_df, test_df):
    # note: tran_df shares the same columns with test_df.

    for c in train_df.columns:
        if train_df[c].dtype=='object': 
            lbl = LabelEncoder() # each label is assigned a unique integer based on alphabetical    ordering.
            lbl.fit(list(train_df[c].values) + list(test_df[c].values)) # pooling train set and test set.
            train_df[c] = lbl.transform(train_df[c].values)
            test_df[c] = lbl.transform(test_df[c].values)
    return train_df, test_df

def one_hot_encoding_before_data_splits(train_df, test_df, drop_first=True):
    # note: tran_df shares the same columns with test_df.
    size_train_df = len(train_df)
    combined_df = pd.concat([train_df, test_df], axis=0)
    for c in train_df.columns:
        if train_df[c].dtype=='object': 
            combined_df[c] = pd.get_dummies(combined_df[c], drop_first=drop_first) # FIXME multiple columns are assigned to an identical column.
    train_df = combined_df.iloc[:size_train_df, :]
    test_df = combined_df.iloc[size_train_df:, :]
    test_df = test_df.drop('target', axis=1)
    return train_df, test_df

# ------------------------------- RANDOM FOREST ------------------------------ #

# 
def train_random_forest(X_train, y_train):  
    seed_everything(seed=1903)
    rfc = RandomForestClassifier(n_estimators=200, max_depth=7, n_jobs=-1)
    rfc.fit(X_train, y_train)
    rfc_pred = rfc.predict_proba(X_test)[:, 1] # This grabs the positive class prediction.
    score = roc_auc_score(y_test, rfc_pred)
    print(f'rf: {score:0.5f}') 
    rf_df = pd.DataFrame(data=[roc_auc_score(y_test, rfc_pred)], 
        columns=['Random Forest Score'],
        index=["ROC AUC Score"])
    return rf_df, rfc, rfc_pred

    # Tutorial: Learn XGBoost in details [mlmaster](https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/)
    # Advanced: How to Configure the Gradient Boosting Algorithm? [Answer](https://machinelearningmastery.com/configure-gradient-boosting-algorithm/)


# --------------------------------- CATBOOST --------------------------------- #

# 
params = {'iterations': 3000,
          'learning_rate':0.01,
          'depth':3,
          'eval_metric':'AUC',
          'verbose':200,
          'od_type':'Iter',
          'od_wait':50}

def train_catboost(X_train, y_train, params):
    seed_everything(seed=1903)
    cat_model = CatBoostClassifier(**params)
    cat_model.fit(X_train,y_train)
    cb_pred = cat_model.predict_proba(X_test)[:, 1] # This grabs the positive class prediction.
    score = roc_auc_score(y_test, cb_pred)
    print(f'cat: {score:0.5f}')
    cb_df = pd.DataFrame(data=[roc_auc_score(y_test, cb_pred)], 
        columns=['CatBoost Score'],
        index=["ROC AUC Score"])
    return cb_df, cat_model, cb_pred                 



# ---------------------------------- XGBOOST --------------------------------- #

# 
def train_xgboost(X_train, y_train):
    seed_everything(seed=1903)
    xgb_model = XGBClassifier(
        eval_metric="auc",
        random_state=42,
        tree_method="gpu_hist",
        gpu_id="0",
        use_label_encoder=False,verbose=200)
    xgb_model.fit(X_train,y_train)
    xgb_pred = xgb_model.predict_proba(X_test)[:, 1] # This grabs the positive class prediction
    score = roc_auc_score(y_test, xgb_pred)
    print(f'xgb: {score:0.5f}') 

    xgb_df = pd.DataFrame(data=[roc_auc_score(y_test, xgb_pred)], 
                 columns=['XGBoost Score'],
                 index=["ROC AUC Score"])
    return xgb_df, xgb_model, xgb_pred

                 

# ----------------------------- VOTING CLASSIFIER ---------------------------- #
# 
def train_voting_classifier(estimators, X_train, y_train):
    seed_everything(seed=1903)
    voting_clf = VotingClassifier(estimators = estimators, voting = 'soft') 
    voting_clf.fit(X_train,y_train)
    vc_pred = voting_clf.predict_proba(X_test)[:, 1] # This grabs the positive class prediction
    score = roc_auc_score(y_test, vc_pred)
    print(f'voting: {score:0.5f}') 
    vc_df = pd.DataFrame(data=[roc_auc_score(y_test, vc_pred)], 
                 columns=['Voting Classifier Score'],
                 index=["ROC AUC Score"])
    return vc_df, voting_clf, vc_pred



# ---------------------- GRADIENT BOSSTING ALGOS RESULTS --------------------- #
#
# Overall
def plot_gradient_boosting_algos(df_models, note=': One-Hot Encoding', save='save.png'):
    colors = ["lightgray","#0e4f66"]
    colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
    background_color = "#f6f5f5"

    fig = plt.figure(figsize=(12,3), facecolor=background_color) # create figure
    gs = fig.add_gridspec(1, 1)
    gs.update(wspace=0.1, hspace=0.5)
    ax0 = fig.add_subplot(gs[0, :])

    
    sns.heatmap(df_models, cmap=colormap,annot=True,fmt=".2%",vmax=0.885,vmin=0.883, linewidths=2.5,cbar=False,annot_kws={"fontsize":15})

    ax0.text(-0.63,-0.4,f'Model Performance Overview & Selection{note}',fontfamily='serif',fontsize=20,fontweight='bold')
    ax0.text(-0.63,-0.23,'We tried several tree models, but Gradient Boosting Models out-performed the rest.',fontfamily='serif',fontsize=15)

    ax0.set_yticklabels(ax0.get_yticklabels(), fontfamily='serif', rotation=0, fontsize=12)
    ax0.set_xticklabels(ax0.get_xticklabels(), fontfamily='serif', rotation=0, fontsize=12)

    for lab, annot in zip(ax0.get_xticklabels(), ax0.texts):
        text =  lab.get_text()
        if text == 'CatBoost Score': 
            # set the properties of the ticklabel
            lab.set_weight('bold')
            lab.set_size(15)
            lab.set_color('black')

    for lab, annot in zip(ax0.get_xticklabels(), ax0.texts):
        text =  lab.get_text()
        if text == 'XGBoost Score': 
            # set the properties of the ticklabel
            lab.set_weight('bold')
            lab.set_size(15)
            lab.set_color('black')

    ax0.add_patch(Rectangle((0, 0), 2, 4, fill=True,color='#0e4f66', edgecolor='white', lw=0,alpha=0.5))
    fig.savefig(f'{save}')

# %% [code]

train_df, test_df = label_encoding_before_data_splits(train_df, test_df)
target = train_df.pop('target')
X_train, X_test, y_train, y_test = train_test_split(train_df, target, train_size=0.80)
rf_df, rfc, rfc_pred = train_random_forest(X_train, y_train)
cb_df, cat_model, cb_pred = train_catboost(X_train, y_train, params)
xgb_df, xgb_model, xgb_pred = train_xgboost(X_train, y_train)    
estimators = [('rfc',rfc), ('cat_model',cat_model), ('xgb_model',xgb_model)]
vc_df, voting_clf, vc_pred = train_voting_classifier(estimators, X_train, y_train)
df_models = round(pd.concat([vc_df,rf_df,cb_df,xgb_df], axis=1),4)
plot_gradient_boosting_algos(df_models, note=': Label Encoding', save='label_encoding.png')    

# %% [code]
ids = test['id'].values
ypre = xgb_model.predict_proba(test_df)[:, 1]
output = pd.DataFrame({'id': ids, 'target': ypre})
output.to_csv('submission.csv', index=False)

# %% [code]
# -------------------- ONE-HOT ENCODING EXPERIMENTAL ZONE -------------------- #
# DROP THE FIRST COLUMN 

#     train_df = train.copy()
#     test_df = test.copy()
#     train_df, test_df = one_hot_encoding_before_data_splits(train_df, test_df, drop_first=True)
#     target = train_df.pop('target')
#     X_train, X_test, y_train, y_test = train_test_split(train_df, target, train_size=0.80)
#     rf_df1, rfc1, rfc_pred1 = train_random_forest(X_train, y_train)
#     cb_df1, cat_model1, cb_pred1 = train_catboost(X_train, y_train, params)
#     xgb_df1, xgb_model1, xgb_pred1 = train_xgboost(X_train, y_train)   
#     estimators1 = [('rfc',rfc1), ('cat_model',cat_model1), ('xgb_model',xgb_model1)]
#     vc_df1, voting_clf1, vc_pred1 = train_voting_classifier(estimators1, X_train, y_train)
#     df_models1 = round(pd.concat([vc_df1,rf_df1,cb_df1,xgb_df1], axis=1),4)
#     plot_gradient_boosting_algos(df_models1, note=': One-Hot, Drop First', save='one_hot_drop_1.png')


#%%
# KEEP THE FIRST COLUMN  

#     train_df = train.copy()
#     test_df = test.copy()
#     train_df, test_df = one_hot_encoding_before_data_splits(train_df, test_df, drop_first=False)
#     target = train_df.pop('target')
#     X_train, X_test, y_train, y_test = train_test_split(train_df, target, train_size=0.80)
#     rf_df2, rfc2, rfc_pred2 = train_random_forest(X_train, y_train)
#     cb_df2, cat_model2, cb_pred2 = train_catboost(X_train, y_train, params)
#     xgb_df2, xgb_model2, xgb_pred2 = train_xgboost(X_train, y_train)   
#     estimators2 = [('rfc',rfc), ('cat_model',cat_model), ('xgb_model',xgb_model)]
#     vc_df2, voting_clf2, vc_pred2 = train_voting_classifier(estimators2, X_train, y_train)
#     df_models2 = round(pd.concat([vc_df2,rf_df2,cb_df2,xgb_df2], axis=1),4)
#     plot_gradient_boosting_algos(df_models2, note=': One-Hot, Keep First', save='one_hot_keep_1.png')




#%%
# -------------------------- GRID SEARCH FOR XGBOOST ------------------------- #

# -stackoverflow: https://tinyurl.com/y98buher
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline

# define scoring function 
def custom_auc(ground_truth, predictions):
    # We need only the second column of predictions["0" and "1"]. Otherwise, 
    # we get an error here while trying to return both columns at once
    fpr, tpr, _ = roc_curve(ground_truth, predictions[:, 1], pos_label=1)    
    return auc(fpr, tpr)
# custom scorer        
my_scorer = make_scorer(custom_auc, greater_is_better=True, needs_proba=True)
params = {
    'tree_method': 'gpu_hist', 
    'max_depth':8,
    'alpha': 0.1,
    'gamma': 0.3, 
    'subsample': 0.6, 
    'scale_pos_weight': 1, 
    'random_state': 42,
    'learning_rate': 0.05, 
    'n_estimators': 1000,
    'objective':'binary:logistic', 
    'eval_metric': 'auc', 
    'min_child_weight': 5,
    'silent':True,
    'scale_pos_weight': 1.0,  
    'gpu_id': "0",
    'verbose': 200,
    'use_label_encoder': False,
    'verbose_eval': False
}
pipeline = Pipeline([
    # ("transformer", TruncatedSVD(n_components=70)),
    ("classifier", XGBClassifier(**params))
])
grid_params={
    "classifier__learning_rate": [0.05, 0.1, 0.3],
    "classifier__max_depth": [8, 12, 15],
    "classifier__n_estimators": [200, 600, 1000]
}
grid_search= GridSearchCV(
    pipeline,
    grid_params,
    scoring = my_scorer,
    n_jobs = -1,
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
)
train_df, test_df = label_encoding_before_data_splits(train_df, test_df)
target = train_df.pop('target')
with Timer() as t:
    grid_search.fit(train_df, target)
print(f'time elapsed: {t.interval} minutes')
grid_search.best_estimaor_
grid_search.best_params_


