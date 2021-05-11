# %% [markdown]

# 不成功的版本。。。EDA的部份很好。簡潔但詳盡。可以參考。

# # Notebook en français.
# 1st part. credits to https://www.kaggle.com/valtintin/r-gressions-dataviz-comp-tition-en-fran-ais 
# 2nd half. credits to https://www.kaggle.com/lipinjuan/bayesian-optimization-with-optuna-stacking/edit

# %% [markdown]
# ## <p style="font-family:newtimeroman; font-size:200%; text-align:center">Sommaire</p>
# 
# * [1. Télechargement des données](#1)
# * [2. l'analyse du fichier](#2)
#      * [2.1 Forme du fichier](#2.1)
#      * [2.2 Déclaration des variables](#2.2)
# * [3. Chiffres clés](#3)
#     * [3.1 pandas_profiling](#3.1)
#     * [3.2 Methode classique](#3.2)
#     * [3.3 Variables cibles](#3.3)
#     * [3.4 Variables numériques](#3.4)
#     * [3.4 Variables caractères](#3.5)    
# * [4. Modèlisation](#4)
# * [5. Optimisation](#5)
# * [6. Fichier soumission](#6)

# %% [code]
import numpy as np 
import pandas as pd 
import os
from datetime import timedelta 
from datetime import datetime
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import learning_curve
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,BaggingClassifier,GradientBoostingClassifier,StackingClassifier,VotingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
RANDOM_STATE = 42
warnings.simplefilter(action='ignore', category=FutureWarning)

# %% [markdown]
# <a id='1'></a>
# # <p style="font-family:newtimeroman; font-size:150%; text-align:center">1. Télechargement des données </p>

# %% [code]
RunningInCOLAB = 'COLAB_GPU' in os.environ or os.environ.get('PWD') == '/kaggle/working'
if RunningInCOLAB:
    os.chdir('/kaggle/working')
    train = pd.read_csv('../input/tabular-playground-series-mar-2021/train.csv', index_col=0)
    test = pd.read_csv('../input/tabular-playground-series-mar-2021/test.csv', index_col=0)
    sample_submission = pd.read_csv('../input/tabular-playground-series-mar-2021/sample_submission.csv')
else:
    os.chdir('G:\kagglePlayground')
    train = pd.read_csv('train.csv', index_col=0)
    test = pd.read_csv('test.csv', index_col=0)
    sample_submission = pd.read_csv('sample_submission.csv') 
# train.set_index("id",inplace=True)
# test.set_index("id",inplace=True)

# %%


# %% [markdown]
# <a id='2'></a>
# ## <p style=" font-family:newtimeroman; font-size:150%; text-align:center">2. l'analyse du fichier</p>

# %% [markdown]
# <a id='2.1'></a>
# ## <p style=" font-family:newtimeroman; font-size:110%; text-align:center">2.1 Forme du fichier</p>

# %% [code]
print(train.shape)
print(test.shape)

# %% [code]
train.columns

# %% [code]
train.dtypes.value_counts().plot.pie(title='Répartition des variables par type')

# %% [code]
print('Nombre de valeures manquante Train {}'.format(train.isna().sum().sum()))
print('Nombre de valeures manquante Test  {}'.format(test.isna().sum().sum()))

# %% [code]
train.head()

# %% [markdown]
# <a id='2.2'></a>
# ## <p style=" font-family:newtimeroman; font-size:110%; text-align:center">2.2 Déclaration des variables</p>

# %% [code]
numeric_columns = train.select_dtypes(['float','int']).columns
Cat_columns=train.select_dtypes('object').columns

# %% [markdown]
# <a id='3'></a>
# ## <p style=" font-family:newtimeroman; font-size:150%; text-align:center">3. Chiffres cles</p>

# %% [markdown]
# <a id='3.1'></a>
# ## <p style=" font-family:newtimeroman; font-size:110%; text-align:center">3.1 pandas_profiling</p>

# %% [markdown]
# ANCHOR Panda profiling
# Pandas profiling may take some times.

# %% [code]
# import pandas_profiling as pp
pp.ProfileReport(train)

# %% [markdown]
# <a id='3.2'></a>
# ## <p style=" font-family:newtimeroman; font-size:110%; text-align:center">3.2 Methode classique</p>

# %% [code]
train.describe()

# %% [code]
# REVIEW sort_values works on axis 1.
tmp = train.describe(include=['O'])
tmp.sort_values('unique', axis=1)

# %% [markdown]
# <a id='3.3'></a>
# ## <p style=" font-family:newtimeroman; font-size:110%; text-align:center">3.3 Variables cibles</p>

# %% [code]
train['target'].value_counts(normalize=True)*100

# %% [code]
sns.countplot(x="target", data=train,
                   facecolor=(0, 0, 0, 0),
                   linewidth=5,
                   edgecolor=sns.color_palette("dark", 3))

# %% [markdown]
# La variables Target = 26 %

# %% [markdown]
# <a id='3.4'></a>
# ## <p style=" font-family:newtimeroman; font-size:110%; text-align:center">3.4 Variables numériques</p>

# %% [code]
# train.hist(bins=50, figsize=(20,15))
# plt.show()

# %% [code]
# ANCHOR show cont. feature distributions
def show_dist_per_cont_col():
    positive_train = train[train['target'] == 1]
    negative_train = train[train['target'] == 0]
    for col in train.select_dtypes('float'):
        plt.figure(figsize=(3,3))
        sns.distplot(positive_train[col], label='positive')
        sns.distplot(negative_train[col], label='negative')
        plt.legend()

# %% [code]
def show_median_per_col_by_target():
    for col in train.select_dtypes('float') :
        Chiffre = train.groupby('target').agg({
            col : ['median']
        })
        print( f'{col :-<5} {Chiffre} ')

# %% [code]
# ANCHOR show correlation triangle
def show_correlation_triangle():
    def cp(n, b=220):
        return sns.diverging_palette(1, b, n=n)
    mask = np.zeros_like(train[numeric_columns].corr())
    mask[np.triu_indices_from(mask)] = True # Trick <---------------------------
    from pylab import rcParams
    rcParams['figure.figsize'] = (12,8)
    sns.heatmap(
        train[numeric_columns].corr(),
        cmap = cp(200),
        annot=True,
        mask=mask,
        center = 0,
    )
show_correlation_triangle()

# %% [markdown]
# <a id='3.5'></a>
# ## <p style=" font-family:newtimeroman; font-size:110%; text-align:center">3.5 Variables caractères</p>

# %% [markdown]
# 

# %% [code]
def show_cat_levels():
    for col in train.select_dtypes('object'):
        print(f'{col :-<5} {train[col].nunique():-<5} {train[col].unique()}')
show_cat_levels()

# %% [code]
Cinq_modalite_moins = [] # low cardinality
Cinq_modalite_plus = [] # high cardinality

for col in train.select_dtypes('object'):
    if train[col].nunique() <= 5 :
        Cinq_modalite_moins.append(col)
    else:
        Cinq_modalite_plus.append(col)

# %% [code]
for col in Cinq_modalite_moins:
    plt.figure(figsize=(3,3))
    train[col].value_counts().plot.pie()

# %% [code]
# ANCHOR shoe feature level inconsistency between datasets
for col in Cat_columns:
    if set(train[col].unique()) != set(test[col].unique()):
        print(f"Level inconsistency: {col}")

# %% [code]
train_cat10 = set(train['cat10'].unique())
test_cat10 = set(test['cat10'].unique())

print(f'Modalités dans le train mais pas dans le test: {train_cat10.difference(test_cat10)}.')
print(f'Modalités dans le test mais pas dans le train: {test_cat10.difference(train_cat10)}.')

# %% [code]
def list_high_cardinality(threshold=0):
    return [col for col in train.select_dtypes('object').columns if train.loc[:, col].nunique() > threshold]

# ANCHOR Bayesian target encoding
# Credits: https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features
# Bayesian target encoding with noise

def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))

def target_encode(
        trn_series=None, 
        tst_series=None, 
        target=None, 
        min_samples_leaf=1, # 100
        smoothing=1, # 10
        noise_level=0 # 0.01
):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior  
    
    min_samples_leaf defines a threshold where prior and target mean (for a given category value) have the same weight. 
    Below the threshold prior becomes more important and above mean becomes more important.    
    """ 
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean 
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior) # NOTE - The resulting encoded new feature!!
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index 
    # Similarly
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)

# %% [code]
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('Time taken : %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

# %% [code]
# del train['cat10']
# del test['cat10']
# Cat_columns=train.select_dtypes('object').columns

# %% [code]
# for col in Cinq_modalite_moins:
#     plt.figure(figsize=(3,3))
#     sns.countplot(x=col, hue="target", data=train,palette="Set3")

# %% [markdown]
# <a id='4'></a>
# ## <p style=" font-family:newtimeroman; font-size:150%; text-align:center">4. Modèlisation</p>

# %% [code]
# train_alea = train.sample(n=130000,random_state=0)

# %% [code]
# train.shape

# %% [code]
# train['target'].value_counts(normalize=True)*100

# %% [code]
# trainset, testset = train_test_split(train_alea, test_size=0.2, random_state=0)

# %% [code]
# print(trainset['target'].value_counts())
# print(testset['target'].value_counts())

# %% [code]
# for col in Cat_columns:
#     if set(trainset[col].unique()) != set(testset[col].unique()):
#         print(f"La liste des variables avec des modalitées différents entre le test et le train: {col}")

# %% [code]
# def imputation(df):
#     df = df.dropna(axis=0)
#     df = df.dropna(axis=1)
#     df.drop_duplicates(keep = 'first', inplace=True)
#     return  df

# %% [code]
# def encodage(df):
#     ohe = OneHotEncoder(sparse=False)
#     ohe.fit(df[Cat_columns])
#     df = pd.merge(df[numeric_columns], 
#           pd.DataFrame(columns = ohe.get_feature_names().tolist(),
#               data = ohe.fit_transform(df[Cat_columns])).set_index(df.index),
#         left_index = True, right_index = True)
#     return df

# %% [code]
# def preprocessing(df):
    
#     df = encodage(df)
#     df = imputation(df)
    
#     X = df.drop('target', axis=1)
#     y = df['target']
    
#     print(y.value_counts(normalize=True))
    
#     return X, y

# %% [code]
# X_train, y_train = preprocessing(trainset)

# %% [code]
# X_test, y_test = preprocessing(testset)

# %% [code] 
# ANCHOR loading packages
import re, pickle, random, os

import lightgbm as lgb
import xgboost as xgb
from catboost import Pool
from catboost import CatBoostClassifier
from sklearn.linear_model import ElasticNet, RidgeClassifier, LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
import optuna
optuna.logging.disable_default_handler()

# ensure consistent results
def seed_everything(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

RANDOM_STATE = 42
seed_everything(seed=RANDOM_STATE)

class Model:
    def __init__(self, method):
        self.method = method
        self.model = None
        self.categorical_features_indices = None

    def train(
        self,
        params,
        X,
        y,
        X_test,
        y_test=None,
        categorical_features_indices=None,
        nn_params=None,
    ):
        self.categorical_features_indices = categorical_features_indices

        # ANCHOR model lgb
        if self.method == "lgb":
            train_data = lgb.Dataset(X, label=y)
            eval_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

            model = lgb.train(
                params,
                train_data,
                valid_sets=[train_data, eval_data],
                valid_names=["train", "eval"],
                early_stopping_rounds=500,
                # n_estimators=1000,
                verbose_eval=1000
            )
            self.model = model

        # # XGBoost
        # elif self.method == "xgb":
        #     dtrain = xgb.DMatrix(X, label=y)
        #     model = xgb.train(params, dtrain)
        #     self.model = model

        # ANCHOR model catboost
        elif self.method == "catboost_classifier":
            train_pool = Pool(X, y, cat_features=categorical_features_indices) # train
            test_pool = Pool(X_test, y_test, cat_features=categorical_features_indices) # val not test
            model = CatBoostClassifier(**params)
            model.fit(train_pool, eval_set=test_pool) # test_pool is avtually our vlidatation data.
            self.model = model

        # # Random Forest
        # elif self.method == "rf":
        #     model = RandomForestClassifier(**params)
        #     model.fit(X, y)
        #     self.model = model

        # svm
        elif self.method == "svm":
            model = SVC(**params)
            model.fit(X, y)
            self.model = model
        
        elif self.method == 'ridge_classifier':
            model = RidgeClassifier(**params)
            model.fit(X, y)
            self.model = model

        # # logistic regression
        # elif self.method == "log":
        #     model = LogisticRegression(**params)
        #     model.fit(X, y)
        #     self.model = model

        # # neural network
        # elif self.method == "nn":
        #     model = nn_params.model
        #     torch_train = torch.utils.data.TensorDataset(
        #         torch.tensor(X, dtype=torch.float32),
        #         torch.tensor(y, dtype=torch.float32),
        #     )
        #     train_loader = torch.utils.data.DataLoader(
        #         torch_train, batch_size=nn_params.batch_size, shuffle=True
        #     )

        #     # モデルのトレーニング
        #     for epoch in range(nn_params.num_epochs):
        #         model.train()
        #         for i, (X_batch, y_batch) in enumerate(train_loader):
        #             y_pred = model(X_batch).view(-1)
        #             loss = nn_params.criterion(y_pred, y_batch)
        #             nn_params.optimizer.zero_grad()
        #             loss.backward()
        #             nn_params.optimizer.step()

        #             print(
        #                 "Epoch [%d/%d], Step [%d/%d], Loss: %.4f"
        #                 % (
        #                     epoch + 1,
        #                     nn_params.num_epochs,
        #                     i + 1,
        #                     len(X_batch) // nn_params.batch_size,
        #                     loss.data.item(),
        #                 )
        #             )

        else:
            raise Exception("method not found")

    def predict(self, X_test, predict_proba=False):
        # ANCHOR predict lgb
        if self.method == "lgb":
            pred = self.model.predict(X_test)
            if predict_proba:
                return pred
            else:
                return (pred > 0.5).astype(int)

        # # XGBoost
        # elif self.method == "xgb":
        #     dtest = xgb.DMatrix(X_test)
        #     pred = self.model.predict(dtest)
        #     if predict_proba:
        #         return pred
        #     else:
        #         return (pred > 0.5).astype(int)

        # ANCHOR predict catboost
        elif self.method == "catboost_classifier":
            test_pool = Pool(X_test, cat_features=self.categorical_features_indices)
            pred = self.model.predict(test_pool)
            if predict_proba:
                return self.model.predict_proba(test_pool)[:, -1]
            else:
                return self.model.predict(test_pool)

        # # Random Forest
        # elif self.method == "rf":
        #     if predict_proba:
        #         return self.model.predict_proba(X_test)[:, 1]
        #     else:
        #         return self.model.predict(X_test)

        # svm
        elif self.method == "svm":
            if predict_proba:
                self.model.probability = True
                return self.model.predict_proba(X_test)[:, 1]
            else:
                return self.model.predict(X_test)

        elif self.method == 'ridge_classifier': # LIPIN custom model
            if predict_proba:
                self.model.probability = True
                d = self.model.decision_function(X_test)
                return np.exp(d) / (1 + np.exp(d))
                # Ridge classifier api: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html
                # SO - covert rdige classifier's decision back to probability: https://stackoverflow.com/questions/22538080/scikit-learn-ridge-classifier-extracting-class-probabilities
            else:
                return self.model.predict(X_test)


        # # logistic regression
        # elif self.method == "log":
        #     if predict_proba:
        #         return self.model.predict_proba(X_test)[:, 1]
        #     else:
        #         return self.model.predict(X_test)

        else:
            raise Exception("method not found")

def asbinary(fp):            
    return (fp > 0.5).astype(int)

# %% [code]
# ANCHOR cross validation
def cross_validate(
    method,
    params,
    X=None,
    y=None,
    X_test=None,
    categorical_features_indices=None,
    predict_proba=False,
    model=None,
    n_splits = 5
):
    scores = np.zeros(n_splits)
    oof = np.zeros(len(y))
    if X_test is not None:
        pred_tests = np.zeros((len(X_test), n_splits))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    for i, (train_index, valid_index) in enumerate(skf.split(X, y)):

        X_train, y_train = X.iloc[train_index,:], y.iloc[train_index]
        X_valid, y_valid = X.iloc[valid_index,:], y.iloc[valid_index]

        model = model or Model(method)
        model.train(
            params,
            X_train,
            y_train,
            X_test=X_valid,
            y_test=y_valid,
            categorical_features_indices=categorical_features_indices
        )
        pred_valid = model.predict(X_valid, predict_proba=predict_proba)
        scores[i] = roc_auc_score(y_valid, asbinary(pred_valid)) # np.rint()
        oof[valid_index] = pred_valid

        if X_test is not None:
            pred_test = model.predict(X_test, predict_proba=predict_proba)
            pred_tests[:, i] = pred_test
    
    if X_test is None:
        return scores, oof
    else:
        return scores, oof, pred_tests.mean(axis=1)
    
# ------------------------------------ lgb ----------------------------------- #
def optuna_lgb1(trial):

    MIN_SAMPLES_LEAF = trial.suggest_discrete_uniform("min_samples_leaf", 1, 100, 1) # 294best: 0.2
    NOISE_LEVEL = trial.suggest_discrete_uniform("noise_level", 0.0, 1.0, 0.05) # 294best: 0.05
    
    trn_df = train.copy()
    tst_df = test.copy()
    for col in list_high_cardinality(0):
        trn_df[col], tst_df[col] = target_encode(trn_df[col], 
                                                 tst_df[col], 
                                                 trn_df['target'], 
                                                 min_samples_leaf=MIN_SAMPLES_LEAF, 
                                                 noise_level=NOISE_LEVEL
                                                )    
    param = {
        'class_weight': 'balanced',
        'alpha': 0.25938636330061576, # trial.suggest_loguniform('ridge_alpha', 0.001, 10 ** 5), # 294best: 0.25938636330061576
        'fit_intercept': True,
        'normalize': True
    }

    X = trn_df.drop('target', axis=1)
    y = trn_df['target']
    X_test = tst_df
    scores, oof, pred_test = cross_validate("ridge_classifier", param, X=X, y=y, X_test=X_test, predict_proba=True)    
    oof_auc_score = roc_auc_score(y, asbinary(oof))
    print(f"Oof score : {oof_auc_score}.")
    return oof_auc_score

# study_ridge = optuna.create_study(direction="maximize")
# study_ridge.optimize(optuna_lgb1, n_trials=ridge_trial)  # 1000
# print("Number of finished trials: {}".format(len(study_ridge.trials)))
# print("Best trial:")
# trial = study_ridge.best_trial
# print("  Value: {}".format(trial.value))
# print("  Params: ")
# for key, value in trial.params.items():
#     print("    {}: {}".format(key, value))


# ANCHOR optuna: ridge1

RIDGE1_TRIAL = 300

def optuna_ridge1(trial):
    MIN_SAMPLES_LEAF = trial.suggest_discrete_uniform("min_samples_leaf", 1, 100, 1) # 294best: 0.2
    NOISE_LEVEL = trial.suggest_discrete_uniform("noise_level", 0.0, 1.0, 0.05) # 294best: 0.05
    
    trn_df = train.copy()
    tst_df = test.copy()

    for col in list_high_cardinality(0):
        trn_df[col], tst_df[col] = target_encode(trn_df[col], 
                                                 tst_df[col], 
                                                 trn_df['target'], 
                                                 min_samples_leaf=MIN_SAMPLES_LEAF, 
                                                 noise_level=NOISE_LEVEL
                                                )    
    param = {
        'class_weight': 'balanced',
        'alpha': trial.suggest_loguniform('ridge_alpha', 0.001, 10 ** 5), # 294best: 0.25938636330061576
        'fit_intercept': True,
        'normalize': True
    }
    X = trn_df.drop('target', axis=1)
    y = trn_df['target']
    X_test = tst_df
    val_auc_scores, oof, pred_test = cross_validate(
        "ridge_classifier", 
        param, 
        X=X, 
        y=y, 
        X_test=X_test, 
        predict_proba=True)    
    oof_auc_score = roc_auc_score(y, asbinary(oof))
    print(f"Oof score : {oof_auc_score}.")
    return oof_auc_score

# study_ridge1 = optuna.create_study(direction="maximize")
# study_ridge1.optimize(optuna_ridge1, n_trials=RIDGE1_TRIAL)  # 1000
# print("Number of finished trials: {}".format(len(study_ridge1.trials)))
# print("Best trial:")
# trial = study_ridge1.best_trial
# print("  Value: {}".format(trial.value))
# print("  Params: ")
# for key, value in trial.params.items():
#     print("    {}: {}".format(key, value))

#%%
# ANCHOR optuna_catboost1
CATBOOST1_TRIAL = 10
def optuna_catboost1(trial):
    cat_vars = train.drop('target', axis=1).select_dtypes(exclude = ['float', 'int']).columns.to_list()
    num_vars = train.drop('target', axis=1).select_dtypes(include = ['float']).columns.to_list()
    trn_df = train.copy()
    tst_df = test.copy()
    trn_df[cat_vars] = trn_df[cat_vars].astype('category')
    tst_df[cat_vars] = tst_df[cat_vars].astype('category')
    X = trn_df.drop('target', axis=1)
    y = trn_df['target']
    X_test = tst_df
    categorical_features_indices = [train.columns.get_loc(col) for col in cat_vars]
    param = {
        'task_type': 'GPU',
        # 'learning_rate': 0.08, # trial.suggest_float('learning_rate', 1e-5, 1, log=True),
        # "random_state": RANDOM_STATE,
        # # 'l2_leaf_reg': trial.suggest_int("iterations", 2, 30),
        # "iterations": 2000, # trial.suggest_int("iterations", 50, 100),
        # "depth": 16, # trial.suggest_int("depth", 4, 16), # maximm depth is officially set to 16.
        # # "random_strength": trial.suggest_int("random_strength", 0, 100),
        # # "od_type": trial.suggest_categorical("od_type", ["IncToDec", "Iter"]),
        # "bootstrap_type": trial.suggest_categorical(
        #     "bootstrap_type", ["Bayesian"]), # ["Bayesian", "Bernoulli", "MVS"]),        
        "od_wait": 50, # trial.suggest_int("od_wait", 10, 100), # only one of the parameters od_wait, early_stopping_rounds should be initialized.
        "eval_metric": "AUC",
        "verbose": 500,
        'border_count': 32, # for speed up in GPU
        'use_best_model': True
    }

    # if param["bootstrap_type"] == "Bayesian":
    #     param["bagging_temperature"] = 0.14658448791284684 # trial.suggest_float("bagging_temperature", 0.01, 100)
    # elif param["bootstrap_type"] == "Bernoulli":
    #     param["subsample"] = trial.suggest_float("subsample", 0.1, 1)

    val_auc_scores, oof, pred_test = cross_validate("catboost_classifier", 
        param, 
        X=X, 
        y=y, 
        X_test=X_test,
        categorical_features_indices=categorical_features_indices,
        predict_proba=True,
        n_splits=3
    )    
    oof_auc_score = roc_auc_score(y, asbinary(oof))
    print(f"-------------------Oof score : {oof_auc_score}. val_auc_scores: {np.mean(val_auc_scores)}")
    return val_auc_scores # oof_auc_score

start_time = timer(None)
study = optuna.create_study(direction="maximize")
study.optimize(optuna_catboost1, n_trials=CATBOOST1_TRIAL)  # 1000
print("Number of finished trials: {}".format(len(study.trials)))
print("Best trial:")
trial = study.best_trial
print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))    
timer(start_time)

#%%
# # ------------------------------------ SVM ----------------------------------- #

# svm_trial = 150

# def optuna_svm(trial):
#     param = {
#         "random_state": RANDOM_STATE,
#         "C": trial.suggest_loguniform("C", 2 ** -10, 2 ** 15),
#         "gamma": trial.suggest_loguniform("gamma", 2 ** -20, 2 ** 10),
#     }
#     MIN_SAMPLES_LEAF = trial.suggest_discrete_uniform("min_samples_leaf", 0.1, 1.0, 0.1)
#     NOISE_LEVEL = trial.suggest_discrete_uniform("noise_level", 0.0, 1.0, 0.05)
    
#     trn_df = train.copy()
#     tst_df = test.copy()
#     for col in list_high_cardinality(0):
#         trn_df[col], tst_df[col] = target_encode(trn_df[col], 
#                                                  tst_df[col], 
#                                                  trn_df['target'], 
#                                                  min_samples_leaf=MIN_SAMPLES_LEAF, 
#                                                  noise_level=NOISE_LEVEL
#                                                 )
#     X = trn_df.drop('target', axis=1)
#     y = trn_df['target']
#     X_test = tst_df
#     _, oof = cross_validate("svm", param, X=X, y=y, X_test=X_test)

#     return roc_auc_score(y, oof)


# study_svm = optuna.create_study(direction="maximize")
# study_svm.optimize(optuna_svm, n_trials=svm_trial)  # 1000
# print("Number of finished trials: {}".format(len(study_svm.trials)))
# print("Best trial:")
# trial = study_svm.best_trial
# print("  Value: {}".format(trial.value))
# print("  Params: ")
# for key, value in trial.params.items():
#     print("    {}: {}".format(key, value))

# with open('./study_svm.pkl', 'wb') as f:
#     pickle.dump(study_svm, f)

# # %% [code]
# def evaluation(model):
    
#     model.fit(X_train, y_train)
#     ypred = model.predict(X_test)
    
#     print(confusion_matrix(y_test, ypred))
#     print(classification_report(y_test, ypred))
#     print(roc_auc_score(y_test, ypred))
    
#     #N, train_score, val_score = learning_curve(model, X_train, y_train,
#     #                                          cv=2, scoring='accuracy',
#     #                                           train_sizes=np.linspace(0.1, 1, 10))
    
    
#     #plt.figure(figsize=(12, 8))
#     #plt.plot(N, train_score.mean(axis=1), label='train score')
#     #plt.plot(N, val_score.mean(axis=1), label='validation score')
#     #plt.legend()

# # %% [code]
# preprocessor = make_pipeline(SelectKBest(f_classif, k=10))

# # %% [code]
# RandomForest = make_pipeline(RandomForestClassifier(random_state=0))
# AdaBoost = make_pipeline( AdaBoostClassifier(random_state=0))
# SVM = make_pipeline(preprocessor, StandardScaler(), SVC(random_state=0))
# ridge = make_pipeline(RidgeClassifier(random_state=0))
# Bagging = make_pipeline(BaggingClassifier(random_state=0))
# Gradient= make_pipeline(GradientBoostingClassifier(random_state=0))

# #VotingClassifier =make_pipeline(VotingClassifier(random_state=0))

# # %% [code]
# dict_of_models = {'Bagging' : Bagging,
#                   'Gradient' : Gradient,
#                   'ridge' : ridge, 
#                   'AdaBoost' : AdaBoost,
#                   #'SVM': SVM,
#                   'RandomForest': RandomForest
                  
#                  }

# # %% [code]
# for name, model in dict_of_models.items():
#     print(name)
#     evaluation(model)

# # %% [code]
# estimators = [
# ('RandomForest' ,make_pipeline(RandomForestClassifier(random_state=0))),
# ('AdaBoost' ,make_pipeline( AdaBoostClassifier(random_state=0))),
# ('ridge' , make_pipeline(RidgeClassifier(random_state=0))),
# ('Gradient', make_pipeline(GradientBoostingClassifier(random_state=0)))
#  ]

# # %% [code]
# #StackingClassifier =make_pipeline(StackingClassifier(random_state=0))
# clf = StackingClassifier(estimators=estimators)
# clf.fit(X_train, y_train).score(X_test, y_test)

# # %% [code]
# clf.predict()

# # %% [code]
# y_pred = clf.predict(X_test)

# print(roc_auc_score(y_test, ypred))

# # %% [code]
# evaluation(clf)

# # %% [markdown]
# # <a id='5'></a>
# # ## <p style=" font-family:newtimeroman; font-size:150%; text-align:center">5. Optimisation</p>

# # %% [code]
# RandomForest.get_params().keys()

# # %% [code]
# hyper_params = {'randomforestclassifier__n_estimators':[1, 5,100,20,30 ],
#                'randomforestclassifier__max_depth' : [1,2,3,4,5],
#                'randomforestclassifier__n_jobs' : [-1,1]}

# # %% [code]
# grid = RandomizedSearchCV(RandomForest, hyper_params, scoring='accuracy', cv=4,
#                           n_iter=10)

# grid.fit(X_train, y_train)

# # %% [code]
# print(grid.best_params_)

# y_pred = grid.predict(X_test)

# print(classification_report(y_test, y_pred))

# # %% [code]
# evaluation(grid.best_estimator_)

# # %% [code]
# AdaBoost = make_pipeline( AdaBoostClassifier(random_state=0,n_estimators=500,learning_rate=1.3))

# # %% [code]
# evaluation(AdaBoost)

# # %% [code]
# from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# # %% [code]
# hyper_params = {'adaboostclassifier__n_estimators':[1, 50,100,150,500],
#                'adaboostclassifier__learning_rate' : [1.1,1.2,1.3,1.4,1.5],
#                'adaboostclassifier__algorithm' : ['SAMME','SAMME.R']}

# # %% [code]
# AdaBoost.get_params().keys()

# # %% [code]
# grid = RandomizedSearchCV(AdaBoost, hyper_params, scoring='recall', cv=4,
#                           n_iter=10)

# grid.fit(X_train, y_train)

# # %% [code]
# RandomForest.get_params().keys()

# # %% [code]
# hyper_params = {'adaboostclassifier__n_estimators':[1, 50,100,150,500],
#                'adaboostclassifier__learning_rate' : [1.1,1.2,1.3,1.4,1.5],
#                'adaboostclassifier__algorithm' : ['SAMME','SAMME.R']}

# # %% [code]
# grid = RandomizedSearchCV(AdaBoost, hyper_params, scoring='recall', cv=4,
#                           n_iter=10)

# grid.fit(X_train, y_train)

# # %% [code]
# print(grid.best_params_)

# y_pred = grid.predict(X_test)

# print(classification_report(y_test, y_pred))

# # %% [code]
# evaluation(grid.best_estimator_)

# # %% [code]
# from sklearn.metrics import precision_recall_curve

# # %% [code]
# precision, recall, threshold = precision_recall_curve(y_test, grid.best_estimator_.decision_function(X_test))

# # %% [code]
# plt.plot(threshold, precision[:-1], label='precision')
# plt.plot(threshold, recall[:-1], label='recall')
# plt.legend()

# # %% [code]
# def model_final(model, X, threshold=0):
#     return model.decision_function(X) > threshold

# # %% [code]
# y_pred = model_final(grid.best_estimator_, X_test, threshold=0.5)

# # %% [code]
# f1_score(y_test, y_pred)

# # %% [code]
# print(confusion_matrix(y_test, y_pred))

# # %% [code]
# test = pd.read_csv('/kaggle/input/tabular-playground-series-mar-2021/test.csv')

# # %% [code]
# test.set_index('id',inplace=True)

# # %% [code]
# numeric_columns = train.select_dtypes(['float']).columns

# # %% [code]
# test = encodage(test)
# test = imputation(test)

# # %% [code]
# ypred = model_final(grid.best_estimator_, test, threshold=0.5)

# # %% [code]
# ypred = pd.DataFrame(data=ypred, columns=['target2'])

# # %% [code]
# ypred['target']= np.where(ypred['target2']==True,1,0)

# # %% [code]
# test.reset_index(inplace=True)

# # %% [code]
# test=pd.merge(test,ypred,how='left',left_index=True,right_index=True)

# # %% [code]
# test['target'].value_counts(normalize=True)

# # %% [code]
# sub = test[['id','target']]

# # %% [code]
# sub.to_csv('submission.csv', index=False)

# # %% [code]
# n_folds = 10
# seed_list = [i for i in range(2000, 2022)]
