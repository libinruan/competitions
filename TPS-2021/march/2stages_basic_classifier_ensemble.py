from tensorflow.keras import layers, optimizers, callbacks, utils, losses, metrics, backend as K
from sklearn import metrics as skmetrics, preprocessing
from tensorflow.keras.models import Model, load_model
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA, FactorAnalysis    
from umap import UMAP
from scipy.stats import rankdata
import os, gc, joblib, warnings, time, random
import tensorflow_addons as tfa
import tensorflow as tf
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

def seed_everything(seed=1903):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


# SECTION preprocessing
# --------------------------------- RAW DATA --------------------------------- #
os.chdir('/kaggle/working')
train = pd.read_csv('../input/tabular-playground-series-mar-2021/train.csv')
test = pd.read_csv('../input/tabular-playground-series-mar-2021/test.csv')
sample_submission = pd.read_csv('../input/tabular-playground-series-mar-2021/sample_submission.csv')

print(train.shape) # (300000, 31)
print(test.shape)  # (200000, 30)
print(sample_submission.shape) # (200000, 2)

sparse_features = [col for col in train.columns if col.startswith('cat')] # 19
dense_features = [col for col in train.columns if col not in sparse_features+['target']] # 11
dense_data = pd.concat([train[dense_features], test[dense_features]], axis=0)

test['target'] = -1
data = pd.concat([train, test], axis=0)

# train.isnull().sum() # no missing values
# test.isnull().sum() # same as above

# Some categories have lots of levels. ss: https://i.postimg.cc/SKL9p5y3/2021-03-11-at-11-43-02.png
def table_unique_level_count_sparse_feature():
    tmp = []
    for c in sparse_features:
        tmp.append(pd.DataFrame({c: [data[c].nunique()]}, index=['count']))
    pd.concat(tmp, axis=1)

def plot_cat10_high_cardinalirty():
    sns.set_style("whitegrid")
    plt.figure(figsize=(30,5))
    ax = sns.countplot(x='cat10', data=train, hue='target',palette='mako')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
    plt.tight_layout()
    plt.xticks(
        rotation=45, 
        horizontalalignment='right',
        fontweight='light',
        fontsize='x-small'  
    );

# ANCHOR Handle test data with XOR feature values (categorical features preprocessing)
def check_whether_exclusive_sparse_features():
    # NOTE: For each categorical features, silence levels not shared by both of train and test sets.
    # For tabular-march-2021 data, this code block is useless; no train or test only levels of features exists. ss: https://i.postimg.cc/SNnW3DH7/2021-03-11-at-12-06-15.png
    tmp = pd.DataFrame(index=['train_only','test_only','both'])
    for col in sparse_features:
        train_only = list(set(train[col].unique()) - set(test[col].unique()))
        test_only = list(set(test[col].unique()) - set(train[col].unique()))
        both = list(set(test[col].unique()).union(set(train[col].unique())))
        tmp[col] = [len(train_only), len(test_only), len(both)]
        train.loc[train[col].isin(train_only), col] = np.nan # Silence those levels only shown in train set
        test.loc[test[col].isin(test_only), col] = np.nan # Silence those levels only shown in train set
        # if any(train[col].isin(train_only)): print('yes') # tmp['train', col] = sum(train[col].isin(train_only))
        # if any(test[col].isin(test_only)): print('yes') # tmp['test', col] = sum(test[col].isin(test_only))
        mode = train[col].mode().values[0]
        train[col] = train[col].fillna(mode)
        test[col] = test[col].fillna(mode)    
    tmp

# ------------------------------ PCA EXPERIMENT ------------------------------ #
def plot_pca_on_dense_features():
    from sklearn.preprocessing import StandardScaler
    # from sklearn.linear_model import LogisticRegression
    # logisticRegr = LogisticRegression(solver = 'lbfgs') # default solver is too slower.

    scaler = StandardScaler()
    scaler.fit(dense_data)
    pca = PCA().fit(scaler.transform(dense_data))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance') # It shows n_components=10 the cumulative explained variance reaches 100%.

# !SECTION preprocessing

# SECTION cross validation
#%% 
# LINK (code source): https://www.kaggle.com/craigmthomas/tps-mar-2021-stacked-starter
from category_encoders import LeaveOneOutEncoder, TargetEncoder 
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder # requires ordinal feature

cat_features = [col for col in train.columns if col.startswith('cat')] # 19
cont_features = [col for col in train.columns if col not in sparse_features+['target']] # 11
target = train['target']

xgb_cat_features = []
lgb_cat_features = []
cb_cat_features = []
ridge_cat_features = []
sgd_cat_features = []
hgbc_cat_features = [] # for HistGradientBoosting clasifier

loo_features = [] # from leave one out encoder
le_features = [] # from label encoder

def label_encode(train_df, test_df, column):
    le = LabelEncoder()
    new_feature = "{}_le".format(column)
    le.fit(train_df[column].unique().tolist() + test_df[column].unique().tolist())
    train_df[new_feature] = le.transform(train_df[column])
    test_df[new_feature] = le.transform(test_df[column])
    return new_feature

def loo_encode(train_df, test_df, column): 
    loo = LeaveOneOutEncoder()
    new_feature = "{}_loo".format(column)
    loo.fit(train_df[column], train_df["target"])
    train_df[new_feature] = loo.transform(train_df[column])
    test_df[new_feature] = loo.transform(test_df[column])
    return new_feature

for feature in cat_features:
    loo_features.append(loo_encode(train, test, feature))
    le_features.append(label_encode(train, test, feature))

xgb_cat_features.extend(loo_features)
ridge_cat_features.extend(loo_features)
sgd_cat_features.extend(loo_features) 
hgbc_cat_features.extend(loo_features)
cb_cat_features.extend(cat_features) # NOTE Catboost can take care of categorical encoding itself.
lgb_cat_features.extend(le_features) # NOTE lightGBM requries to get cateogrical feature label encoded beforehand.

# Generate level one model
import warnings
warnings.filterwarnings("ignore")

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

random_state = 2021
n_folds = 10
k_fold = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=True)

xgb_train_preds = np.zeros(len(train.index), ) # the same as np.zeros(len(train.index))
xgb_test_preds = np.zeros(len(test.index), )
xgb_features = xgb_cat_features + cont_features

lgb_train_preds = np.zeros(len(train.index), )
lgb_test_preds = np.zeros(len(test.index), )
lgb_features = lgb_cat_features + cont_features

cb_train_preds = np.zeros(len(train.index), )
cb_test_preds = np.zeros(len(test.index), )
cb_features = cb_cat_features + cont_features

ridge_train_preds = np.zeros(len(train.index), )
ridge_test_preds = np.zeros(len(test.index), )
ridge_features = ridge_cat_features + cont_features # FIXME need to scale numerical features.

sgd_train_preds = np.zeros(len(train.index), )
sgd_test_preds = np.zeros(len(test.index), )
sgd_features = sgd_cat_features + cont_features # FIXME need to  use scaling and PCA with SVM...

hgbc_train_preds = np.zeros(len(train.index), )
hgbc_test_preds = np.zeros(len(test.index), )
hgbc_features = hgbc_cat_features + cont_features

# ANCHOR Cross validation code
for fold, (train_index, test_index) in enumerate(k_fold.split(train, target)):
    print("--> Fold {}".format(fold + 1))
    y_train = target.iloc[train_index]
    y_valid = target.iloc[test_index]

    xgb_x_train = pd.DataFrame(train[xgb_features].iloc[train_index])
    xgb_x_valid = pd.DataFrame(train[xgb_features].iloc[test_index])

    lgb_x_train = pd.DataFrame(train[lgb_features].iloc[train_index])
    lgb_x_valid = pd.DataFrame(train[lgb_features].iloc[test_index])

    cb_x_train = pd.DataFrame(train[cb_features].iloc[train_index])
    cb_x_valid = pd.DataFrame(train[cb_features].iloc[test_index])

    ridge_x_train = pd.DataFrame(train[ridge_features].iloc[train_index])
    ridge_x_valid = pd.DataFrame(train[ridge_features].iloc[test_index])

    sgd_x_train = pd.DataFrame(train[sgd_features].iloc[train_index])
    sgd_x_valid = pd.DataFrame(train[sgd_features].iloc[test_index])

    hgbc_x_train = pd.DataFrame(train[hgbc_features].iloc[train_index])
    hgbc_x_valid = pd.DataFrame(train[hgbc_features].iloc[test_index])

    xgb_model = XGBClassifier(
        seed=random_state,
        n_estimators=10000,
        verbosity=1,
        eval_metric="auc",
        tree_method="gpu_hist",
        gpu_id=0,
        alpha=7.105038963844129,
        colsample_bytree=0.25505629740052566,
        gamma=0.4999381950212869,
        reg_lambda=1.7256912198205319,
        learning_rate=0.011823142071967673,
        max_bin=338,
        max_depth=8,
        min_child_weight=2.286836198630466,
        subsample=0.618417952155855,
    )
    xgb_model.fit(
        xgb_x_train,
        y_train,
        eval_set=[(xgb_x_valid, y_valid)], 
        verbose=0,
        early_stopping_rounds=200
    )

    train_oof_preds = xgb_model.predict_proba(xgb_x_valid)[:,1]
    test_oof_preds = xgb_model.predict_proba(test[xgb_features])[:,1]
    xgb_train_preds[test_index] = train_oof_preds
    xgb_test_preds += test_oof_preds / n_folds
    print(": XGB - ROC AUC Score = {}".format(roc_auc_score(y_valid, train_oof_preds, average="micro")))

    lgb_model = LGBMClassifier(
        cat_feature=[x for x in range(len(lgb_cat_features))],
        random_state=random_state,
        cat_l2=25.999876242730252,
        cat_smooth=89.2699690675538,
        colsample_bytree=0.2557260109926193,
        early_stopping_round=200,
        learning_rate=0.00918685483594994,
        max_bin=788,
        max_depth=81,
        metric="auc",
        min_child_samples=292,
        min_data_per_group=177,
        n_estimators=1600000,
        n_jobs=-1,
        num_leaves=171,
        reg_alpha=0.7115353581785044,
        reg_lambda=5.658115293998945,
        subsample=0.9262904583735796,
        subsample_freq=1,
        verbose=-1,
    )
    lgb_model.fit(
        lgb_x_train,
        y_train,
        eval_set=[(lgb_x_valid, y_valid)], 
        verbose=0,
    )

    train_oof_preds = lgb_model.predict_proba(lgb_x_valid)[:,1]
    test_oof_preds = lgb_model.predict_proba(test[lgb_features])[:,1]
    lgb_train_preds[test_index] = train_oof_preds
    lgb_test_preds += test_oof_preds / n_folds
    print(": LGB - ROC AUC Score = {}".format(roc_auc_score(y_valid, train_oof_preds, average="micro")))

    cb_model = CatBoostClassifier(
        verbose=0,
        eval_metric="AUC",
        loss_function="Logloss",
        random_state=random_state,
        num_boost_round=20000,
        od_type="Iter",
        od_wait=200,
        task_type="GPU",
        devices="0",
        cat_features=[x for x in range(len(cb_cat_features))],
        bagging_temperature=1.288692494969795,
        grow_policy="Depthwise",
        l2_leaf_reg=9.847870133539244,
        learning_rate=0.01877982653902465,
        max_depth=8,
        min_data_in_leaf=1,
        penalties_coefficient=2.1176668909602734,
    )
    cb_model.fit(
        cb_x_train,
        y_train,
        eval_set=[(cb_x_valid, y_valid)], 
        verbose=0,
    )

    train_oof_preds = cb_model.predict_proba(cb_x_valid)[:,1]
    test_oof_preds = cb_model.predict_proba(test[cb_features])[:,1]
    cb_train_preds[test_index] = train_oof_preds
    cb_test_preds += test_oof_preds / n_folds
    print(": CB - ROC AUC Score = {}".format(roc_auc_score(y_valid, train_oof_preds, average="micro")))
    
    ridge_model = CalibratedClassifierCV(
        RidgeClassifier(random_state=random_state),
        cv=3,
    )
    ridge_model.fit(
        ridge_x_train,
        y_train,
    )

    train_oof_preds = ridge_model.predict_proba(ridge_x_valid)[:,-1]
    test_oof_preds = ridge_model.predict_proba(test[ridge_features])[:,-1]
    ridge_train_preds[test_index] = train_oof_preds
    ridge_test_preds += test_oof_preds / n_folds
    print(": Ridge - ROC AUC Score = {}".format(roc_auc_score(y_valid, train_oof_preds, average="micro")))
    
    sgd_model = CalibratedClassifierCV(
        SGDClassifier(
            random_state=random_state,
            n_jobs=-1,
            loss="squared_hinge",
        ),
        cv=3,
    )
    sgd_model.fit(
        sgd_x_train,
        y_train,
    )

    train_oof_preds = sgd_model.predict_proba(sgd_x_valid)[:,-1]
    test_oof_preds = sgd_model.predict_proba(test[sgd_features])[:,-1]
    sgd_train_preds[test_index] = train_oof_preds
    sgd_test_preds += test_oof_preds / n_folds
    print(": SGD - ROC AUC Score = {}".format(roc_auc_score(y_valid, train_oof_preds, average="micro")))
    
    hgbc_model = HistGradientBoostingClassifier(
        l2_regularization=1.766059063693552,
        learning_rate=0.10675193678150449,
        max_bins=128,
        max_depth=31,
        max_leaf_nodes=185,
        random_state=2021
    )
    hgbc_model.fit(
        hgbc_x_train,
        y_train,
    )

    train_oof_preds = hgbc_model.predict_proba(hgbc_x_valid)[:,-1]
    test_oof_preds = hgbc_model.predict_proba(test[hgbc_features])[:,-1]
    hgbc_train_preds[test_index] = train_oof_preds
    hgbc_test_preds += test_oof_preds / n_folds
    print(": HGBC - ROC AUC Score = {}".format(roc_auc_score(y_valid, train_oof_preds, average="micro")))
    print("")
    
print("--> Overall metrics")
print(": XGB - ROC AUC Score = {}".format(roc_auc_score(target, xgb_train_preds, average="micro")))
print(": LGB - ROC AUC Score = {}".format(roc_auc_score(target, lgb_train_preds, average="micro")))
print(": CB - ROC AUC Score = {}".format(roc_auc_score(target, cb_train_preds, average="micro")))
print(": Ridge - ROC AUC Score = {}".format(roc_auc_score(target, ridge_train_preds, average="micro")))
print(": SGD - ROC AUC Score = {}".format(roc_auc_score(target, sgd_train_preds, average="micro")))
print(": HGBC - ROC AUC Score = {}".format(roc_auc_score(target, hgbc_train_preds, average="micro")))

# !SECTION cross validation


# Build Level 2 Model
# SECTION Second stage model
from scipy.special import expit
from sklearn.calibration import CalibratedClassifierCV

random_state = 2021
n_folds = 10
k_fold = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=True)

l1_train = pandas.DataFrame(data={
    "xgb": xgb_train_preds.tolist(),
    "lgb": lgb_train_preds.tolist(),
    "cb": cb_train_preds.tolist(),
    "ridge": ridge_train_preds.tolist(),
    "sgd": sgd_train_preds.tolist(),
    "hgbc": hgbc_train_preds.tolist(),
    "target": target.tolist()
})
l1_test = pandas.DataFrame(data={
    "xgb": xgb_test_preds.tolist(),
    "lgb": lgb_test_preds.tolist(),
    "cb": cb_test_preds.tolist(),
    "sgd": sgd_test_preds.tolist(),
    "ridge": ridge_test_preds.tolist(),    
    "hgbc": hgbc_test_preds.tolist(),
})

train_preds = numpy.zeros(len(l1_train.index), )
test_preds = numpy.zeros(len(l1_test.index), )
features = ["xgb", "lgb", "cb", "ridge", "sgd", "hgbc"]

for fold, (train_index, test_index) in enumerate(k_fold.split(l1_train, target)):
    print("--> Fold {}".format(fold + 1))
    y_train = target.iloc[train_index]
    y_valid = target.iloc[test_index]

    x_train = pandas.DataFrame(l1_train[features].iloc[train_index])
    x_valid = pandas.DataFrame(l1_train[features].iloc[test_index])
    
    model = CalibratedClassifierCV(
        RidgeClassifier(random_state=random_state), 
        cv=3
    )
    model.fit(
        x_train,
        y_train,
    )

    train_oof_preds = model.predict_proba(x_valid)[:,-1]
    test_oof_preds = model.predict_proba(l1_test[features])[:,-1]
    train_preds[test_index] = train_oof_preds
    test_preds += test_oof_preds / n_folds
    print(": ROC AUC Score = {}".format(roc_auc_score(y_valid, train_oof_preds, average="micro")))
    print("")
    
print("--> Overall metrics")
print(": ROC AUC Score = {}".format(roc_auc_score(target, train_preds, average="micro")))

submission = pandas.read_csv("../input/tabular-playground-series-mar-2021/sample_submission.csv")
submission["target"] = test_preds.tolist()
submission.to_csv("submission.csv", index=False)

# !SECTION Second stage model

#%%
import lightgbm as lgbm
from sklearn.model_selection import StratifiedKFold
y = train["target"]
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
oof = np.zeros(len(train))
score_list = []
fold = 1
test_preds = []
seed_list = [None, 2, 3] # Use more. Original list: [None,2,3,4,5]


for train_index, test_index in kf.split(train, y):
    X_train, X_val = train.iloc[train_index], train.iloc[test_index]
    y_train, y_val = y.iloc[train_index], y.iloc[test_index]
    
    # X_train = X_train.abs() # Taking aboslute was also a bit improving.
    y_pred_list = []
    for seed in seed_list:
        dtrain = lgbm.Dataset(X_train[sparse_features], y_train)
        dvalid = lgbm.Dataset(X_val[sparse_features], y_val)
        print(seed)
        params = {"objective": "regression",
              "metric": "rmse",
              "verbosity": -1,
              "boosting_type": "gbdt",
              "feature_fraction":0.5,
              "num_leaves": 250,
              "lambda_l1":7,
              "lambda_l2":2,
              "learning_rate":0.01,
              'min_child_samples': 35,
              "bagging_fraction":0.75,
              "bagging_freq":1,
             }
        params["seed"] = seed
        model = lgbm.train(params,
                        dtrain,
                        valid_sets=[dtrain, dvalid],
                        verbose_eval=100,
                        num_boost_round=100000,
                        early_stopping_rounds=100
                    )

        dtrain = lgbm.Dataset(X_train[cont_features], y_train)
        dvalid = lgbm.Dataset(X_val[cont_features], y_val)
        params = {"objective": "regression",
                  "metric": "rmse",
                  "verbosity": -1,
                  "boosting_type": "gbdt",
                  "feature_fraction":0.5,
                  "num_leaves": 350,
                  "lambda_l1":7,
                  "lambda_l2":1,
                  "learning_rate":0.003,
                  'min_child_samples': 35,
                  "bagging_fraction":0.8,
                  "bagging_freq":1,
                 }
        
        params["seed"] = seed
        model = lgbm.train(params,
                            dtrain,
                            valid_sets=[dtrain, dvalid],
                            verbose_eval=100,
                            num_boost_round=100000,
                            early_stopping_rounds=200,
                           init_model = model
                        )

    
    
        y_pred_list.append(model.predict(X_val[cont_features]))
        print(np.sqrt(mean_squared_error(y_val,   np.mean(y_pred_list,axis=0)       )))
        test_preds.append(model.predict(test[cont_features]))
        
    
   
    
    oof[test_index] = np.mean(y_pred_list,axis=0)    
    score = np.sqrt(mean_squared_error(y_val, oof[test_index]))
    score_list.append(score)
    print(f"RMSE Fold-{fold} : {score}")
    fold+=1

np.mean(score_list)
