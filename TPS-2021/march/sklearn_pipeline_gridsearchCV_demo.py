#%%
import random, os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import ElasticNet, RidgeClassifier, LogisticRegression
# from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
# from sklearn.model_selection import GridSearchCV
from sklearn.base import TransformerMixin, BaseEstimator
from category_encoders.target_encoder import TargetEncoder
from category_encoders import CatBoostEncoder
from category_encoders.glmm import GLMMEncoder # pretty slow...score lower
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.compose import make_column_selector as selector
# from category_encoders.m_estimate import MEstimateEncoder
# from sklearn.linear_model import ElasticNet,LogisticRegression, Ridge, RidgeClassifier
# from sklearn.metrics import make_scorer
# from sklearn.metrics import balanced_accuracy_score

def seed_everything(seed=1903):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(seed=2020)

os.chdir('/kaggle/working')
train = pd.read_csv('../input/tabular-playground-series-mar-2021/train.csv')
test = pd.read_csv('../input/tabular-playground-series-mar-2021/test.csv')
sample_submission = pd.read_csv('../input/tabular-playground-series-mar-2021/sample_submission.csv')

select_numeric_features = selector(dtype_include='number')
numeric_features = select_numeric_features(train) # 記得 scaleing for linear models with regularization. Without regularization, linear models doesn't need to be scaled simply for prediction.

train_id = train.loc[:,'id']
test_id = test.loc[:, 'id']
train.drop(['id'], axis=1, inplace=True)
test.drop(['id'], axis=1, inplace=True)

cat_features = selector(dtype_exclude='number')(train.drop('target', axis=1))
num_features = selector(dtype_include='number')(train.drop('target', axis=1))

cat_preprocessor = Pipeline(steps=[
    ('oh', OneHotEncoder(handle_unknown='ignore')),
    ('ss', StandardScaler(with_mean=False))
])
num_preprocessor = Pipeline(steps=[ 
    ('pt', PowerTransformer(method='yeo-johnson')),
    ('ss', StandardScaler())                                   
]) 
preprocessor = ColumnTransformer(transformers=[ 
    ('cat', cat_preprocessor, cat_features),
    ('num', num_preprocessor, num_features)                                                       
])
model = Pipeline(steps=[
    ('prep', preprocessor),
    ('clf', RidgeClassifier(class_weight='balanced', alpha=1000, fit_intercept=False))
])

X = train.drop('target', axis=1)
y = train['target']
# scores = cross_val_score(model, X, y, scoring='roc_auc', cv=2, n_jobs=-1)
# scores
skf = StratifiedKFold(n_splits=2, shuffle=True)
param_grid = {
    'clf': [RidgeClassifier(class_weight='balanced', alpha=1000, fit_intercept=False)], # ElasticNet(alpha=1.0, fit_intercept=False) too slow. RidgeClassifier(class_weight='balanced', alpha=1000, fit_intercept=False), 
    'prep__cat__oh': [OneHotEncoder(handle_unknown='ignore')], # TargetEncoder(handle_unknown='value'); GLMMEncoder()]: TargetEncoder, GLMMEncoder are not good.
    'prep__cat__ss': [StandardScaler(with_mean=False)],
    'prep__num__pt': [PowerTransformer(method='yeo-johnson')], # QuantileTransformer() is not good.
    'clf__alpha': [0.001]
    
    # 'clf__l1_ratio': [0.1, 0.3, 0.5],
    # 'clf__fit_intercept': [True] # Ridge: False is 0.881, True is 0.119; Elastic: True is better.
}
gs = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=4,
    cv=skf
)
gs.fit(X, y)
pd.DataFrame(gs.cv_results_).iloc[:,-3:] 
gs.best_params_
gs.cv_results_
# %%
