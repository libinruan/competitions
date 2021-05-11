#%%
import random, os
import numpy as np
import pandas as pd
import seaborn as sns
import optuna
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
from sklearn.preprocessing import FunctionTransformer
# from category_encoders.m_estimate import MEstimateEncoder
# from sklearn.linear_model import ElasticNet,LogisticRegression, Ridge, RidgeClassifier
# from sklearn.metrics import make_scorer
# from sklearn.metrics import balanced_accuracy_score

RunningInCOLAB = 'google.colab' in str(get_ipython())
if RunningInCOLAB:
    # print('Running on CoLab')
    os.chdir('/kaggle/working')
    train = pd.read_csv('../input/tabular-playground-series-mar-2021/train.csv')
    test = pd.read_csv('../input/tabular-playground-series-mar-2021/test.csv')
    sample_submission = pd.read_csv('../input/tabular-playground-series-mar-2021/sample_submission.csv')
else:
    # print('Not running on CoLab')
    os.chdir('G:\kagglePlayground')
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    sample_submission = pd.read_csv('sample_submission.csv')    

def seed_everything(seed=1903):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(seed=2020)

def table_unique_level_count_sparse_feature(df):
    tmp = [] # alternatively, dict approach
    for c in df.select_dtypes(['object','category']):
        tmp.append(pd.DataFrame([df[c].nunique()], index=[c], columns=['count']))
    return pd.concat(tmp, axis=0).sort_values(by=['count'])
# tmp = table_unique_level_count_sparse_feature(train)
# tmp.transpose()


select_numeric_features = selector(dtype_include='number')
numeric_features = select_numeric_features(train) # 記得 scaleing for linear models with regularization. Without regularization, linear models doesn't need to be scaled simply for prediction.

train_id = train.loc[:,'id']
test_id = test.loc[:, 'id']
train.drop(['id'], axis=1, inplace=True)
test.drop(['id'], axis=1, inplace=True)

def ridgeCLF_objective(trial):
    seed_everything(seed=2020)
    cat_features = selector(dtype_exclude='number')(train.drop('target', axis=1))
    num_features = selector(dtype_include='number')(train.drop('target', axis=1))

    #categorical features zone
    cat_preprocessor = Pipeline(steps=[
        ('oh', OneHotEncoder(handle_unknown='ignore')),
        ('ss', StandardScaler(with_mean=False))
    ])

    # MAX_OF_CARDINALITY = trial.suggest_categorical('max_cardi', [100])
    # def get_low_cardinality_features(df):
    #     cols = df \
    #         .select_dtypes(['object', 'category']) \
    #         .apply(lambda col: col.nunique()) \
    #         .loc[lambda x: x <= MAX_OF_CARDINALITY] \
    #         .index.tolist()     
    #     return df.loc[:, cols]

    # cat_low_cardi_preprocessor = Pipeline([
    #     ('cat_low', FunctionTransformer(func=get_low_cardinality_features)),
    #     ('oh', OneHotEncoder(handle_unknown='ignore')),
    #     ('ss', StandardScaler(with_mean=False))        
    # ])    

    # def get_high_cardinality_features(df):
    #     cols = df \
    #         .select_dtypes(['object', 'category']) \
    #         .apply(lambda col: col.nunique()) \
    #         .loc[lambda x: x > MAX_OF_CARDINALITY] \
    #         .index.tolist()     
    #     return df.loc[:, cols]    

    # SMOOTHING = 0.2182996635284694 # trial.suggest_float('smooth', 0.001, 1.0)
    # cat_high_cardi_preprocessor = Pipeline([
    #     ('cat_high', FunctionTransformer(func=get_high_cardinality_features)),
    #     ('te', TargetEncoder(smoothing=SMOOTHING)),
    #     ('ss', StandardScaler(with_mean=False))        
    # ])    

    def generate_num_polynomial(X):
        cols = X.columns
        for i in range(len(cols)-1):
            for j in range(i+1, len(cols)):
                colname = cols[i] + '_' + cols[j]
                X[colname] = X[cols[i]] * X[cols[j]]
        for i in range(len(cols)-1):
            colname= cols[i] + '^2'
            X[colname] = X[cols[i]].pow(2)
        return X

    num_polynomial = Pipeline([
        ('interact', FunctionTransformer(func=generate_num_polynomial))
    ])        
    num_polynomial_switch = trial.suggest_categorical('ph', [True])

    # numerical features zone
    if num_polynomial_switch:
        num_preprocessor = Pipeline(steps=[ 
            ('ac', num_polynomial),
            ('pt', PowerTransformer(method='yeo-johnson')),
            ('ss', StandardScaler())                                   
        ])
    else:        
        num_preprocessor = Pipeline(steps=[ 
            ('pt', PowerTransformer(method='yeo-johnson')),
            ('ss', StandardScaler())                                   
        ]) 

    preprocessor = ColumnTransformer(transformers=[ 
        ('cat', cat_preprocessor, cat_features),
        # ('cat_low', cat_low_cardi_preprocessor, cat_features),
        # ('cat_high', cat_high_cardi_preprocessor, cat_features),
        ('num', num_preprocessor, num_features)                                                       
    ])

    # if conduct hyperparameter tunning with Optuna, take the comment off in the next line.
    # alpha = trial.suggest_loguniform('clf_alpha', 0.001, 10.0) # [0.001, 10] the first 200 rounds lead to best para = 9.961215980791827. [10, 1e4] the first 60 rounds lead to 9983.72346180751. [1e4, 1e8] leads to 40482.85448271827. <<--- the best lambad so far.
    model = Pipeline(steps=[
        ('prep', preprocessor),
        ('clf', RidgeClassifier(class_weight='balanced', alpha=40482.85448271827, fit_intercept=False))
    ])

    X = train.drop('target', axis=1)
    y = train['target']
    skf = StratifiedKFold(n_splits=2, shuffle=True)

    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=3, n_jobs=-1) # remove n_jobs=-1 to avoid "Timeout or by a memory leak."
    return scores.mean()

study = optuna.create_study(direction='maximize')    
study.optimize(ridgeCLF_objective, n_trials=3)
trial = study.best_trial # study.best_trial, study.trials
print(f'roc_auc: {trial.value}')
print(f'best hyperparameters: {trial.params}')
display(study.trials_dataframe())

optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_slice(study)

# screenshot: slice https://i.postimg.cc/gj3mr0ms/2021-03-18-at-14-51-34.png, history https://i.postimg.cc/RhxzrzKH/2021-03-18-at-14-52-09.png

# param_grid = {
#     'clf': [RidgeClassifier(class_weight='balanced', alpha=1000, fit_intercept=False)], # ElasticNet(alpha=1.0, fit_intercept=False) too slow. RidgeClassifier(class_weight='balanced', alpha=1000, fit_intercept=False), 
#     'prep__cat__oh': [OneHotEncoder(handle_unknown='ignore')], # TargetEncoder(handle_unknown='value'); GLMMEncoder()]: TargetEncoder, GLMMEncoder are not good.
#     'prep__cat__ss': [StandardScaler(with_mean=False)],
#     'prep__num__pt': [PowerTransformer(method='yeo-johnson')], # QuantileTransformer() is not good.
#     'clf__alpha': [0.001]
    
    # 'clf__l1_ratio': [0.1, 0.3, 0.5],
    # 'clf__fit_intercept': [True] # Ridge: False is 0.881, True is 0.119; Elastic: True is better.
# }
# gs = GridSearchCV(

#     estimator=model,
#     param_grid=param_grid,
#     scoring='roc_auc',
#     n_jobs=-1,
#     verbose=4,
#     cv=skf
# )
# gs.fit(X, y)
# pd.DataFrame(gs.cv_results_).iloc[:,-3:] 
# gs.best_params_
# gs.cv_results_
# %%


