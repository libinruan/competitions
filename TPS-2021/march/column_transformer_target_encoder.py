
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

seed_everything(seed=2020)
def seed_everything(seed=1903):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

# --------------------------------- RAW DATA --------------------------------- #

os.chdir('/kaggle/working')
train = pd.read_csv('../input/tabular-playground-series-mar-2021/train.csv')
test = pd.read_csv('../input/tabular-playground-series-mar-2021/test.csv')
sample_submission = pd.read_csv('../input/tabular-playground-series-mar-2021/sample_submission.csv')

train_id = train.loc[:,'id']
test_id = test.loc[:, 'id']
train.drop(['id'], axis=1, inplace=True)
test.drop(['id'], axis=1, inplace=True)

print(train.shape) # (300000, 31)
print(test.shape)  # (200000, 30)
print(sample_submission.shape) # (200000, 2)

sparse_features = [col for col in train.columns if col.startswith('cat')] # 19
dense_features = [col for col in train.columns if col not in sparse_features+['target']] # 11
data = pd.concat([train, test], axis=0)

def table_unique_level_count_sparse_feature():
    tmp = [] # alternatively, dict approach
    for c in sparse_features:
        tmp.append(pd.DataFrame({c: [data[c].nunique()]}, index=['count']))
    display(pd.concat(tmp, axis=1))
# table_unique_level_count_sparse_feature()

def cat_features_size_less_than(number):
    tmp = dict()
    for c in sparse_features:
        tmp[c] = data[c].nunique()
    return pd.Series(tmp).sort_values()[pd.Series(tmp).sort_values()<number].index.tolist()
cat_features_less_num = cat_features_size_less_than(10) # [10, 16, 25, 62, 100]
cat_features_to_target_encode = [col for col in sparse_features if col not in cat_features_less_num]

# # sanity check
# len(cat_features_less_num) + len(cat_features_to_target_encode) == len(sparse_features)
# set(cat_features_less_num).intersection(set(cat_features_to_target_encode)) # empty
# set(train.id).intersection(set(test.id)) # empty

# ------------------------------- stackexchange ------------------------------ #
# NOTE How to do target encoding with test dataset?
# Source: https://datascience.stackexchange.com/questions/81260/targetl-encoding-with-kfold-cross-validation-how-to-transform-test-set

import random
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from category_encoders.target_encoder import TargetEncoder
from category_encoders.m_estimate import MEstimateEncoder
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge, RidgeClassifier
from sklearn.metrics import make_scorer
from sklearn.metrics import balanced_accuracy_score

# y = random.choices([1, 0], weights=[0.2, 0.8], k=100)
# cat = random.choices(["A", "B", "C"], k=100)
# df = pd.DataFrame.from_dict({"y": y, "cat": cat})
# X_train, X_test, y_train, y_test = train_test_split(df[["cat"]], df["y"], train_size=0.8, random_state=42)


random.seed(1234)
X = train.drop(['target'], axis=1)
y = train['target'].values.reshape(-1,1).astype('int')

skf = StratifiedKFold(n_splits=5)

# data preprocessing
num_pipeline = Pipeline([
    ('std_scaler', StandardScaler()) # 
])
# te = TargetEncoder()
ridgeClf = RidgeClassifier()
full_pipeline = ColumnTransformer([
    ('num', num_pipeline, dense_features),
    ('cat', OneHotEncoder(), sparse_features)
])
param_grid = {
#     "te__smoothing": [0.0001]
    'alpha': [1, 0.1],
    'fit_intercept': [False]
}
scoring = {'AUC' : 'roc_auc', 'accuracy': make_scorer(balanced_accuracy_score),}
clf = GridSearchCV(
    estimator=ridgeClf, # old mistake in classification by using ridge instead of ridge classifier, which in multi-metric scoring case returns nothing (although no error arises).
    param_grid=param_grid,
    scoring=scoring, # defining model evaluation rules # 'balanced_accuracy'
    refit='AUC',
    n_jobs=-1,
    return_train_score=True,
    cv=skf
)
clf.fit(full_pipeline.fit_transform(X), y)

clf.best_params_
clf.best_estimator_
clf.cv_results_ 
pd.Series(clf.cv_results_.keys())
pd.DataFrame(clf.cv_results_)
clf.cv_results_['params'][clf.best_index_] # retrieve the best parameter setting. TRICK

import joblib
joblib.dump(clf, 'clf_obj.pkl')
# clf2 = joblib.load('clf_obj.pkl')

#%%
# ANCHOR Column transformer 101.
# Basic column transformer comprising categorical and numerical transformers
# ------------------------------- Experiment 1 ------------------------------- #

import pandas as pd
data = pd.DataFrame([
    ('a',1,1), 
    ('a',2,0),
    ('b',3,0),
    ('c',4,1)],
    index = ['a1','a2','a3','a4'], columns = ['cat','num','tar']
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

num_tr1 = ColumnTransformer(transformers=[
    ('ss', MinMaxScaler(), ['num']),
], remainder='passthrough')
num_tr1.fit_transform(data)

num_tr2 = ColumnTransformer(transformers=[
    ('ss', MinMaxScaler(), ['num']),
], remainder='drop') # default
num_tr2.fit_transform(data)

cat_tr1 = ColumnTransformer(transformers=[
    ('oh', OneHotEncoder(), ['cat'])
], remainder='drop') # default
cat_tr1.fit_transform(data)

cat_tr2 = ColumnTransformer(transformers=[
    ('oh', OneHotEncoder(), ['cat'])
], remainder='passthrough') 
cat_tr2.fit_transform(data)

# 誰先誰後結果都一樣
# 想像the initial resulting object is an emtpy data frame.
# 想像the full data set 會一個都不缺地傳給每一個transformer
# 想像所得到的結果都會依序填入那個dataframe。
poo_tr1 = ColumnTransformer(transformers=[
    ('oh', OneHotEncoder(), ['cat']),
    ('ss', MinMaxScaler(), ['num']),
])
poo_tr1.fit_transform(data)

# 有沒有remainder=passthrough，決定"從來沒有"被用到的特徵去留
poo_tr1 = ColumnTransformer(transformers=[
    ('oh', OneHotEncoder(), ['cat']),
    ('ss', MinMaxScaler(), ['num']),
], remainder='passthrough')
poo_tr1.fit_transform(data)

#%%

# ------------------------------- Experiment 2 ------------------------------- #

# %%
import random
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector as selector
seed_everything(seed=2020)
K = 6
tar = random.choices([1, 0], weights=[0.2, 0.8], k=K)
cat = random.choices(["A", "B", "C"], k=K)
dog = random.choices(["D", "E", "F"], k=K)
n1 = [random.randint(1,9) for _ in range(K)]
n2 = [random.randint(1,9) for _ in range(K)]
# df = pd.DataFrame.from_dict({"tar": tar, "cat": cat, 'dog': dog, 'n1': num1, 'n2': n2})
df = pd.DataFrame.from_dict({"tar": tar, "cat": cat, 'dog': dog, 'n1': n1})
display(df)

# NOTE LabelBinarizer is not supposed to be used with features.
tr1 = ColumnTransformer([
    ('cat2n2', OrdinalEncoder(), ['cat']) # Don't use LabelEncoder here. Otherwise, https://stackoverflow.com/questions/46162855/fit-transform-takes-2-positional-arguments-but-3-were-given-with-labelbinarize
], remainder='passthrough')
tr1.fit_transform(df)

# ANCHOR Column selector
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self._key = key
    def fit(self, df, y=None):
        return self
    def transform(self, df):
        return df[self._key]

tr2 = Pipeline([
    ('selector2', DataFrameSelector(['n1']))
])
tr2.fit_transform(df)

# %% numerical features方法一
class CategoricalFeaturesSelector(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.loc[:, X.select_dtypes(include=['int64', 'float64']).columns.tolist()]
tr3 = Pipeline([
    ('selector3', CategoricalFeaturesSelector())
])
tr3.fit_transform(df)

# %% 選出categorical features方法二
from sklearn.compose import make_column_selector as selector
class dummyTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return pd.DataFrame(X, columns=selector(dtype_include='object')(X)) # return pd.DataFrame
tr4 = ColumnTransformer(transformers=[
    ('dum', dummyTransformer(), selector(dtype_include='object'))
])
tr4.fit_transform(df) # Note: returns a ndarray

# %%
# ANCHOR Custom transformers
# ------------------------------- EXPERIMENT 3 ------------------------------- #
import random
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector as selector

def seed_everything(seed=1903):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(seed=2020)
# toy example: generate the dataframe `df`.
K = 6
label = random.choices([1, 0], weights=[0.2, 0.8], k=K)
cat = random.choices(["A", "B", "C"], k=K)
bird = random.choices(["D", "E", "F"], k=K)
horse = random.choices(["G", "H", "I"], k=K)
n1 = [random.randint(1,9) for _ in range(K)]
n2 = [random.randint(1,9) for _ in range(K)]
df = pd.DataFrame.from_dict({"cat": cat, 'bird': bird, 'horse': horse, 'n1': n1, 'n2': n2, "label": label})

# Appraoch 1: use an empty Dataframe to store data. This approach is difficult to keep the columns ordered in the original order.
# a custom made class can dynamically screen out a certain categorical features for encoding.
class MySparseFeatureTransformer(): # change to cat to con (not one hot)
    def __init__(self, threshold):
        self.threshold = threshold

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        """Transform features of length less than self.threshold
        with ordinal encoder
        """
        temp = pd.DataFrame(index=range(X.shape[0])) # initialize a Dataframe
        enc = OrdinalEncoder()
        cats = selector(dtype_include='object')(X)
        cats_to_encode = list(filter(lambda x: len(x) < self.threshold, cats))
        nums_to_keep = set(X.columns).difference(set(cats))
        m = 0
        for i in set(cats_to_encode):
            temp[i] = enc.fit_transform(X.loc[:,i].to_numpy().reshape(-1,1)).astype('int')

        return pd.concat([temp, X.loc[:, [z for z in cats if z not in cats_to_encode]], X.loc[:, nums_to_keep]], axis=1)

# Usage 1. generate a dataframe with Pipeline
pl = MySparseFeatureTransformer(4)
pipe1 = Pipeline(steps=[
    ('pl', pl)
])        
dg = pipe1.fit_transform(df)

# Pipeline returns pandas dataframe.

# Usage 2. generate a numpy array using ColumnTransformer
pl2 = MySparseFeatureTransformer(4)
pipe2 = Pipeline(steps=[
    ('pl', pl2)
])   
coltr = ColumnTransformer(transformers=[
    ('pipe', pipe2, selector(dtype_include='object'))
], remainder='passthrough')
coltr.fit_transform(df) # 'human' is screened out
# ColumnTransformer returns numpy array.
# screenshot: https://i.postimg.cc/xdBGTsP7/2021-03-14-at-19-54-06.png

# Approach 2: Using slicing assignment to avoid "SettingWithCopy" warning. Seems much better to control the order of features.
class HerSparseFeatureTransformer(): # change to cat to con (not one hot)
    def __init__(self, threshold):
        self.threshold = threshold

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        """Transform features of length less than self.threshold
        with ordinal encoder
        """
        dX = X.copy() # use deep copy!! 
        enc = OrdinalEncoder()
        cats = selector(dtype_include='object')(X)
        cats_to_encode = list(filter(lambda x: len(x) < self.threshold, cats))
        nums_to_keep = set(X.columns).difference(set(cats))
        for i in dX.columns:
            if i in cats_to_encode:
                dX.loc[:,i] = enc.fit_transform(dX.loc[:,i].to_numpy().reshape(-1,1)).astype('int')
        return dX        

# Usage 1. generate a dataframe with Pipeline
ht = HerSparseFeatureTransformer(4)
pipe2 = Pipeline(steps=[
    ('pl', ht)
])        
dh = pipe2.fit_transform(df)
display(df)
display(dh) # screenshot: https://i.postimg.cc/8k9GkTRz/2021-03-14-at-19-41-15.png

# Usage 2. generate a numpy array using ColumnTransformer
ht2 = HerSparseFeatureTransformer(5)   
pipe2 = Pipeline(steps=[
    ('pl', ht2)
])
coltr2 = ColumnTransformer(transformers=[
    ('pipe', pipe2, selector(dtype_include='object'))
], remainder='passthrough')
display(df)
coltr2.fit_transform(df) # screenshot: https://i.postimg.cc/43WCDG4Z/2021-03-14-at-19-41-35.png
# ColumnTransformer returns numpy array.

# Another takeaway is "Column Transformer returns numpy arrays; Pipeline returns pandas dataframes."

#%%
# ------------------------------- Experiment 4 ------------------------------- #
# Target Encoding on 'horse', One Hot Encoding on 'bird' and 'cat'.
# Numerical numbers are standardized.

# it works.

import random
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector as selector
from category_encoders.target_encoder import TargetEncoder

def seed_everything(seed=1903):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(seed=2020)

K = 6
label = random.choices([1, 0], weights=[0.5, 0.5], k=K)
cat = random.choices(["A", "B"], k=K)
bird = random.choices(["C", "D"], k=K)
horse = random.choices(["E", "F", "G"], k=K)
n1 = [random.randint(1,9) for _ in range(K)]
n2 = [random.randint(1,9) for _ in range(K)]
df = pd.DataFrame.from_dict({"cat": cat, 'bird': bird, 'horse': horse, 'n1': n1, 'n2': n2, "label": label})
display(df)

# NOTE `filter()` on pd.DataFrame
THRESHOLD = 4
a = [x for x in filter(lambda x: len(x) > THRESHOLD, df.columns) if x not in ['label']] # fe
b = [x for x in df.select_dtypes(exclude='number') if x not in a+['label']] # ohe, don't drop the first.
c = [x for x in df.select_dtypes(include='number') if x not in ['label']] # dense

te = TargetEncoder()
ct = ColumnTransformer(transformers=[
    ('te', TargetEncoder(), a), # smoothing, additive smoothing should be better. 
    # ('oe', OrdinalEncoder(), b)
    ('ohe', OneHotEncoder(), b), # sparse=False
    ('scale', StandardScaler(), c)
    # experiments on scale one-hot or not
], remainder='passthrough')
pipe = Pipeline(steps=[
    ('ct', ct)
])
output = pipe.fit_transform(df.drop(['label'], axis=1), y=df['label'])
pd.DataFrame(output)


# ------------------------------- Experiment 5 ------------------------------- #
# %%
# This experiment shows when the sample size is small like this example (size=6), the n_fold value is determinant to the output of target encoding. Check the result of this experiment for reference.
# ANCHOR Experimental target encoder
# STACKOVERFLOW LINK https://medium.com/@pouryaayria/k-fold-target-encoding-dfe9a594874b
import random
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector as selector
from category_encoders.target_encoder import TargetEncoder

def seed_everything(seed=1903):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(seed=2020)

K = 6
label = random.choices([1, 0], weights=[0.5, 0.5], k=K)
cat = random.choices(["A", "B"], k=K)
bird = random.choices(["C", "D"], k=K)
horse = random.choices(["E", "F", "G"], k=K)
n1 = [random.randint(1,9) for _ in range(K)]
n2 = [random.randint(1,9) for _ in range(K)]
df = pd.DataFrame.from_dict({"cat": cat, 'bird': bird, 'horse': horse, 'n1': n1, 'n2': n2, "label": label})
display(df)

class KFoldTargetEncoderTrain(BaseEstimator, TransformerMixin):
    def __init__(self,colnames,
            targetName,
            n_fold=5, 
            verbosity=True,
            discardOriginal_col=False):
        self.colnames = colnames
        self.targetName = targetName
        self.n_fold = n_fold
        self.verbosity = verbosity
        self.discardOriginal_col = discardOriginal_col
    def fit(self, X, y=None):
        return self
    def transform(self,X):
        assert(type(self.targetName) == str)
        assert(type(self.colnames) == str)
        assert(self.colnames in X.columns)
        assert(self.targetName in X.columns)

        mean_of_target = X[self.targetName].mean()
        kf = KFold(n_splits = self.n_fold, #stratifiedKFold 有可能遇到某個classes中所對應的個數少於切割數的問題，尤其在樣本數太小的情況下。
                   shuffle = False, random_state=2019)
        col_mean_name = self.colnames + '_' + 'Kfold_Target_Enc'
        X[col_mean_name] = np.nan # new column
        for tr_ind, val_ind in kf.split(X): # to change
            X_tr, X_val = X.iloc[tr_ind], X.iloc[val_ind]
            X.loc[X.index[val_ind], col_mean_name] = X_val[self.colnames].map(X_tr.groupby(self.colnames)
                                     [self.targetName].mean())
            X[col_mean_name].fillna(mean_of_target, inplace = True) # Nice!!
        if self.verbosity:
            encoded_feature = X[col_mean_name].values
            print('Correlation between the new feature, {} and, {} is {}.'.format(
                col_mean_name,
                self.targetName,                    
                np.corrcoef(X[self.targetName].values, encoded_feature)[0][1]))
        if self.discardOriginal_col:
            X = X.drop(self.targetName, axis=1)
        return X

display(df)
temp = []
nfolds = [2, 3, 4]
for n in nfolds:
    te = KFoldTargetEncoderTrain(colnames='horse', targetName='label', n_fold=n,verbosity=False)
    temp.append(te.fit_transform(df).copy().iloc[:,-1])
temp # 顯示結果差異很大隨著n_folds

# The code beow seems have been not compatible with the code above (because I have revied the one above).
class KFoldTargetEncoderTest(base.BaseEstimator, base.TransformerMixin):
    def __init__(self,train,colNames,encodedName): 
        self.train = train
        self.colNames = colNames
        self.encodedName = encodedName
        
    def fit(self, X, y=None):
        return self
    def transform(self,X):
        # read the encoded training data first into a dictionary
        mean =  self.train[[self.colNames, self.encodedName]].groupby(self.colNames).mean().reset_index() 
        dd = {}
        for index, row in mean.iterrows():
            dd[row[self.colNames]] = row[self.encodedName]
        X[self.encodedName] = X[self.colNames]
        X = X.replace({self.encodedName: dd}) # <-------------------------------
        return X

#%%

# ANCHOR Merge low-frequency values into a single one
# ------------------------------- Experiment 6 ------------------------------- #
# 用處；出現次數太少的categorical values歸為一起。
# 目前推斷應該用於categorical encoding之前。
# SOURCE: https://tinyurl.com/yz9uyc4a

from collections import defaultdict
class CategoryGrouper(BaseEstimator, TransformerMixin):  
    """A tranformer for combining low count observations for categorical features.

    This transformer will preserve category "values" that are above a certain
    threshold, while bucketing together all the other values. This will fix issues where new data may have an unobserved category value that the training data did not have.
    """

    def __init__(self, threshold=0.05):
        """Initialize method.

        Args:
            threshold (float): The threshold to apply the bucketing when
                categorical values drop below that threshold.
        """
        self.d = defaultdict(list) # columns to be preserved (leraned in the fit phase)
        self.threshold = threshold

    def transform(self, X, **transform_params):
        """Transforms X with new buckets.

        Args:
            X (obj): The dataset to pass to the transformer.

        Returns:
            The transformed X with grouped buckets.
        """
        X_copy = X.copy()
        for col in X_copy.columns:
            X_copy[col] = X_copy[col].apply(lambda x: x if x in self.d[col] else 'CategoryGrouperOther') # IMPORTANT!! the else leads to 'CategoryGrouperOther'
        return X_copy

    def fit(self, X, y=None, **fit_params):
        """Fits transformer over X.

        Builds a dictionary of lists where the lists are category values of the
        column key for preserving, since they meet the threshold.
        """
        df_rows = len(X.index)
        for col in X.columns:
            calc_col = X.groupby(col)[col].agg(lambda x: (len(x) * 1.0) / df_rows)
            self.d[col] = calc_col[calc_col >= self.threshold].index.tolist() # select the level to be kept.
        return self

#%% 
# ANCHOR Gaussian target encoder sample (source unknown. The author also wrote articles about Beta target encoder)
# ------------------------------- Experiment 6 ------------------------------- #
class GaussianTargetEncoder():
        
    def __init__(self, group_cols, target_col="target", prior_cols=None):
        self.group_cols = group_cols
        self.target_col = target_col
        self.prior_cols = prior_cols

    def _get_prior(self, df):
        if self.prior_cols is None:
            prior = np.full(len(df), df[self.target_col].mean())
        else:
            prior = df[self.prior_cols].mean(1)
        return prior
                    
    def fit(self, df):
        self.stats = df.assign(mu_prior=self._get_prior(df), y=df[self.target_col])
        self.stats = self.stats.groupby(self.group_cols).agg(
            n        = ("y", "count"),
            mu_mle   = ("y", np.mean),
            sig2_mle = ("y", np.var),
            mu_prior = ("mu_prior", np.mean),
        )        
    
    def transform(self, df, prior_precision=1000, stat_type="mean"):        
        precision = prior_precision + self.stats.n/self.stats.sig2_mle        
        if stat_type == "mean":
            numer = prior_precision*self.stats.mu_prior\
                    + self.stats.n/self.stats.sig2_mle*self.stats.mu_mle
            denom = precision
        elif stat_type == "var":
            numer = 1.0
            denom = precision
        elif stat_type == "precision":
            numer = precision
            denom = 1.0
        else: 
            raise ValueError(f"stat_type={stat_type} not recognized.")
        
        mapper = dict(zip(self.stats.index, numer / denom))
        if isinstance(self.group_cols, str):
            keys = df[self.group_cols].values.tolist()
        elif len(self.group_cols) == 1:
            keys = df[self.group_cols[0]].values.tolist()
        else:
            keys = zip(*[df[x] for x in self.group_cols])
        
        values = np.array([mapper.get(k) for k in keys]).astype(float)
        
        prior = self._get_prior(df)
        values[~np.isfinite(values)] = prior[~np.isfinite(values)]
        
        return values
    
    def fit_transform(self, df, *args, **kwargs):
        self.fit(df)
        return self.transform(df, *args, **kwargs)
