# LINK # KERNEL SOURCE: https://www.kaggle.com/siavrez/kerasembeddings
#%%
from tensorflow.keras import layers, optimizers, callbacks, utils, losses, metrics, backend as K
from sklearn import metrics as skmetrics, preprocessing
from tensorflow.keras.models import Model, load_model
from sklearn.model_selection import StratifiedKFold
from scipy.stats import rankdata
import os, gc, joblib, warnings, time
import tensorflow_addons as tfa
import tensorflow as tf
import pandas as pd
import numpy as np
from datetime import datetime

import os, warnings
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn import metrics as skmetrics, preprocessing
from sklearn.decomposition import PCA, FactorAnalysis    
from umap import UMAP

warnings.filterwarnings('ignore')

def seed_everything(seed=1903):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

os.chdir('/kaggle/working')
train = pd.read_csv('../input/tabular-playground-series-mar-2021/train.csv')
test = pd.read_csv('../input/tabular-playground-series-mar-2021/test.csv')
sample_submission = pd.read_csv('../input/tabular-playground-series-mar-2021/sample_submission.csv')

test_id = test.id.values
train = train.drop(['id'], axis=1)
test = test.drop(['id'], axis=1)  

print(train.shape) # (300000, 31)
print(test.shape)  # (200000, 30)
print(sample_submission.shape) # (200000, 2)

sparse_features = [col for col in train.columns if col.startswith('cat')] # 19
dense_features = [col for col in train.columns if col not in sparse_features+['target']] # 11
MAX_EMBED_DIM = 20

# ANCHOR - Embedding layer. 
def create_model(data, catcols):    
    inputs = []
    outputs = []
    for c in catcols:
        num_unique_values = int(data[c].nunique()) # number of unique (NA is set to be dropped by default).
        embed_dim = int(min(np.ceil((num_unique_values)/2), MAX_EMBED_DIM)) # the maximum is 20.
        inp = layers.Input(shape=(1,)) # scalar per data point, inp's shape is (None, 1)
        out = layers.Embedding(num_unique_values + 1, embed_dim, name=c)(inp) # Turns positive integers (indexes) into dense vectors of fixed size; https://keras.io/api/layers/core_layers/embedding/; This layer can only be used as the first layer in a model. 
        out = layers.SpatialDropout1D(0.25)(out) #  A. promote independence between feature maps; B. https://keras.io/api/layers/regularization_layers/spatial_dropout1d/; C. graphical answer: https://blog.csdn.net/weixin_43896398/article/details/84762943#commentBox; D. why we use spatial1D dropout here: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52058.
        out = layers.Reshape(target_shape=(embed_dim, ))(out) # out's shape is (None, embed_dim).
        inputs.append(inp)
        outputs.append(out)
    x = layers.Concatenate()(outputs) # keras layers.concatenate by defatul along axis = -1.
    x = layers.BatchNormalization()(x) # where do we place batchnormalization layers? Answer: https://stackoverflow.com/questions/34716454/where-do-i-call-the-batchnormalization-function-in-keras. 
    x = layers.Dense(300, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(300, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)
    y = layers.Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=y)
    return model

# ANCHOR - Missing categorical values.
# Replacing levels not shared by both of train and test sets with NaN.
for col in sparse_features:
    train_only = list(set(train[col].unique()) - set(test[col].unique()))
    test_only = list(set(test[col].unique()) - set(train[col].unique()))
    both = list(set(test[col].unique()).union(set(train[col].unique())))
    train.loc[train[col].isin(train_only), col] = np.nan # Silence those levels only shown in train set
    test.loc[test[col].isin(test_only), col] = np.nan # Silence those levels only shown in train set
    mode = train[col].mode().values[0]
    train[col] = train[col].fillna(mode)
    test[col] = test[col].fillna(mode)
    
# CLIP TEST
for feat in dense_features:
    test[feat] = np.clip(test[feat], train[feat].min(), train[feat].max())

test['target'] = -1
data = pd.concat([train, test]).reset_index(drop=True)

# NOTE - After adding binned features derivated from contiuous features, Use
# ordinal encoder to encode all the sparse features (invluding newly added
# ones). Do Label Encoder and Ordinal Encoder have the same functionality? Yes.
# But the idea is a little different behind. 
# See LINK https://datascience.stackexchange.com/questions/39317/difference-between-ordinalencoder-and-labelencoder/64177#:~:text=Ordinal%20encoding%20should%20be%20used,matter%2C%20like%20blonde%20%2C%20brunette%20).

for c in dense_features:
    data[f'q_{c}'], bins_ = pd.qcut(data[c], 25, retbins=True, labels=[i for i in range(25)])
    data[f'q_{c}'] = data[f'q_{c}'].astype('str')
    sparse_features.append(f'q_{c}')

features = sparse_features
for feat in features:
    lbl_enc = preprocessing.OrdinalEncoder()
    data[feat] = lbl_enc.fit_transform(data[feat].fillna('-1').values.reshape(-1,1).astype(str))

train = data[data.target != -1].reset_index(drop=True)
test = data[data.target == -1].reset_index(drop=True)
# Since target column is stuck between original sparse featurs (prefixed with 'cat') and new sparse features (prefixed with 'q_cont'), we use the next line to screen out the target column.
test_data = [test.loc[:, features].values[:, k] for k in range(test.loc[:, features].values.shape[1])] # probably dropping the target column would be easier.

oof_preds = np.zeros((len(train)))
bagged_oof_preds = np.zeros((len(train)))
test_preds = np.zeros((len(test)))
bagged_test_preds = np.zeros((len(test)))
learning_rate = 1e-3 # LINK https://medium.com/軟體之心/deep-learning-為什麼adam常常打不過sgd-癥結點與改善方案-fd514176f805
label_smoothing = 0.0 
Verbose = 0

n_bags = 2 # manual grid search; used with variables n_splits and seeds.
n_splits = [10, 15] # Control the number of splits in cross validation.
seeds = [2021, 2021] 

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

# ANCHOR - Label smoothing.
# Question: Why and how do we use label smoothing in CV classification with mislabeled data? Answer: https://i.postimg.cc/3NZQWfLv/2021-03-10-at-16-40-28.png.
# Question: What's softmax loss and cross entropy loss? The former is not a correct term. See this quora: https://www.quora.com/Is-the-softmax-loss-the-same-as-the-cross-entropy-loss.

# NOTE - tfa.optimizers.SWA: stochastic weight averaging https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers/SWA.
print(f'start at {datetime.now()}')  
for bag in range(n_bags):
    print(f'Iteration {bag+1} splits {n_splits[bag]} seed {seeds[bag]}')
    for fold, (train_index, test_index) in enumerate(StratifiedKFold(n_splits=n_splits[bag], shuffle=True, random_state=seeds[bag]).split(train, train.target.values)):
        X_train, X_test = train.iloc[train_index, :], train.iloc[test_index, :]
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train, y_test = X_train.target.values, X_test.target.values
        model = create_model(data, features)
        model.compile(
            optimizer=tfa.optimizers.SWA(tf.keras.optimizers.Adam(learning_rate=learning_rate)),
            loss=losses.BinaryCrossentropy(label_smoothing=label_smoothing),
            metrics=metrics.AUC(name="AUC")
        )
        X_train = [X_train.loc[:, features].values[:, k] for k in range(X_train.loc[:, features].values.shape[1])]
        X_test = [X_test.loc[:, features].values[:, k] for k in range(X_test.loc[:, features].values.shape[1])]
        # ANCHOR EarlyStopping, Checkpoint and learning rate shcheduling
        es = callbacks.EarlyStopping(monitor='val_AUC', min_delta=0.000001, patience=10, verbose=Verbose, mode='max', baseline=None, restore_best_weights=True)
        sb = callbacks.ModelCheckpoint('./nn_model.w8', save_weights_only=True, save_best_only=True, verbose=Verbose, monitor='val_AUC',mode='max')
        plateau  = callbacks.ReduceLROnPlateau(monitor='val_AUC', factor=0.5, patience=2, verbose=Verbose, mode='max', min_delta=0.0001, cooldown=0, min_lr=1e-7)
        model.fit(X_train,
                  y_train,
                  validation_data=(X_test, y_test),
                  verbose=Verbose,
                  batch_size=1024,
                  callbacks=[es, sb, plateau],
                  epochs=100
        )
        valid_fold_preds = model.predict(X_test)
        test_fold_preds = model.predict(test_data)
        oof_preds[test_index] = rankdata(valid_fold_preds.ravel())/len(X_test)
        test_preds += rankdata(test_fold_preds.ravel() / n_splits[bag])/len(test)
        print(f'fold {fold+1} AUC : {skmetrics.roc_auc_score(y_test, valid_fold_preds)}')
        K.clear_session()
    print(f'Overall AUC of Iteration {bag+1} = {skmetrics.roc_auc_score(train.target.values, oof_preds)}')
    np.save(f'oof_preds_{bag}',oof_preds)
    np.save(f'test_preds_{bag}',test_preds)
    bagged_test_preds += test_preds / n_bags
    bagged_oof_preds += oof_preds / n_bags
print("Overall AUC={}".format(skmetrics.roc_auc_score(train.target.values, bagged_oof_preds)))
print('Saving submission file')
submission = pd.DataFrame.from_dict({
    'id': test_id,
    'target': bagged_test_preds,
})

submission.to_csv('submission.csv', index=False)
submission.head(3)

# ANCHOR rankdata()
top_public = pd.read_csv('/kaggle/input/tps-mar-2021-stacked-starter/submission.csv')
submission['target'] = (rankdata(submission.target) * 0.275 + rankdata(top_public.target) * 0.725)/len(submission)
submission.to_csv('blend.csv', index=False)
submission.head(3)

# Screenshot : https://i.postimg.cc/jq12c5K9/2021-03-11-at-00-07-33.png
# runtime: 1 hr 34 min.









# # get rid of continuous features
# N_BIN = 25
# for c in dense_features:
#     data[f'q_{c}'], bins_ = pd.qcut(data[c], N_BIN, retbins=True, labels=np.arange(N_BIN).tolist()) # [i for i in range(N_BIN)]
#     data[f'q_{c}'] = data[f'q_{c}'].astype('str')
#     sparse_features.append(f'q_{c}')
# features = sparse_features
# for feat in features:
#     lbl_enc = preprocessing.OrdinalEncoder()
#     data[feat] = lbl_enc.fit_transform(data[feat].fillna('-1').values.reshape(-1,1).astype(str))
        

# train2 = data[data.target != -1].reset_index(drop=True)
# test2 = data[data.target == -1].reset_index(drop=True).drop(['target'], axis=1)    



# data = pd.concat([train.loc[:,dense_features], test.loc[:,dense_features]])

# # IS_TRAIN = True
# MODEL_DIR = '/kaggle/working'
# n_comp = 90
# n_dim = 45

# # FA, UMAP
# fa = FactorAnalysis(n_components=n_comp, random_state=1903).fit(data)
# pd.to_pickle(fa, f'{MODEL_DIR}/factor_analysis.pkl')
# umap = UMAP(n_components=n_dim, random_state=1903).fit(data)
# pd.to_pickle(umap, f'{MODEL_DIR}/umap.pkl')
# data2 = fa.transform(data)
# data3 = umap.transform(data) 


# #%%




# # %%
