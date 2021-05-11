#%%
import subprocess, os, datetime, pickle, gc
from google.cloud import storage
import numpy as np
import pandas as pd
import warnings
from scipy.special import erfinv
from matplotlib import pyplot as plt
from tensorflow import keras

warnings.simplefilter('ignore')
plt.style.use('fivethirtyeight')
pd.set_option('max_columns', 100)

COLAB = True

if COLAB:
    COLAB = True
    import sshColab
    os.chdir('/root/.kaggle/')
    json_file = 'gcs-colab.json'
    subprocess.call(f'chmod 600 /root/.kaggle/{json_file}', shell=True)        
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = f'/root/.kaggle/{json_file}' 
    subprocess.call('echo $GOOGLE_APPLICATION_CREDENTIALS', shell=True)    
else:
    subprocess.call('pip install ssh-Colab', shell=True)    
    subprocess.call('pip install google-colab', shell=True)
    import sshColab

project = "strategic-howl-305522"
bucket_name = "gcs-station-168"           
storage_client = storage.Client(project=project)
bucket = storage_client.bucket(bucket_name)

train_df = pd.read_csv('/kaggle/input/tabular-playground-series-apr-2021/train.csv')
test_df = pd.read_csv('/kaggle/input/tabular-playground-series-apr-2021/test.csv')

train_label = train_df['Survived']
train_id = train_df['PassengerId']
test_id = test_df['PassengerId']
del train_df['Survived'], train_df['PassengerId']
del test_df['PassengerId']

train_rows = train_df.shape[0]

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('Time taken : %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

def rank_gauss(x):
    N = x.shape[0]
    temp = x.argsort()
    rank_x = temp.argsort() / N
    rank_x -= rank_x.mean()
    rank_x *= 2
    efi_x = erfinv(rank_x)
    efi_x -= efi_x.mean()
    return efi_x

file_to_load = '10-tps-apr-simple-ensemble-submission.csv'
sshColab.download_to_colab(project, bucket_name, 
    destination_directory = '/kaggle/working', 
    remote_blob_path=f'tps-apr-2021-label/{file_to_load}', 
    local_file_name=file_to_load)
test_pseudo_label = pd.read_csv(f'/kaggle/working/{file_to_load}')     

sshColab.download_to_colab(project, bucket_name, 
    destination_directory = '/kaggle/working', 
    remote_blob_path='tps-apr-2021-label/1clean_data.pkl', 
    local_file_name='1clean_data.pkl') 
data = pickle.load(open('/kaggle/working/1clean_data.pkl', 'rb'))

sshColab.download_to_colab(project, bucket_name, 
    destination_directory = '/kaggle/working', 
    remote_blob_path='tps-apr-2021-label/2missing_code_map.pkl', 
    local_file_name='2missing_code_map.pkl') 
missing_code_map = pickle.load(open('/kaggle/working/2missing_code_map.pkl', 'rb'))

sshColab.download_to_colab(project, bucket_name, 
    destination_directory = '/kaggle/working', 
    remote_blob_path='tps-apr-2021-label/6imputed_df.pkl', 
    local_file_name='6imputed_df.pkl') 
imputed_df = pickle.load(open('/kaggle/working/6imputed_df.pkl', 'rb'))

cols_to_drop = ['Age', 'Fare', 'Embarked_enc'] + \
    [col for col in data.columns.tolist() if col.startswith('Cab')] + \
    [col for col in data.columns.tolist() if col.startswith('Tick')]   

df = data.drop(cols_to_drop, axis=1).join(imputed_df)
df['Embarked_enc_imp'] = df['Embarked_enc_imp'].astype(int)

df = df.rename(lambda x: x.replace('_imp', ''), axis=1)
df = df.rename(lambda x: x.replace('_enc', ''), axis=1)

# Alternatively
# df.columns = list(map(lambda x: x.replace('_imp', ''), df.columns.tolist()))
# df.columns = list(map(lambda x: x.replace('_enc', ''), df.columns.tolist()))

# ANCHOR data preprocessing phase 2 based on the feature importance computed by Xgboost.

# df['Age'] = rank_gauss(df['Age'].values)
# df['Fare'] = rank_gauss(df['Fare'].values)

cols_to_delete = ['misEmbarked', 'misFare', 'misCabin', 'Tick2_first2', 'misCount', 'misTick2', 'Cab2_first2', 'Cab2_first1', 'misTicket', 'misAge', 'Tick2_first1']
# print(len(df.columns.tolist()), len(cols_to_delete), len(set(df.columns.tolist()).difference(set(cols_to_delete))))
# print(df.drop(cols_to_delete, axis=1).columns.tolist())
df = df.drop(cols_to_delete, axis=1)
df.nunique().sort_values(ascending=False) 

# merge pseudo label
TARGET = 'Survived'
df.loc[train_rows:, TARGET] = test_pseudo_label.loc[:, TARGET].values # NOTE IF you encounter "only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices", convert the RHS into np.numpy by "".values".
df.loc[:train_rows-1, TARGET] = train_label.values


#%%
# ANCHOR feature distribution
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def feature_distribution():
    plt.figure(figsize=(16, 32))
    for i, col in enumerate(df.columns.tolist()):
        ax = plt.subplot(10, 2, i + 1)
        ax.set_title(col)
        df[col].hist(bins=50)
feature_distribution()

df = df.assign(LastCnt=df['Last'].map(df['Last'].value_counts()))
df = df.assign(FirstCnt=df['First'].map(df['First'].value_counts()))

cat_cols = ['Tick2_first3', 'Last', 'First', 'Tick1', 'Cab1', 'Embarked', 'Pclass', 'misTick1', 'Sex']
num_cols = ['Fare', 'Age', 'LastCnt', 'FirstCnt', 'FamilySize', 'Parch', 'SibSp']

ss = StandardScaler()
for col in num_cols:
    df[col] = rank_gauss(df[col].values)
    # df[col] = ss.fit_transform(df[col].values.reshape(-1,1)) # NOTE It seems rank_gauss transformation perfroms standard scaling as well.
feature_distribution()

file = '11dataframe_xgboost_based_trim.pkl'
os.chdir('/kaggle/working')
pickle.dump(df, open(f'/kaggle/working/{file}', 'wb'))
sshColab.upload_to_gcs(project, bucket_name, f'tps-apr-2021-label/{file}', f'/kaggle/working/{file}')

file = '11cols_tuple.pkl'
os.chdir('/kaggle/working')
pickle.dump((cat_cols, num_cols), open(f'/kaggle/working/{file}', 'wb'))
sshColab.upload_to_gcs(project, bucket_name, f'tps-apr-2021-label/{file}', f'/kaggle/working/{file}')

#%%

file_to_load = '11dataframe_xgboost_based_trim.pkl'
sshColab.download_to_colab(project, bucket_name, 
    destination_directory = '/kaggle/working', 
    remote_blob_path=f'tps-apr-2021-label/{file_to_load}', 
    local_file_name=file_to_load) 
df = pickle.load(open(f'/kaggle/working/{file_to_load}', 'rb'))

file_to_load = '11cols_tuple.pkl'
sshColab.download_to_colab(project, bucket_name, 
    destination_directory = '/kaggle/working', 
    remote_blob_path=f'tps-apr-2021-label/{file_to_load}', 
    local_file_name=file_to_load) 
cat_cols, num_cols = pickle.load(open(f'/kaggle/working/{file_to_load}', 'rb'))


def illustrate_embedding_dimension():
    xlist = list(range(0, 2001))
    ylist = [int(np.log2(1 + x) * 4) for x in xlist]
    plt.plot(xlist, ylist)
# illustrate_embedding_dimension()

#%%
# CONSTRUCTION
# ANCHOR autoencoder
encoding_dim = 64 # TODO 1
def get_model(encoding_dim, dropout=.2):
    num_dim = len(num_cols)
    num_input = keras.layers.Input((num_dim,), name='num_input')
    cat_inputs = []
    cat_embs = []
    emb_dims = 0

    for col in cat_cols:
        cat_input = keras.layers.Input((1,), name=f'{col}_input')
        emb_dim = max(8, int(np.log2(1 + df[col].nunique()) * 4)) # TODO 2
        cat_emb = keras.layers.Embedding(input_dim=df[col].max() + 1, output_dim=emb_dim)(cat_input) # https://keras.io/api/layers/core_layers/embedding/
        cat_emb = keras.layers.Dropout(dropout)(cat_emb)
        cat_emb = keras.layers.Reshape((emb_dim,))(cat_emb)

        cat_inputs.append(cat_input)
        cat_embs.append(cat_emb)
        emb_dims += emb_dim

    merged_inputs = keras.layers.Concatenate()([num_input] + cat_embs)

    encoded = keras.layers.Dense(encoding_dim * 3, activation='relu')(merged_inputs)
    encoded = keras.layers.Dropout(dropout)(encoded)
    encoded = keras.layers.Dense(encoding_dim * 2, activation='relu')(encoded)
    encoded = keras.layers.Dropout(dropout)(encoded)    
    encoded = keras.layers.Dense(encoding_dim, activation='relu')(encoded)
    
    decoded = keras.layers.Dense(encoding_dim * 2, activation='relu')(encoded)
    decoded = keras.layers.Dropout(dropout)(decoded)
    decoded = keras.layers.Dense(encoding_dim * 3, activation='relu')(decoded)
    decoded = keras.layers.Dropout(dropout)(decoded)    
    decoded = keras.layers.Dense(num_dim + emb_dims, activation='linear')(decoded)

    encoder = keras.Model([num_input] + cat_inputs, encoded)
    ae = keras.Model([num_input] + cat_inputs, decoded)
    ae.add_loss(keras.losses.mean_squared_error(merged_inputs, decoded))
    ae.compile(optimizer='adam')
    return ae, encoder

# %%
