# %%
"""
pytorch: https://towardsdatascience.com/pytorch-tabular-multiclass-classification-9f8211a123ab (major)
kfold: https://www.kaggle.com/carlmcbrideellis/tabnet-a-very-simple-regression-example
tabnet: https://www.kaggle.com/marcusgawronsky/tabnet-in-tensorflow-2-0

NOT WORKING!
"""
import seaborn as sns
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.preprocessing import MinMaxScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


import pandas as pd
import numpy as np
import random
import os
import pickle
import datetime
import pytz

# from sklearnex import patch_sklearn
# patch_sklearn()

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

sns.countplot(x = 'target', data=train, order = train['target'].value_counts().index[::-1])
plt.show()

X_trn, X_tst = all_df.iloc[:len(train)], all_df.iloc[len(train):]
y_trn = train['target']
le = LabelEncoder()
y_trn = le.fit_transform(y_trn)
pd.Series(y_trn).value_counts().plot(kind="bar")
plt.show()

# %%
X = X_trn[features].values
y = y_trn.copy() # !!
X_test = X_tst[features].values

X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=69)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.1, stratify=y_trainval, random_state=21)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

X_train, y_train = np.array(X_train), np.array(y_train)
X_val, y_val = np.array(X_val), np.array(y_val)
X_test, y_test = np.array(X_test), np.array(y_test)

# ANCHOR Custom Dataset
class ClassifierDataset(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

# Make sure X is a float while y is long.
train_dataset = ClassifierDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
val_dataset = ClassifierDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
test_dataset = ClassifierDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())    

# NOTE Because there’s a class imbalance, we use stratified split to create our
# train, validation, and test sets. While it helps, it still does not ensure
# that each mini-batch of our model see’s all our classes. We need to
# over-sample the classes with less number of values. To do that, we use the
# WeightedRandomSampler.

target_list = []
for _, t in train_dataset:
    target_list.append(t)
    
target_list = torch.tensor(target_list)
target_list = target_list[torch.randperm(len(target_list))]

class_count = [i for i in pd.Series(y_train).value_counts().values]
class_weights = 1./torch.tensor(class_count, dtype=torch.float) 
print(class_weights)

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
class_weights_all = class_weights[target_list]
weighted_sampler = WeightedRandomSampler(
    weights=class_weights_all,
    num_samples=len(class_weights_all),
    replacement=True
)

layer1node = 1024 # 8 # 16 # 64 # 512
layer2node = 128 # 32 # 128
layer3node = 32 # 4 # 64
dropoutrate = 0.5 # 0.2
EPOCHS = 300
BATCH_SIZE = 64
LEARNING_RATE = 0.005
NUM_FEATURES = len(features)
NUM_CLASSES = 4

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          sampler=weighted_sampler
)
val_loader = DataLoader(dataset=val_dataset, batch_size=1)
test_loader = DataLoader(dataset=test_dataset, batch_size=1)
class MulticlassClassification(nn.Module):
    def __init__(self, num_feature, num_class):
        super(MulticlassClassification, self).__init__()

        self.layer_1 = nn.Linear(num_feature, layer1node)
        self.layer_2 = nn.Linear(layer1node, layer2node)
        self.layer_3 = nn.Linear(layer2node, layer3node)
        self.layer_out = nn.Linear(layer3node, num_class) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropoutrate)
        self.batchnorm1 = nn.BatchNorm1d(layer1node)
        self.batchnorm2 = nn.BatchNorm1d(layer2node)
        self.batchnorm3 = nn.BatchNorm1d(layer3node)
        
    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        
        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_out(x)
        
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)      

# NOTE Initialize the model, optimizer, and loss function. Transfer the model to
# GPU. We’re using the nn.CrossEntropyLoss because this is a multiclass
# classification problem. We don’t have to manually apply a log_softmax layer
# after our final layer because nn.CrossEntropyLoss does that for us. However,
# we need to apply log_softmax for our validation and testing.

model = MulticlassClassification(num_feature = NUM_FEATURES, num_class=NUM_CLASSES)
model.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
print(model)

def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    
    acc = torch.round(acc * 100)
    
    return acc

accuracy_stats = {
    'train': [],
    "val": []
}
loss_stats = {
    'train': [],
    "val": []
}    

print("Begin training.")
for e in tqdm(range(1, EPOCHS+1)):
    
    # TRAINING
    train_epoch_loss = 0
    train_epoch_acc = 0
    model.train()
    for X_train_batch, y_train_batch in train_loader:
        X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
        optimizer.zero_grad()
        
        y_train_pred = model(X_train_batch)
        
        train_loss = criterion(y_train_pred, y_train_batch)
        train_acc = multi_acc(y_train_pred, y_train_batch)
        
        train_loss.backward()
        optimizer.step()
        
        train_epoch_loss += train_loss.item()
        train_epoch_acc += train_acc.item()
        
        
    # VALIDATION   
    # NOTE The procedure we follow for training is the exact same for validation
    # except for the fact that we wrap it up in torch.no_grad and not perform
    # any back-propagation. torch.no_grad() tells PyTorch that we do not want to
    # perform back-propagation, which reduces memory usage and speeds up
    # computation.
    with torch.no_grad():
        # NOTE At the top of this for-loop, we initialize our loss and accuracy
        # per epoch to 0. After every epoch, we’ll print out the loss/accuracy
        # and reset it back to 0.
        val_epoch_loss = 0
        val_epoch_acc = 0
        
        model.eval()
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
            
            y_val_pred = model(X_val_batch)
                        
            val_loss = criterion(y_val_pred, y_val_batch)
            val_acc = multi_acc(y_val_pred, y_val_batch)
            
            val_epoch_loss += val_loss.item()
            val_epoch_acc += val_acc.item()
    loss_stats['train'].append(train_epoch_loss/len(train_loader))
    loss_stats['val'].append(val_epoch_loss/len(val_loader))
    accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
    accuracy_stats['val'].append(val_epoch_acc/len(val_loader))
                              
    
    print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| Val Acc: {val_epoch_acc/len(val_loader):.3f}')

# NOTE You can see we’ve put a model.train() at the before the loop.
# model.train() tells PyTorch that you’re in training mode. Well, why do we need
# to do that? If you’re using layers such as Dropout or BatchNorm which behave
# differently during training and evaluation (for example; not use dropout
# during evaluation), you need to tell PyTorch to act accordingly. Similarly,
# we’ll call model.eval() when we test our model. We’ll see that below.

# NOTE After training is done, we need to test how our model fared. Note that we’ve
# used model.eval() before we run our testing code. To tell PyTorch that we do
# not want to perform back-propagation during inference, we use torch.no_grad(),
# just like we did it for the validation loop above.
y_pred_list = []
with torch.no_grad():
    model.eval()
    for X_batch, _ in test_loader:
        # We move our input mini-batch to GPU.
        X_batch = X_batch.to(device)
        # Apply log_softmax activation to the predictions and pick the index of highest probability.
        y_test_pred = model(X_batch)
        _, y_pred_tags = torch.max(y_test_pred, dim = 1)
        # Move the batch to the GPU from the CPU.
        # Convert the tensor to a numpy object and append it to our list.      
        y_pred_list.append(y_pred_tags.cpu().numpy())
# Flatten out the list so that we can use it as an input to confusion_matrix and classification_report.  
y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
# %%
