# %%
from abc import abstractclassmethod
import pandas as pd
import numpy as np
import os
os.chdir('/kaggle/working')
train=pd.read_csv("../input/tabular-playground-series-may-2021/train.csv",index_col = 0)
test=pd.read_csv("../input/tabular-playground-series-may-2021/test.csv",index_col = 0)
sample=pd.read_csv("../input/tabular-playground-series-may-2021/sample_submission.csv")
# Preprocessing. Take only this observations
features=[f'feature_{i}' for i in range(50)]


# %%
def avg(models,weights):
    sub_lb=0
    for n_model,model in enumerate(models): sub_lb+=model*weights[n_model]
    sub_lb.id=sample.id
    sub_lb.to_csv(f'sub_lb_{sub_lb.iloc[0,1]}_{weights[0]}.csv',index=False) #sub_lb.iloc[0,1] is and index
    return sub_lb

pseudo = pd.read_csv('/kaggle/working/may_model/lazo/stacking_1.085187436530773.csv')
ybf2 = pd.read_csv('/kaggle/working/may_model/lazo/blending_1.0882100959554668.csv')
ybf1 = pd.read_csv('/kaggle/working/may_model/lazo/blending_1.0874228185612054.csv')
stacking = pd.read_csv('/kaggle/working/may_model/lazo/stacking_1.0858039765820955.csv')
avg1 = pd.read_csv('/kaggle/working/may_model/lazo/sub_lb_0.09080355083449096_0.7.csv')
avg2 = pd.read_csv('/kaggle/working/may_model/lazo/sub_lb_0.09075169522314322_0.7.csv')
avg3 = pd.read_csv('/kaggle/working/may_model/lazo/sub_lb_0.08744196634592771_0.7.csv')

ysf = stacking
yspf = pseudo
bset_lb = ybf2
yvg_1=avg([best_lb,ysf],[0.7,0.3]) #lb:1.08510
yvg_2=avg([best_lb,yspf],[0.7,0.3])
yvg_3=avg([best_lb,yspf,ybf2],[0.7,0.25,0.05])

yvg_4 = avg([ybf2,ysf, yspf],[0.7,0.25,0.05])

for tab in [yvg_1, yvg_2, yvg_3]:
    display(tab.head())

for tab in [avg1, avg2, avg3]:
    display(tab.head())    


# The author uses it.
bset_lb = ybf1
yvg_1a=avg([best_lb,ysf],[0.7,0.3]) #lb:1.08510
yvg_2a=avg([best_lb,yspf],[0.7,0.3])
yvg_3a=avg([best_lb,yspf,ybf2],[0.7,0.25,0.05])

for tab in [yvg_1a, yvg_2a, yvg_3a]:
    display(tab.head())
# %%
abcde = pd.read_csv('/kaggle/working/may_model/lazo/abcde.csv')
sub857 = pd.read_csv('/kaggle/working/may_model/lazo/sub8571.csv')
hydra = pd.read_csv('/kaggle/working/may_model/lazo/hydra.csv')

exp1 = avg([yvg_1a, abcde, sub857], [0.333, 0.333, 0.333])
exp2 = avg([yvg_4, abcde, sub857], [0.333, 0.333, 0.333])
exp3 = avg([yvg_1a, sub857], [0.7, 0.3])
exp4 = avg([yvg_1a, hydra, sub857], [0.333, 0.333, 0.333])
exp5 = avg([yvg_1a, hydra, sub857], [0.55, 0.4, 0.05])

exp1.to_csv('/kaggle/working/may_model/lazo/sub-exp1.csv', index=False) # 1.08510
exp2.to_csv('/kaggle/working/may_model/lazo/sub-exp2.csv', index=False) # 1.08522
exp3.to_csv('/kaggle/working/may_model/lazo/sub-exp3.csv', index=False) # 1.08520
exp4.to_csv('/kaggle/working/may_model/lazo/sub-exp4.csv', index=False) # 1.08517
exp5.to_csv('/kaggle/working/may_model/lazo/sub-exp5.csv', index=False) # 1.08524