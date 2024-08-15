import os
from glob import glob
import random
from sklearn.model_selection import KFold,StratifiedKFold
import joblib
import numpy as np



root = "/data1/zzf/pathdata/Graph_data/TISSUE_GRAPH"
data_list = [pkl_name.split('/')[-1] for pkl_name in glob(os.path.join(root,'*.pkl'))]
random.shuffle(data_list)
Y = [int(dl[:-4].split('_')[-1])for dl in data_list]
kf = StratifiedKFold(n_splits=5)

fold_dict = {}
cnt=0
for train_idx, test_idx in kf.split(data_list,Y):
    tmp_train_list = [data_list[_idx] for _idx in train_idx]
    val_list = random.sample(tmp_train_list,len(tmp_train_list)//5)
    train_list = list(set(tmp_train_list)-set(val_list))
    test_list  = [data_list[_idx] for _idx in test_idx]
    train_Y = [int(dl[:-4].split('_')[-1])for dl in train_list]
    val_Y = [int(dl[:-4].split('_')[-1])for dl in val_list]
    test_Y = [int(dl[:-4].split('_')[-1])for dl in test_list]
    fold_dict[f"train_{cnt}"]=train_list
    fold_dict[f"val_{cnt}"]=val_list
    fold_dict[f"test_{cnt}"]=test_list
    cnt += 1

joblib.dump(fold_dict,"/data2/cm/pathdata/five_fold.pkl")