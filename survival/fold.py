import joblib
import os  
import random 
import torch 
import numpy as np
from sklearn.model_selection import StratifiedKFold

patients = joblib.load('/path_to/esca_patients_155.pkl')
#  package features
all_features = {}
for f in os.listdir('/path_to/pt_files/'):
    print('-'.join(f.split('-')[:3]))
    if '-'.join(f.split('-')[:3]) in patients:
        all_features['-'.join(f.split('-')[:3])]=torch.load('/path_to/pt_files/'+f,map_location='cpu')

joblib.dump(all_features,'/path_to/esca_155_features.pkl')

all_graphs = {}
for f in os.listdir('/path_to/graph_files/'):
    print('-'.join(f.split('-')[:3]))
    if '-'.join(f.split('-')[:3]) in patients:
        all_graphs['-'.join(f.split('-')[:3])]=torch.load('/path_to/graph_files/'+f,map_location='cpu')

joblib.dump(all_graphs,'/path_to/esca_155_graphs.pkl')

# Five folds   
S_T = joblib.load('/path_to/esca_sur_and_time.pkl')
new_S_T = {k:S_T[k] for k in patients}
items = list(new_S_T.items())
random.shuffle(items)
shuffled_S_T = dict(items)

folds = {}
X = list(shuffled_S_T.keys())
censor = [shuffled_S_T[k][0] for k in shuffled_S_T]
skf = StratifiedKFold(n_splits=5)
for i,(train_index,test_index) in enumerate(skf.split(X,censor)):
    tmp_train_list = [X[t_idx] for t_idx in train_index]
    test_list = [X[t_idx] for t_idx in test_index]
    val_list = random.sample(tmp_train_list,len(test_list))
    train_list = list(set(tmp_train_list)-set(val_list))
    folds[f'train_{i}']=train_list
    folds[f'val_{i}']=val_list
    folds[f'test_{i}']=test_list

# print(folds)
for k in folds:
    print(k,len(folds[k]))
joblib.dump(folds,'/path_to/kirc_five_folds.pkl')