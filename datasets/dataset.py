import os 
import torch
from typing import Mapping, Optional, Callable, List, Union, Tuple, Dict
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset, Dataset
from torch_geometric.utils import remove_self_loops
from glob import glob 


import os 
import torch
from typing import Mapping, Optional, Callable, List, Union, Tuple, Dict
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset, Dataset
from torch_geometric.utils import remove_self_loops
from glob import glob 
import random
import joblib


class PathDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, is_eval=False, mod='train', fold=0):
        self.root = root
        self.is_eval = is_eval
        
        if is_eval:
            self.fold_split = joblib.load("/path_to/fold_splits.pkl")
            print(f"{mod}_{fold}",len(self.fold_split[f"{mod}_{fold}"]))
            self.data_list = self.fold_split[f"{mod}_{fold}"]
        else:
            self.data_list = [pkl_name.split('/')[-1] for pkl_name in glob(os.path.join(root,'*.pkl'))]

        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return self.data_list

    @property
    def processed_file_names(self):
        return self.data_list

    def download(self):
        pass

    def process(self):
        pass

    def len(self):
        if self.is_eval:
            return len(self.data_list)
        else:
            return len(self.data_list)

    def get(self, idx):
        data = torch.load(os.path.join(self.root, self.data_list[idx]))
        data.name = self.data_list[idx]
        data.edge_index, data.edge_attr = remove_self_loops(data.edge_index, data.edge_attr)
        if self.is_eval:
            
            return data, int(self.data_list[idx][:-4].split('_')[-1])-1, os.path.join(self.root, self.data_list[idx])
        else:
            return data, os.path.join(self.root, self.data_list[idx])