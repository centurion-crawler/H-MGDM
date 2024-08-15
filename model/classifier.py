from typing import Optional, Tuple
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from .common import MultiLayerPerceptron
from torch_geometric.nn import GlobalAttention
from torch_geometric.nn import SAGPooling, GlobalAttention_gated, TopKPooling, global_max_pool, global_mean_pool
class classifier(nn.Module):
    def __init__(self,config):
        super(classifier,self).__init__()
        self.config = config
        self.num_classes = config.num_classes 
        self.pool_name = config.pool_name
        self.x_pool = self.get_pool(pool_name=self.pool_name)
        self.final_MLP = MultiLayerPerceptron(
            config.model.hidden_dim,
            [config.model.hidden_dim//2, config.num_classes],
            activation='relu'
        )
    
    def get_pool(self,pool_name):
        hd = self.config.model.hidden_dim
        if pool_name == 'SAG':
            return SAGPooling(hd,1)
        elif pool_name == 'GAP':
            return GlobalAttention_gated(gate_nn = nn.Sequential(nn.Linear(hd, hd//2),nn.BatchNorm1d(hd//2),nn.ReLU(),nn.Linear(hd//2,1)))
        elif pool_name == 'TopK':
            return TopKPooling(hd,1)
        elif pool_name == 'Max':
            return global_max_pool
        elif pool_name == 'Mean':
            return global_mean_pool
        else:
            raise NotImplementedError
        # elif pool_name == 'Edge':
        #     return EdgePooling(hd,1)
        

    
    def forward(self,enc_data, batch_node, batch_edge):
        x_e, pos_e, edge_index_e, edge_attr_e = enc_data
        print('x_e:',x_e.shape)
        if self.pool_name == 'SAG':
            x_e_pool, edge_index_e_pool, edge_attr_e_pool, _, _, _ = self.x_pool(x=x_e, edge_index=edge_index_e,edge_attr=edge_attr_e, batch=batch_node)
        elif self.pool_name == 'GAP':
            x_e_pool,gate = self.x_pool(x_e,batch_node)
        elif self.pool_name =='TopK':
            x_e_pool, edge_index_e_pool, edge_attr_e_pool, _, _, _ = self.x_pool(x=x_e,edge_index=edge_index_e, edge_attr=edge_attr_e, batch=batch_node)
        elif self.pool_name == 'Max':
            x_e_pool = self.x_pool(x=x_e, batch=batch_node)
        elif self.pool_name == 'Mean':
            x_e_pool = self.x_pool(x=x_e, batch=batch_node)
        
        logits = self.final_MLP(x_e_pool)
        if self.pool_name == 'GAP':
            return logits, gate
        else:
            return logits