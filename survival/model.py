import os
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.utils import softmax
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from torch_geometric.nn import DeepGCNLayer, SGConv, GCNConv,SAGEConv,GATConv,TransformerConv,GINConv,TAGConv,GENConv,SAGPooling,global_mean_pool,global_max_pool,GlobalAttention_gated
from torch.nn import Sequential as Seq
from torch.nn import Linear, LayerNorm, ReLU
from torch_geometric.nn import GraphConv, TopKPooling, SAGPooling
from torch_geometric.nn import global_mean_pool as gavgp, global_max_pool as gmp, global_add_pool as gap
from torch_geometric.transforms.normalize_features import NormalizeFeatures


class Attn_Net_Gated(nn.Module):

    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)
        return A, x
    
    
def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()
        
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

class AMIL_Surv(nn.Module):
    def __init__(self, in_feats=1024, size_arg="small", dropout=0.25, fusion_dim=128,output_dim=1):
        super(AMIL_Surv, self).__init__()
        self.size_dict = {"small": [in_feats, 512, 256], "big": [in_feats, 512, 384]}
        size = self.size_dict[size_arg]
        
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.features = nn.Sequential(*[nn.Linear(size[1], fusion_dim), nn.ReLU(), nn.Dropout(dropout)])
        self.linear = nn.Linear(fusion_dim, output_dim)

        initialize_weights(self)

    def forward(self, x):
        A, h = self.attention_net(x)
        A = torch.transpose(A, 1, 0)
        
        A = F.softmax(A, dim=1) 
        h = torch.mm(A, h) 
        h = self.features(h)

        logits = self.linear(h)

        hazards = torch.sigmoid(logits)
        
        return hazards 

class PatchGCN_Surv(torch.nn.Module):
    def __init__(self, num_layers=4, edge_agg='spatial', multires=False, resample=0,
        fusion=None, num_features=1024, hidden_dim=128, linear_dim=64, use_edges=False, pool=False, dropout=0.25, n_classes=1):
        super(PatchGCN_Surv, self).__init__()
        self.use_edges = use_edges
        self.fusion = fusion
        self.pool = pool
        self.edge_agg = edge_agg
        self.multires = multires
        self.num_layers = num_layers-1
        self.resample = resample

        if self.resample > 0:
            self.fc = nn.Sequential(*[nn.Dropout(self.resample), nn.Linear(num_features, 256), nn.ReLU(), nn.Dropout(0.25)])
        else:
            self.fc = nn.Sequential(*[nn.Linear(num_features, hidden_dim), nn.ReLU(), nn.Dropout(0.25)])

        self.layers = torch.nn.ModuleList()
        for i in range(1, self.num_layers+1):
            conv = GENConv(hidden_dim, hidden_dim, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(hidden_dim, elementwise_affine=True)
            act = ReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm, act, block='res', dropout=0.1, ckpt_grad=i % 3)
            self.layers.append(layer)

        self.path_phi = nn.Sequential(*[nn.Linear(hidden_dim*4, hidden_dim*4), nn.ReLU(), nn.Dropout(0.25)])

        self.path_attention_head = Attn_Net_Gated(L=hidden_dim*4, D=hidden_dim*4, dropout=dropout, n_classes=1)
        self.path_rho = nn.Sequential(*[nn.Linear(hidden_dim*4, hidden_dim*4), nn.ReLU()])

        self.classifier = torch.nn.Linear(hidden_dim*4, n_classes)    

    def forward(self, data, **kwargs):
                
        if self.edge_agg == 'spatial':
            edge_index = data.edge_index
        elif self.edge_agg == 'latent':
            edge_index = data.edge_latent

        batch = data.batch
        edge_attr = None

        x = self.fc(data.x)
        x_ = x 
        
        x = self.layers[0].conv(x_, edge_index, edge_attr)
        x_ = torch.cat([x_, x], axis=1)
        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)
            x_ = torch.cat([x_, x], axis=1)
        
        h_path = x_
        h_path = self.path_phi(h_path)

        A_path, h_path = self.path_attention_head(h_path)
        A_path = torch.transpose(A_path, 1, 0)
        h_path = torch.mm(F.softmax(A_path, dim=1), h_path)
        h = self.path_rho(h_path).squeeze()
        logits  = self.classifier(h).unsqueeze(0) # logits needs to be a [1 x 4] vector 
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        
        return hazards

class DeepGraphConv_Surv(torch.nn.Module):
    def __init__(self, edge_agg='spatial', resample=0, num_features=1024, hidden_dim=256, 
        linear_dim=256, use_edges=False, pool=True, dropout=0.25, n_classes=4 ,device='cuda:0', ratio=0.25):
        super(DeepGraphConv_Surv, self).__init__()
        self.use_edges = use_edges
        self.resample = resample
        self.edge_agg = edge_agg
        self.pool = pool
        self.device = device

        if self.pool:
            self.pool1 = SAGPooling(in_channels=hidden_dim,ratio=ratio)
            self.pool2 = SAGPooling(in_channels=hidden_dim,ratio=ratio)
        if self.resample > 0:
            self.fc = nn.Sequential(*[nn.Dropout(self.resample)])

        self.conv1 = GINConv(Seq(nn.Linear(num_features, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.conv2 = GINConv(Seq(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.conv3 = GINConv(Seq(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        
        self.path_attention_head = Attn_Net_Gated(L=hidden_dim, D=hidden_dim, dropout=dropout, n_classes=1)
        self.path_rho = nn.Sequential(*[nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.classifier = torch.nn.Linear(hidden_dim, n_classes)

    def relocate(self):
        from torch_geometric.nn import DataParallel
        if torch.cuda.device_count() >= 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.conv1 = nn.DataParallel(self.conv1, device_ids=device_ids).to(self.device)
            self.conv2 = nn.DataParallel(self.conv2, device_ids=device_ids).to(self.device)
            self.conv3 = nn.DataParallel(self.conv3, device_ids=device_ids).to(self.device)
            self.path_attention_head = nn.DataParallel(self.path_attention_head, device_ids=device_ids).to(self.device)

        self.path_rho = self.path_rho.to(device)
        self.classifier = self.classifier.to(device)

    def forward(self, data):
        x = data.x
        if self.edge_agg == 'spatial':
            edge_index = data.edge_index
        elif self.edge_agg == 'latent':
            edge_index = data.edge_latent
        batch = data.batch
        edge_attr = None

        if self.resample:
            x = self.fc(x)

        x1 = F.relu(self.conv1(x, edge_index, edge_attr))
        if self.pool:
            x1, edge_index, _, batch, perm, score = self.pool1(x1, edge_index, None, batch)
            x1_cat = torch.cat([gmp(x1, batch), gap(x1, batch)], dim=1)

        x2 = F.relu(self.conv2(x1, edge_index, edge_attr))
        if self.pool:
            x2, edge_index, _, batch, perm, score = self.pool2(x2, edge_index, None, batch)
            x2_cat = torch.cat([gmp(x2, batch), gap(x2, batch)], dim=1)

        x3 = F.relu(self.conv3(x2, edge_index, edge_attr))
        

        h_path = x3

        A_path, h_path = self.path_attention_head(h_path)
        A_path = torch.transpose(A_path, 1, 0)
        h_path = torch.mm(F.softmax(A_path, dim=1) , h_path)
        h_path = self.path_rho(h_path).squeeze()
        h = h_path # [256] vector
        
        logits  = self.classifier(h).unsqueeze(0) # logits needs to be a [1 x 4] vector 
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        
        return hazards

class MIL_Sum_FC_surv(nn.Module):
    def __init__(self, num_features=1024, size_arg = "small", dropout=0.25, n_classes=4):
        r"""
        Deep Sets Implementation.
        Args:
            size_arg (str): Size of NN architecture (Choices: small or large)
            dropout (float): Dropout rate
            n_classes (int): Output shape of NN
        """
        super(MIL_Sum_FC_surv, self).__init__()
        self.size_dict_path = {"small": [num_features, 512, 256], "big": [num_features, 512, 384]}

        ### Deep Sets Architecture Construction
        size = self.size_dict_path[size_arg]
        self.phi = nn.Sequential(*[nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)])
        self.rho = nn.Sequential(*[nn.Linear(size[1], size[2]), nn.ReLU(), nn.Dropout(dropout)])
        
        self.classifier = nn.Linear(size[2], n_classes)


    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() >= 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.phi = nn.DataParallel(self.phi, device_ids=device_ids).to('cuda:0')

        self.rho = self.rho.to(device)
        self.classifier = self.classifier.to(device)


    def forward(self, x_path):

        h_path = self.phi(x_path).sum(axis=0)
        h_path = self.rho(h_path)
        h = h_path # [256] vector

        logits  = self.classifier(h).unsqueeze(0) # logits needs to be a [1 x 4] vector 
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        
        return hazards

class MIL_Cluster_FC_surv(nn.Module):
    def __init__(self, num_features=1024, num_clusters=10, size_arg = "small", dropout=0.25, n_classes=4):
        r"""
        Attention MIL Implementation

        Args:
            size_arg (str): Size of NN architecture (Choices: small or large)
            dropout (float): Dropout rate
            n_classes (int): Output shape of NN
        """
        super(MIL_Cluster_FC_surv, self).__init__()
        self.size_dict_path = {"small": [num_features, 512, 256], "big": [num_features, 512, 384]}
        self.num_clusters = num_clusters
        
        ### FC Cluster layers + Pooling
        size = self.size_dict_path[size_arg]
        phis = []
        for phenotype_i in range(num_clusters):
            phi = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout),
                   nn.Linear(size[1], size[1]), nn.ReLU(), nn.Dropout(dropout)]
            phis.append(nn.Sequential(*phi))
        self.phis = nn.ModuleList(phis)
        self.pool1d = nn.AdaptiveAvgPool1d(1)
        
        ### WSI Attention MIL Construction
        fc = [nn.Linear(size[1], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.rho = nn.Sequential(*[nn.Linear(size[1], size[2]), nn.ReLU(), nn.Dropout(dropout)])

        self.classifier = nn.Linear(size[2], n_classes)


    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() >= 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.attention_net = nn.DataParallel(self.attention_net, device_ids=device_ids).to('cuda:0')
        else:
            self.attention_net = self.attention_net.to(device)

        self.phis = self.phis.to(device)
        self.pool1d = self.pool1d.to(device)
        self.rho = self.rho.to(device)
        self.classifier = self.classifier.to(device)


    def forward(self, **kwargs):
        x_path = kwargs['x_path']
        cluster_id = kwargs['cluster_id'].detach().cpu().numpy()

        ### FC Cluster layers + Pooling
        h_cluster = []
        for i in range(self.num_clusters):
            h_cluster_i = self.phis[i](x_path[cluster_id==i])
            if h_cluster_i.shape[0] == 0:
                h_cluster_i = torch.zeros((1,512)).to(torch.device('cuda'))
            h_cluster.append(self.pool1d(h_cluster_i.T.unsqueeze(0)).squeeze(2))
        h_cluster = torch.stack(h_cluster, dim=1).squeeze(0)

        ### Attention MIL
        A, h_path = self.attention_net(h_cluster)  
        A = torch.transpose(A, 1, 0)
        A_raw = A 
        A = F.softmax(A, dim=1) 
        h_path = torch.mm(A, h_path)
        h_path = self.rho(h_path).squeeze()
        h = h_path

        logits  = self.classifier(h).unsqueeze(0)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        
        return hazards, S, Y_hat, None, None