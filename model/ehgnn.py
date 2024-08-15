import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch_scatter import scatter_add
from torch_geometric.utils import softmax, degree, to_dense_batch
from torch_geometric.nn.conv import MessagePassing, GCNConv
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn.pool.topk_pool import topk
import math


### Hypergraph convolution for message passing on Dual Hypergraph 
class HypergraphConv(MessagePassing):

    def __init__(self, in_channels, out_channels, use_attention=False, heads=1,
                 concat=True, negative_slope=0.2, dropout=0, bias=True,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(HypergraphConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention

        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.weight = Parameter(
                torch.Tensor(in_channels, heads * out_channels))
            self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))
        else:
            self.heads = 1
            self.concat = True
            self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        if self.use_attention:
            glorot(self.att)
        zeros(self.bias)

    def message(self, x_j, edge_index_i, norm, alpha):
        out = norm[edge_index_i].view(-1, 1, 1) * x_j.view(-1, self.heads, self.out_channels)

        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out
        return out

    def forward(self, x, hyperedge_index, hyperedge_weight=None):
        
        x = torch.matmul(x, self.weight)
        alpha = None

        if self.use_attention:
            x = x.view(-1, self.heads, self.out_channels)
            x_i, x_j = x[hyperedge_index[0]], x[hyperedge_index[1]]
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = softmax(alpha, hyperedge_index[0], num_nodes=x.size(0))
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        if hyperedge_weight is None:
            D = degree(hyperedge_index[0], x.size(0), x.dtype)
        else:
            D = scatter_add(hyperedge_weight[hyperedge_index[1]],
                            hyperedge_index[0], dim=0, dim_size=x.size(0))
        D = 1.0 / D
        D[D == float("inf")] = 0

        if hyperedge_index.numel() == 0:
            num_edges = 0
        else:
            num_edges = hyperedge_index[1].max().item() + 1
        B = 1.0 / degree(hyperedge_index[1], num_edges, x.dtype)
        B[B == float("inf")] = 0
        if hyperedge_weight is not None:
            B = B * hyperedge_weight

        num_nodes = x.size(0)
        dif = max([num_nodes, num_edges]) - num_nodes # get size of padding
        x_help = F.pad(x, (0,0,0, dif)) # create dif many nodes

        self.flow = 'source_to_target'
        out = self.propagate(hyperedge_index, x=x_help, norm=B, alpha=alpha)
        self.flow = 'target_to_source'
        out = self.propagate(hyperedge_index, x=out, norm=D, alpha=alpha)

        out = out[:num_nodes] # prune back to original x.size()

        if self.concat is True:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
                            
class GraphRepresentation(nn.Module):

    def __init__(self, **kwargs): 

        super(GraphRepresentation, self).__init__()

        self.num_node_features = kwargs.get('num_node_features')
        self.num_edge_features = kwargs.get('num_edge_features')
        self.nhid = kwargs.get('nhid')
        self.edge_ratio = kwargs.get('edge_ratio')
        self.enhid = kwargs.get('enhid')
        self.num_convs = kwargs.get('num_convs')
        self.time_emb = kwargs.get('time_emb')
        self.context = kwargs.get('context')

    ### Dual Hypergraph Transformation (DHT)
    def DHT(self, edge_index, batch, add_loops=True):

        num_edge = edge_index.size(1)
        device = edge_index.device

        ### Transform edge list of the original graph to hyperedge list of the dual hypergraph
        edge_to_node_index = torch.arange(0,num_edge,1, device=device).repeat_interleave(2).view(1,-1)
        hyperedge_index = edge_index.T.reshape(1,-1)
        hyperedge_index = torch.cat([edge_to_node_index, hyperedge_index], dim=0).long() 

        ### Transform batch of nodes to batch of edges
        edge_batch = hyperedge_index[1,:].reshape(-1,2)[:,0]
        edge_batch = torch.index_select(batch, 0, edge_batch)

        ### Add self-loops to each node in the dual hypergraph
        if add_loops:
            bincount =  hyperedge_index[1].bincount()
            mask = bincount[hyperedge_index[1]]!=1
            max_edge = hyperedge_index[1].max()
            loops = torch.cat([torch.arange(0,num_edge,1,device=device).view(1,-1), 
                                torch.arange(max_edge+1,max_edge+num_edge+1,1,device=device).view(1,-1)], 
                                dim=0)

            hyperedge_index = torch.cat([hyperedge_index[:,mask], loops], dim=1)

        return hyperedge_index, edge_batch

    def get_scoreconvs(self):

        convs = nn.ModuleList()

        for i in range(self.num_convs-1):

            conv = HypergraphConv(self.enhid, 1)
            convs.append(conv)

        return convs

class Model_Hyper_Encoder(GraphRepresentation):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.convs = self.get_convs()
        self.hyperconvs = self.get_convs(conv_type='Hyper')
        self.scoreconvs = self.get_scoreconvs()

        self.mlp_e_m = nn.Sequential(
            nn.Linear(self.enhid, 256),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, self.enhid),
            )

        self.mlp_e_v = nn.Sequential(
            nn.Linear(self.enhid, 256),
            nn.Linear(256, 64),
            nn.Linear(64, self.enhid),
            nn.ReLU())
    
        self.mlp_x_m = nn.Sequential(
            nn.Linear(self.nhid, 256),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, self.nhid),
            )

        self.mlp_x_v = nn.Sequential(
            nn.Linear(self.nhid, 256),
            nn.Linear(256, 64),
            nn.Linear(64, self.nhid),
            nn.ReLU())
        
        self.edge_attr_emblin = nn.Linear(self.num_edge_features, self.num_edge_features) 
        self.node_emblin = nn.Linear(self.num_node_features, self.num_node_features) 

    def reparameterize(self, mean, logvar):
        std = torch.exp(logvar / 2)
        epsilon = torch.randn_like(std)
        return epsilon * std + mean

    def forward(self, x, edge_index_, edge_attr_, batch, x_mask=None, edge_mask=None, first_layer=False, is_eval=False):

        ### Edge feature initialization
        x_init = x
        if first_layer and x_mask is not None:
            pass
            x[x_mask]=0
        if edge_mask is not None:
            edge_attr = edge_attr_[~edge_mask]
            edge_index = edge_index_[:,~edge_mask]
        else:
            edge_attr = edge_attr_
            edge_index = edge_index_
        # if self.time_emb and self.context==False:
        #     h, temb = edge_attr[:, :self.num_edge_features], edge_attr[:, self.num_edge_features:]
        #     edge_attr = self.edge_attr_emblin(h) + temb
        #     h, temb = x[:, :self.num_node_features], x[:, self.num_node_features:]
        #     x = self.node_emblin(h) + temb
        # else:
        edge_attr = self.edge_attr_emblin(edge_attr)
        x = self.node_emblin(x)
        if edge_attr is None:
            edge_attr = torch.ones((edge_index.size(1), 1), device=edge_index.device)

        x_List = []
        edge_attr_List = []

        for _ in range(self.num_convs):
            hyperedge_index, edge_batch = self.DHT(edge_index, batch)
            if _ == 0:
                x = F.relu(self.convs[_](x, edge_index))
            elif _ < self.num_convs-1:
                x = F.relu(self.convs[_](x, edge_index, edge_weight))
            else:
                x = self.convs[_](x, edge_index, edge_weight)
            
            if _ < self.num_convs-1:

                edge_attr = F.relu( self.hyperconvs[_](edge_attr, hyperedge_index))

                score = torch.tanh( self.scoreconvs[_](edge_attr, hyperedge_index).squeeze())
                # perm = topk(score, self.edge_ratio, edge_batch)

                # edge_index = edge_index[:,perm]
                # edge_attr = edge_attr[perm, :]
                # edge_weight = score[perm]
                edge_weight = score
                edge_weight = torch.clamp(edge_weight, min=0, max=1)
            else:
                edge_attr = self.hyperconvs[_](edge_attr, hyperedge_index)

            # xs += torch.cat([gmp(x,batch), gap(x,batch), gsp(x,batch)], dim=1)
            if not is_eval and self.context:
                edge_attr_List.append(edge_attr)
                x_List.append(x)
        # x = self.classifier(xs)
        if not is_eval:
            if self.context:
                h_e = edge_attr # E x E_H
                h_x = x
                m_e = self.mlp_e_m(h_e)
                v_e = self.mlp_e_v(h_e)
                m_x = self.mlp_x_m(h_x)
                v_x = self.mlp_x_m(h_x)
                # z_x = self.reparameterize(m_x,v_x)
                # z_e = self.reparameterize(m_e,v_e)
                # print(len(edge_attr_List))
                return m_x, v_x, x_List, m_e, v_e, edge_attr_List
            else:
                return x, edge_index, edge_attr
        else:
            h_e = edge_attr # E x E_H
            h_x = x
            m_e = self.mlp_e_m(h_e)
            v_e = self.mlp_e_v(h_e)
            m_x = self.mlp_x_m(h_x)
            v_x = self.mlp_x_m(h_x)
            return m_x, v_x, m_e, v_e

    def get_convs(self, conv_type='GCN'):

        convs = nn.ModuleList()

        for i in range(self.num_convs):

            if conv_type == 'GCN':

                if i == 0 :
                    conv = GCNConv(self.num_node_features, self.nhid)
                else:
                    conv = GCNConv(self.nhid, self.nhid)

            elif conv_type == 'Hyper':

                if i == 0 :
                    conv = HypergraphConv(self.num_edge_features, self.enhid)
                else:
                    conv = HypergraphConv(self.enhid, self.enhid)

            else:
                raise ValueError("Conv Name <{}> is Unknown".format(conv_type))

            convs.append(conv)

        return convs
