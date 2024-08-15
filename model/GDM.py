from typing import Optional, Tuple
import torch
from torch import nn
from torch import Tensor
from torch_scatter import scatter_mean 
from tqdm.auto import tqdm
from .gat import GAT
from .gin import GIN
from .ehgnn import Model_Hyper_Encoder
from .transformer import Transformer
from .transformer import MultiheadAttention as CoAttention
from .common import MultiLayerPerceptron, assemble_pair_feature
from .diffusion import get_timestep_embedding, get_beta_schedule
from .geometry import get_distance, eq_transform

from torch_geometric.deprecation import deprecated
from torch_geometric.typing import OptTensor
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch.cuda.amp import autocast as autocast


def detect_nan(x):
    return torch.isnan(x).any()

class GDM(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.config = config
        self.time_emb = True
        self.context = config.context if 'context' in config else []
        self.node_input_dim = config.node_input_dim
        self.edge_input_dim = config.edge_input_dim
        self.origin_node_in_dim = config.origin_node_in_dim
        self.origin_edge_in_dim = config.origin_edge_in_dim
        self.max_node_num = config.max_node_num
        self.max_edge_num = config.max_edge_num
        self.hidden_dim = config.hidden_dim
        self.vae_context = config.vae_context if 'vae_context' in config else False
        self.node_mask_ratio = config.node_mask_ratio 
        self.edge_mask_ratio = config.edge_mask_ratio
        self.ehgnn_edge_ratio = config.ehgnn_edge_ratio 
        self.cross_attention_mode = config.cross_attention_mode # ['x2e','x2x','e2x','e2e']
        self.all_num_layers = config.all_num_layers

        self.x_embedding = nn.Sequential(
            nn.LayerNorm(self.origin_node_in_dim),
            nn.Linear(self.origin_node_in_dim,self.node_input_dim),
        )

        self.edge_attr_embedding = nn.Sequential(
            nn.LayerNorm(self.origin_edge_in_dim),
            nn.Linear(self.origin_edge_in_dim,self.edge_input_dim),
        )

        if self.time_emb:
            self.temb = nn.Module()
            self.temb.dense = nn.ModuleList([
                torch.nn.Linear(self.hidden_dim,
                                self.hidden_dim * 4),
                torch.nn.Linear(self.hidden_dim * 4,
                                self.hidden_dim * 4),
            ])
            self.temb_proj = torch.nn.Linear(self.hidden_dim * 4,
                                                self.hidden_dim)
        self.dec_mode = config.dec_mode

        if self.vae_context:
                self.context_encoder=Model_Hyper_Encoder(
                    num_node_features = self.node_input_dim,
                    num_edge_features = self.edge_input_dim,
                    nhid = self.hidden_dim,
                    edge_ratio = self.ehgnn_edge_ratio,
                    enhid = self.hidden_dim,
                    num_convs = config.all_num_layers,
                    time_emb=False,
                    context = self.vae_context
                )


        if self.context is not None:
            ctx_nf = len(self.context)
        if 'joint' in self.dec_mode:
            self.joint_decoder_x = Transformer(
                dim=self.context_encoder.nhid, 
                depth=config.joint_depth, 
                heads=config.joint_num_heads, 
                dim_head=(self.context_encoder.nhid)//config.joint_num_heads,
                mlp_dim=self.hidden_dim
            )
            self.joint_decoder_e = Transformer(
                dim=self.context_encoder.enhid, 
                depth=config.joint_depth, 
                heads=config.joint_num_heads, 
                dim_head=(self.context_encoder.enhid)//config.joint_num_heads,
                mlp_dim=self.hidden_dim
            )
        if 'coatt' in self.dec_mode:
            self.dec_blks = nn.ModuleList([])
            if 'joint' in self.dec_mode:
                temp_layers = config.all_num_layers-1
            for _ in range(temp_layers):
                self.dec_blks.append(nn.ModuleList([
                    CoAttention(self.context_encoder.nhid, config.coatt_num_heads), # node
                    CoAttention(self.context_encoder.enhid, config.coatt_num_heads), # edge_attr
                    Model_Hyper_Encoder(
                        num_node_features = self.hidden_dim,
                        num_edge_features = self.hidden_dim,
                        nhid = self.hidden_dim,
                        edge_ratio = self.ehgnn_edge_ratio,
                        enhid = self.hidden_dim,
                        num_convs = 1,
                        time_emb= True,
                        context = False
                    ),
                    MultiLayerPerceptron(
                        self.hidden_dim,
                        [self.hidden_dim // 2, self.hidden_dim],
                        activation=None
                    ),  # node
                    MultiLayerPerceptron(
                        self.hidden_dim,
                        [self.hidden_dim // 2, self.hidden_dim],
                        activation=None
                    ),  # edge_attr
                ]))
            
        self.dec_joint_x = MultiLayerPerceptron(
            self.node_input_dim,
            [self.node_input_dim // 2, self.context_encoder.nhid],
            activation=None
        )

        self.dec_joint_e = MultiLayerPerceptron(
            self.edge_input_dim,
            [self.edge_input_dim // 2, self.context_encoder.enhid],
            activation=None
        )

        self.grad_dist_mlp = MultiLayerPerceptron(
            2 * self.hidden_dim,
            [self.hidden_dim, self.hidden_dim // 2, 1],
            activation=config.mlp_act
        )

        self.grad_node_mlp = MultiLayerPerceptron(
            1 * self.hidden_dim,
            [self.hidden_dim, self.node_input_dim, self.origin_node_in_dim],
            activation=config.mlp_act
        )

        self.grad_edge_attr_mlp = MultiLayerPerceptron(
            1 * self.hidden_dim,
            [self.hidden_dim, self.edge_input_dim, self.origin_edge_in_dim],
            activation=config.mlp_act
        )
        
        betas = get_beta_schedule(
            beta_schedule=config.beta_schedule,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            num_diffusion_timesteps=config.num_diffusion_timesteps,
        )

        # define alpha beta for diffusion 
        betas = torch.from_numpy(betas).float()
        self.betas = nn.Parameter(betas, requires_grad=False)
        alphas = (1. - betas).cumprod(dim=0)
        self.alphas = nn.Parameter(alphas, requires_grad=False)
        self.num_timesteps = self.betas.size(0)

        self.theta = config.theta
        self.node_freqs = nn.Parameter(1.0 / (self.theta ** (torch.arange(0, self.origin_node_in_dim, 2)[: (self.origin_node_in_dim // 2)].float() / self.origin_node_in_dim)), requires_grad=False).cuda()
        self.edge_freqs = nn.Parameter(1.0 / (self.theta ** (torch.arange(0, self.origin_edge_in_dim, 2)[: (self.origin_edge_in_dim // 2)].float() / self.origin_edge_in_dim)), requires_grad=False).cuda()
        self.norm_emb_x = nn.LayerNorm(self.hidden_dim)
        self.norm_emb_e = nn.LayerNorm(self.hidden_dim)
        self.device = self.node_freqs.device
    def reparameterize(self, mean, logvar):
        std = torch.exp(logvar / 2)
        epsilon = torch.randn_like(std)
        return epsilon * std + mean

    def mask_noise(self,batch_node,batch_edge,x,pos,bond_index,bond_attr, alpha_G, alpha_pos, alpha_dec_1_pos, alpha_edge, alpha_dec_1_edge, x_mask,edge_mask,if_noise=False, mode='direct'):
                       
        # # print(if_noise)
        if if_noise:
            # # print(pos)
            pos_noise = torch.zeros_like(pos)
            pos_noise.normal_()
            pos_noise[x_mask]=0
            pos_alpha_T = center_pos(pos_noise, batch_node)*(1.0 - alpha_pos).sqrt() / alpha_pos.sqrt()
            pos_alpha_T_1 = center_pos(pos_noise, batch_node)*(1.0 - alpha_dec_1_pos).sqrt() / alpha_dec_1_pos.sqrt()
            pos_perturb = pos + pos_alpha_T

        else:
            pos_perturb = pos
        # pos_perturbed = pos_perturbed[~x_mask]
        # pos_perturb[x_mask]=0

        # perterb x 
        if if_noise:
            x_noise = torch.zeros_like(x)
            x_noise[x_mask] = x_noise[x_mask].normal_()
            # print('x_noise:',x_noise)
            x_perterb = alpha_pos.sqrt() * x + (1.0-alpha_pos).sqrt()*x_noise
            x_perterb_T = x_perterb
            x_perterb_T_1 = alpha_dec_1_pos.sqrt() * x + (1.0-alpha_dec_1_pos).sqrt()*x_noise
            
        else:
            x_perterb = x

        # perterb edge_mask 
        if if_noise:     
            bond_noise = torch.zeros_like(bond_attr)
            bond_noise[edge_mask] = bond_noise[edge_mask].normal_()
            bond_attr_perterb = alpha_edge.sqrt() * bond_attr + (1.0-alpha_edge).sqrt()*bond_noise 
            bond_attr_perterb_T = bond_attr_perterb
            bond_attr_perterb_T_1 = alpha_dec_1_edge.sqrt() * bond_attr + (1.0-alpha_dec_1_edge).sqrt()*bond_noise 
        else:
            bond_attr_perterb = bond_attr


        if if_noise:
            if mode=='direct':
                return x_perterb,pos_perturb,bond_index,bond_attr_perterb,(x_noise,bond_noise,pos_noise)
            elif mode=='step':
                return x_perterb,pos_perturb,bond_index,bond_attr_perterb,(x_perterb_T_1-x_perterb_T,bond_attr_perterb_T_1-bond_attr_perterb_T,pos_alpha_T_1-pos_alpha_T)
        else:
            return x_perterb,pos_perturb,bond_index,bond_attr_perterb


    def encoder(self,batch_node,batch_edge,enc_data,x_mask_e,edge_mask_e,is_finetune=False):
        # perterb pos 
        x,pos,bond_index,bond_attr = enc_data
        
        x_e,pos_e,edge_index_e,edge_attr_e = x,pos,bond_index,bond_attr 
        x_e = self.x_embedding(x_e)
        edge_attr_e = self.edge_attr_embedding(edge_attr_e)
        if self.vae_context:
            if not is_finetune:
                if x_mask_e is not None:
                    m_x, log_var_x, x_List, m_e, log_var_e, edge_attr_List = self.context_encoder(
                        x=x_e,
                        edge_index_=edge_index_e,
                        edge_attr_=edge_attr_e, 
                        batch=batch_node,
                        x_mask=x_mask_e,
                        edge_mask=edge_mask_e
                    )
                else:
                    m_x, log_var_x, x_List, m_e, log_var_e, edge_attr_List = self.context_encoder(
                        x=x_e,
                        edge_index_=edge_index_e,
                        edge_attr_=edge_attr_e, 
                        batch=batch_node
                    )
                
                ctx_x = self.reparameterize(m_x,log_var_x)
                x_e = ctx_x

                

                tmp_kl_x = torch.exp(log_var_x) + m_x ** 2 - 1. - log_var_x
                kl_loss_x = 0.5 * torch.mean(tmp_kl_x)

                ctx_e = self.reparameterize(m_e,log_var_e)
                edge_attr_e = ctx_e

                tmp_kl_e = torch.exp(log_var_e) + m_e ** 2 - 1. - log_var_e
                # kl_loss_e = 0.5 * torch.sum(tmp_kl_e[edge_mask])                kl_loss_e = 0.5 * torch.mean(tmp_kl_e)
                return (x_e,pos_e,edge_index_e,edge_attr_e,x_List,edge_attr_List),kl_loss_x+kl_loss_e
                
            else:
                m_x, log_var_x, m_e, log_var_e = self.context_encoder(
                    x=x_e,
                    edge_index_=edge_index_e,
                    edge_attr_=edge_attr_e, 
                    batch=batch_node,
                    is_eval=True
                )
                ctx_x = self.reparameterize(m_x,log_var_x)
                ctx_e = self.reparameterize(m_e,log_var_e)
                x_e = ctx_x
                edge_attr_e = ctx_e

                kl_loss_x = 0 
                kl_loss_e = 0 
                return (x_e,pos_e,edge_index_e,edge_attr_e),kl_loss_x+kl_loss_e

           
    def decoder(self, batch_node, batch_edge, enc_data, dec_data, alpha_G, alpha_pos, alpha_dec_1_pos, alpha_edge, alpha_dec_1_edge, x_mask_d,edge_mask_d,time_step, mode_1='step'):
        '''
            Time embedding for node and edge 
        '''
        x_e,pos_e,edge_index_e,edge_attr_e,x_e_List,edge_attr_e_List = enc_data
        # print('x_e_List:',len(x_e_List))
        x_init,pos_init,bond_index,bond_attr = dec_data
        
        edge_mask_e = ~edge_mask_d 
        edge_vis_d = edge_mask_e
        edge_vis_e = edge_mask_d

        x_mask_e = ~x_mask_d
        x_vis_d = x_mask_e
        x_vis_e = x_mask_d

        batch_node_e = batch_node[x_vis_e] 
        batch_node_d = batch_node[x_vis_d] 

        batch_edge_e = batch_edge[edge_vis_e] 
        batch_edge_d = batch_edge[edge_vis_d] 
        
        with torch.no_grad():
            attn_mask_x = ~((batch_node.repeat(len(batch_node),1)).T)==(batch_node.repeat(len(batch_node),1))
            attn_mask_x_de = ~((batch_node_d.repeat(len(batch_node_e),1).T)==batch_node_e.repeat(len(batch_node_d),1))
            attn_mask_e = ~((batch_edge.repeat(len(batch_edge),1)).T)==(batch_edge.repeat(len(batch_edge),1))
            attn_mask_e_de = ~((batch_edge_d.repeat(len(batch_edge_e),1).T)==batch_edge_e.repeat(len(batch_edge_d),1))

        # training
        x_d_ = x_init
        pos_d = pos_init
        edge_index_d = bond_index
        edge_attr_d_ = bond_attr


        x_d_ = self.x_embedding(x_d_)
        edge_attr_d_ = self.edge_attr_embedding(edge_attr_d_)
        
        if 'joint' in self.dec_mode or 'coatt' in self.dec_mode:
            x_d = self.dec_joint_x(x_d_)
            edge_attr_d = self.dec_joint_e(edge_attr_d_)

        if self.time_emb:
            time_emb_activation = nn.ReLU()
            temb = get_timestep_embedding(time_step,self.hidden_dim)
            temb = self.temb.dense[0](temb)
            temb = time_emb_activation(temb)
            temb = self.temb.dense[1](temb)
            
            temb = self.temb_proj(time_emb_activation(temb))
        
        if 'coatt' in self.dec_mode:
            
            for i in range(self.all_num_layers-1,-1,-1):
                x_d = x_d + temb.index_select(0, batch_node)
                edge_attr_d = edge_attr_d + temb.index_select(0,batch_edge)
                if 'joint' in self.dec_mode and i==self.all_num_layers-1:
                    x_d_tmp=x_d.clone()
                    x_d_out = torch.zeros((x_d.shape[0],self.hidden_dim),device=x_d.device)
                    edge_attr_d_tmp = edge_attr_d.clone()
                    edge_attr_d_out = torch.zeros((edge_attr_d.shape[0],self.hidden_dim),device=edge_attr_d.device)

                    x_d_tmp[x_vis_e] = x_e[x_vis_e]
                    edge_attr_d_tmp[edge_vis_e] = edge_attr_e
                    
                    x_d_out = self.joint_decoder_x(x_d_tmp.unsqueeze(0), attn_mask = attn_mask_x).squeeze(0)
                    edge_attr_d_out = self.joint_decoder_e(edge_attr_d_tmp.unsqueeze(0), attn_mask = attn_mask_e).squeeze(0)
                    x_d = x_d_out
                    edge_attr_d = edge_attr_d_out
                else:
                    x_e_i =  x_e_List[i][x_vis_e]
                    x_d_i = x_d[x_vis_d]
                    edge_attr_e_i = edge_attr_e_List[i]
                    edge_attr_d_i = edge_attr_d[edge_vis_d]
                    x_d[x_vis_d] = self.dec_blks[i][0](x_d_i.unsqueeze(1),x_e_i.unsqueeze(1),x_e_i.unsqueeze(1),attn_mask=attn_mask_x_de).squeeze(1)
                    edge_attr_d[edge_vis_d] = edge_attr_d[edge_vis_d]+self.dec_blks[i][1](edge_attr_d_i.unsqueeze(1),edge_attr_e_i.unsqueeze(1),edge_attr_e_i.unsqueeze(1),attn_mask=attn_mask_e_de).squeeze(1)

                    x_d_tmp = x_d.clone()
                    edge_attr_d_tmp=edge_attr_d.clone()
                    x_d_tmp[x_vis_e] = x_e_List[i][x_vis_e] # add together
                    edge_attr_d_tmp[edge_vis_e] = edge_attr_e_List[i]
                    if i == self.all_num_layers-2:
                        first_layer=True
                    else:
                        first_layer=False
                    x_d, edge_index_d, edge_attr_d = self.dec_blks[i][2](
                        x=x_d_tmp,
                        edge_index_=edge_index_d,
                        edge_attr_=edge_attr_d_tmp,
                        batch=batch_node,
                        first_layer=first_layer
                    )
                    x_d = self.dec_blks[i][3](x_d)
                    edge_attr_d = self.dec_blks[i][4](edge_attr_d)

            # x_noise_target,edge_noise_target,pos_noise_target = noises
            pair_d = assemble_pair_feature(
                node_attr=x_d,
                edge_index=edge_index_d,
                edge_attr=edge_attr_d,
            )
            
            node_noise_pred = self.grad_node_mlp(x_d[x_vis_d])
            # mode direct:
            # edge_attr_noise_pred = self.grad_edge_attr_mlp(edge_attr_d[edge_vis_d])
            # mode step:
            edge_attr_noise_pred = self.grad_edge_attr_mlp(edge_attr_d[edge_vis_d])


            if mode_1=='direct':
                loss_node = (node_noise_pred - x_init[x_vis_d]) ** 2 # predict origin directly
                loss_edge = (edge_attr_noise_pred - bond_attr[edge_vis_d]) ** 2
            elif mode_1=='step':
                loss_node = (node_noise_pred - x_noise_target[x_vis_d]) ** 2 # predict step
                loss_edge = (edge_attr_noise_pred - edge_noise_target[edge_vis_d]) ** 2


            if mode_1=='step':
                # return loss_node.sum(), loss_edge.sum(), node_noise_pred, edge_attr_noise_pred
                return node_noise_pred, edge_attr_noise_pred
            elif mode_1=='direct':
                return loss_node.mean(), loss_edge.mean()
    
    # @staticmethod
    def precompute_freqs_cis(self,dim, seq_len, mode='node'):
        with torch.no_grad():

        
            t_List = []
            for li in seq_len:
                t_List.append(torch.arange(int(li)).to(self.device))
            t = torch.cat(t_List)
        # freqs.shape = [seq_len, dim // 2] 
            # print('t,node_freqs',t.shape,self.node_freqs.shape)
            if mode=='node':
                freqs = torch.outer(t, self.node_freqs).float()  # 计算m * \theta
            else:
                freqs = torch.outer(t, self.edge_freqs).float()
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs) # 454*256

        return freqs_cis
    # @staticmethod
    def apply_rotary_emb(self,
        x: torch.Tensor, freqs_cis: torch.Tensor):
        # x.shape = [seq_len, dim]
        # x_.shape = [seq_len, dim // 2, 2]
        x_ = x.float().reshape(*x.shape[:-1], -1, 2)
        x_ = torch.view_as_complex(x_)
        
        # xq_out.shape = [batch_size, seq_len, dim]
        # print('x_,freqs_cis',x_.shape,freqs_cis.shape)
        # print(x_.shape, freqs_cis.shape)
        x_out = torch.view_as_real(x_ * freqs_cis)
        x_out = x_out.flatten(1)
        return x_out.type_as(x)

    def sdrf_rewrite(self,):
        pass
    
    def forward_eval(self, databatch, context=None, return_unreduced_loss=False, return_unreduced_edge_loss=False,
                extend_order=True, extend_radius=True, is_sidechain=None,mode_1='step'):
        # x_init = self.x_embedding(databatch.x.float())
        x_init = databatch.x.float()
        pos_init = databatch.pos/256
        bond_index = databatch.edge_index.long()
        bond_attr = databatch.edge_attr.float()

        batch_node = databatch.batch
        batch_edge = batch_node.index_select(0,bond_index[0])

        num_graphs = batch_node[-1]+1
        node_seq_List = [(batch_node==bi).sum() for bi in range(num_graphs)]
        
        freqs_cis_node = self.precompute_freqs_cis(self.origin_node_in_dim,node_seq_List,mode='node').to(x_init.device)
        
        x_init = self.apply_rotary_emb(x_init,freqs_cis_node)
        edge_seq_List = [(batch_edge==bi).sum() for bi in range(num_graphs)]
        freqs_cis_edge = self.precompute_freqs_cis(self.origin_edge_in_dim,edge_seq_List,mode='edge').to(x_init.device)
        bond_attr = self.apply_rotary_emb(bond_attr,freqs_cis_edge)

        num_nodes = len(x_init)
        perm_node = torch.randperm(num_nodes,device=x_init.device)

        alpha = None
        alpha_pos = None
        alpha_edge = None
        x_mask = None
        edge_mask = None
        enc_data = (x_init,pos_init,bond_index,bond_attr)
        # enc_data, kl_loss = self.encoder(batch_node,batch_edge,enc_data,alpha,alpha_pos,alpha_edge,x_mask,edge_mask,is_finetune=True)
        enc_data, kl_loss = self.encoder(batch_node,batch_edge,enc_data,x_mask,edge_mask,is_finetune=True)
        
        return enc_data, batch_node, batch_edge


    def forward(self, databatch, perm_node, perm_edge, time_number=1000, context=None, return_unreduced_loss=False, return_unreduced_edge_loss=False,
                extend_order=True, extend_radius=True, is_sidechain=None, mode_1='step'):
        
        # with autocast(enabled=False):
            # x_init = self.x_embedding(self.norm_ori_x(databatch.x.float()))
        x_init = databatch.x.float()
        
        pos_init = databatch.pos/256
        bond_index = databatch.edge_index.long()
        bond_attr = databatch.edge_attr.float()
        batch_node = databatch.batch
        batch_edge = batch_node.index_select(0,bond_index[0])

        num_graphs = batch_node[-1]+1
        node_seq_List = [(batch_node==bi).sum() for bi in range(num_graphs)]
        freqs_cis_node = self.precompute_freqs_cis(self.origin_node_in_dim,node_seq_List,mode='node').to(x_init.device)
        # print(x_init.shape,freqs_cis_node.shape)
        x_init = self.apply_rotary_emb(x_init,freqs_cis_node)
        edge_seq_List = [(batch_edge==bi).sum() for bi in range(num_graphs)]
        freqs_cis_edge = self.precompute_freqs_cis(self.origin_edge_in_dim,edge_seq_List,mode='edge').to(x_init.device)
        bond_attr = self.apply_rotary_emb(bond_attr,freqs_cis_edge)

        # print(x_init.shape,bond_attr.shape)
        # freqs_cis_node.device = x_init.device
        
        # freqs_cis_edge.device = bond_attr.device

        # sample time step 
        # time_step = torch.randint(0,self.num_timesteps,size=(torch.div(num_graphs, 2, rounding_mode='trunc')+1,),device=pos_init.device)
        time_step = torch.ones((torch.div(num_graphs, 2, rounding_mode='trunc')+1,),device=pos_init.device)*time_number
        time_step = torch.cat(
            [time_step, self.num_timesteps - time_step - 1], dim=0)[:num_graphs]
        time_step = time_step.long()
        alpha = self.alphas.index_select(0, time_step) # (G,)
        
        time_step_dec_1 = time_step-1
        time_step_dec_1[time_step_dec_1<0]=0
        alpha_dec_1 = self.alphas.index_select(0, time_step_dec_1)

        alpha_pos = alpha.index_select(0, batch_node).unsqueeze(-1) # (N, 1)
        alpha_dec_1_pos = alpha_dec_1.index_select(0, batch_node).unsqueeze(-1) # (N, 1)

        alpha_edge = alpha.index_select(0, batch_edge).unsqueeze(-1) # (E, 1)
        alpha_dec_1_edge = alpha_dec_1.index_select(0, batch_edge).unsqueeze(-1) # (E, 1)

        # generate node mask for encoder
        num_nodes = len(x_init)
        # perm_node = torch.randperm(num_nodes,device=x_init.device)
        num_mask_nodes = int(self.node_mask_ratio*num_nodes)
        mask_nodes = perm_node[: num_mask_nodes]
        x_mask = torch.zeros((num_nodes),device=x_init.device).bool()
        x_mask[mask_nodes] = 1

        # generate edge mask for encoder
        num_edges = len(bond_attr)
        # perm_edge = torch.randperm(num_edges,device=bond_attr.device)
        num_mask_edges = int(self.edge_mask_ratio*num_edges)
        mask_edges = perm_edge[: num_mask_edges]
        edge_mask = torch.zeros((num_edges),device=bond_index.device).bool()
        edge_mask[mask_edges]=1 # chosen to mask to 0

        enc_data = (x_init,pos_init,bond_index,bond_attr)
        enc_data, kl_loss = self.encoder(batch_node,batch_edge,enc_data,x_mask,edge_mask)
        dec_data = (x_init,pos_init,bond_index,bond_attr)
                    
        # node_loss, edge_loss = self.decoder(batch_node,batch_edge,enc_data,dec_data, alpha, alpha_pos, alpha_dec_1_pos, alpha_edge, alpha_dec_1_edge, ~x_mask, ~edge_mask,time_step, mode_1=mode_1)
        node_noise_pred, edge_attr_noise_pred = self.decoder(batch_node,batch_edge,enc_data,dec_data, alpha, alpha_pos, alpha_dec_1_pos, alpha_edge, alpha_dec_1_edge, ~x_mask, ~edge_mask,time_step, mode_1=mode_1)
            
        # return kl_loss, node_loss, edge_loss
        return node_noise_pred, edge_attr_noise_pred
        
    def sample(self, databatch, perm_node, perm_edge, time_number=1000, context=None, return_unreduced_loss=False, return_unreduced_edge_loss=False,
                extend_order=True, extend_radius=True, is_sidechain=None, mode_1='step'):   
        x_init = databatch.x.float()
        
        pos_init = databatch.pos/256
        bond_index = databatch.edge_index.long()
        bond_attr = databatch.edge_attr.float()
        batch_node = databatch.batch
        batch_edge = batch_node.index_select(0,bond_index[0])

        num_graphs = batch_node[-1]+1
        node_seq_List = [(batch_node==bi).sum() for bi in range(num_graphs)]
        freqs_cis_node = self.precompute_freqs_cis(self.origin_node_in_dim,node_seq_List,mode='node').to(self.device)
        # print(x_init.shape,freqs_cis_node.shape)
        x_init = self.apply_rotary_emb(x_init,freqs_cis_node)
        edge_seq_List = [(batch_edge==bi).sum() for bi in range(num_graphs)]
        freqs_cis_edge = self.precompute_freqs_cis(self.origin_edge_in_dim,edge_seq_List,mode='edge').to(self.device)
        bond_attr = self.apply_rotary_emb(bond_attr,freqs_cis_edge)

        # print(x_init.shape,bond_attr.shape)
        # freqs_cis_node.device = x_init.device
        
        # freqs_cis_edge.device = bond_attr.device

        # sample time step 
        # time_step = torch.randint(0,self.num_timesteps,size=(torch.div(num_graphs, 2, rounding_mode='trunc')+1,),device=pos_init.device)
        time_step = torch.ones((torch.div(num_graphs, 2, rounding_mode='trunc')+1,),device=pos_init.device)*time_number
        time_step = torch.cat(
            [time_step, self.num_timesteps - time_step - 1], dim=0)[:num_graphs]
        time_step = time_step.long()
        alpha = self.alphas.index_select(0, time_step) # (G,)
        time_step_dec_1 = time_step-1
        time_step_dec_1[time_step_dec_1<0]=0
        alpha_dec_1 = self.alphas.index_select(0, time_step_dec_1)

        alpha_pos = alpha.index_select(0, batch_node).unsqueeze(-1) # (N, 1)
        alpha_dec_1_pos = alpha_dec_1.index_select(0, batch_node).unsqueeze(-1) # (N, 1)

        alpha_edge = alpha.index_select(0, batch_edge).unsqueeze(-1) # (E, 1)
        alpha_dec_1_edge = alpha_dec_1.index_select(0, batch_edge).unsqueeze(-1) # (E, 1)

        # generate node mask for encoder
        num_nodes = len(x_init)
        # perm_node = torch.randperm(num_nodes,device=x_init.device)
        num_mask_nodes = int(self.node_mask_ratio*num_nodes)
        mask_nodes = perm_node[: num_mask_nodes]
        x_mask = torch.zeros((num_nodes),device=x_init.device).bool()
        x_mask[mask_nodes] = 1

        # generate edge mask for encoder
        num_edges = len(bond_attr)
        # perm_edge = torch.randperm(num_edges,device=bond_attr.device)
        num_mask_edges = int(self.edge_mask_ratio*num_edges)
        mask_edges = perm_edge[: num_mask_edges]
        edge_mask = torch.zeros((num_edges),device=bond_index.device).bool()
        edge_mask[mask_edges]=1 # chosen to mask to 0
        
        x_mask_d = ~x_mask 
        edge_mask_d = ~edge_mask

        edge_mask_e = ~edge_mask_d 
        edge_vis_d = edge_mask_e
        edge_vis_e = edge_mask_d

        x_mask_e = ~x_mask_d
        x_vis_d = x_mask_e
        x_vis_e = x_mask_d
        
        x_d_,pos_d,edge_index_d,edge_attr_d_,noises = self.mask_noise(batch_node,batch_edge,x_init,pos_init,bond_index,bond_attr,alpha, alpha_pos, alpha_dec_1_pos, alpha_edge, alpha_dec_1_edge, x_vis_d, edge_vis_d,if_noise=True, mode=mode_1)
        return time_step, x_d_[x_vis_d], edge_attr_d_[edge_vis_d], x_vis_d, edge_vis_d, noises
def center_pos(pos,batch):
    pos_center = pos - scatter_mean(pos,batch,dim=0)[batch]
    return pos_center


def dropout_edge(edge_index: Tensor, p: float = 0.5,
                 force_undirected: bool = False,
                 training: bool = True) -> Tuple[Tensor, Tensor]:
    r"""Randomly drops edges from the adjacency matrix
    :obj:`edge_index` with probability :obj:`p` using samples from
    a Bernoulli distribution.

    The method returns (1) the retained :obj:`edge_index`, (2) the edge mask
    or index indicating which edges were retained, depending on the argument
    :obj:`force_undirected`.

    Args:
        edge_index (LongTensor): The edge indices.
        p (float, optional): Dropout probability. (default: :obj:`0.5`)
        force_undirected (bool, optional): If set to :obj:`True`, will either
            drop or keep both edges of an undirected edge.
            (default: :obj:`False`)
        training (bool, optional): If set to :obj:`False`, this operation is a
            no-op. (default: :obj:`True`)

    :rtype: (:class:`LongTensor`, :class:`BoolTensor` or :class:`LongTensor`)

    Examples:

        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
        ...                            [1, 0, 2, 1, 3, 2]])
        >>> edge_index, edge_mask = dropout_edge(edge_index)
        >>> edge_index
        tensor([[0, 1, 2, 2],
                [1, 2, 1, 3]])
        >>> edge_mask # masks indicating which edges are retained
        tensor([ True, False,  True,  True,  True, False])

        >>> edge_index, edge_id = dropout_edge(edge_index,
        ...                                    force_undirected=True)
        >>> edge_index
        tensor([[0, 1, 2, 1, 2, 3],
                [1, 2, 3, 0, 1, 2]])
        >>> edge_id # indices indicating which edges are retained
        tensor([0, 2, 4, 0, 2, 4])
    """
    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    if not training or p == 0.0:
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index, edge_mask

    row, col = edge_index

    edge_mask = torch.rand(row.size(0), device=edge_index.device) >= p

    if force_undirected:
        edge_mask[row > col] = False

    edge_index = edge_index[:, edge_mask]

    if force_undirected:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_mask = edge_mask.nonzero().repeat((2, 1)).squeeze()

    return edge_index, edge_mask