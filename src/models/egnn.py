import torch
import numpy as np
import torch_geometric as pyg

from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

# ideas borrowed from original EGNN paper and this repo: https://github.com/lucidrains/egnn-pytorch/tree/main/egnn_pytorch

class EGNNConv(MessagePassing):
    
    def __init__(
        self, 
        nf_dim, 
        ef_dim, 
        m_dim, 
        norm_feats = False,
        norm_coors = False,
        aggr = 'sum', 
        dropout = 0.1, 
        edge_mlp_dim=128, 
        node_mlp_dim=128, 
        coors_mlp_dim=32, 
        *, 
        aggr_kwargs = None, 
        flow = "source_to_target", 
        node_dim = -2, 
        decomposed_layers = 1
    ):
        super().__init__(aggr, aggr_kwargs=aggr_kwargs, flow=flow, node_dim=node_dim, decomposed_layers=decomposed_layers)
        
        assert aggr in {'add', 'sum', 'max', 'mean'}, 'pool method must be a valid option'
        self.act = nn.SiLU
        self.nf_dim = nf_dim
        self.ef_dim = ef_dim # concatenation of inputs
        self.m_dim = m_dim
        self.norm_feats = norm_feats
        self.norm_coors = norm_coors
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(self.ef_dim, edge_mlp_dim),
            self.dropout,
            self.act(),
            nn.Linear(edge_mlp_dim, m_dim),
            self.act()
        )
        
        self.node_mlp = nn.Sequential(
            nn.Linear(nf_dim + m_dim, node_mlp_dim),
            self.dropout,
            self.act(),
            nn.Linear(node_mlp_dim, nf_dim),
        )
        
        self.coors_mlp = nn.Sequential(
            nn.Linear(m_dim, coors_mlp_dim),
            self.dropout,
            self.act(),
            nn.Linear(coors_mlp_dim, 1)
        ) 
        
        
        self.node_norm = pyg.nn.norm.LayerNorm(nf_dim) if norm_feats else None
        
        self.apply(self.init_)
        
        
    def init_(self, module):
        if type(module) in {nn.Linear}:
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)
            
    def message(self, x_i, x_j, edge_attr):
        m_ij = self.edge_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))
        return m_ij
    
    def propagate(self, edge_index, size=None, **kwargs):
        size = self._check_input(edge_index, size)
        coll_dict = self._collect(self._user_args, edge_index, size, kwargs)
        msg_kwargs = self.inspector.collect_param_data('message', coll_dict)
        aggr_kwargs = self.inspector.collect_param_data('aggregate', coll_dict)
        update_kwargs = self.inspector.collect_param_data('update', coll_dict)
        
        m_ij = self.message(**msg_kwargs)
        coor_wij = self.coors_mlp(m_ij)
        mhat_i = self.aggregate(coor_wij * kwargs["rel_coors"], **aggr_kwargs)
        C = 1/(kwargs["coors"].size(0))
        coors_out = kwargs["coors"] + C * mhat_i
        
        m_i = self.aggregate(m_ij, **aggr_kwargs)
        hidden_feats = self.node_norm(kwargs["x"], kwargs["batch"]) if self.node_norm else kwargs["x"]
        hidden_out = self.node_mlp(torch.cat([hidden_feats, m_i], dim = -1))
        hidden_out = kwargs["x"] + hidden_out # residual
        
        return self.update((hidden_out, coors_out), **update_kwargs)
    
    def forward(self, feats, coors, edge_index, edge_attr, batch): 
        rel_coors = coors[edge_index[0]] - coors[edge_index[1]]
        rel_dist  = (rel_coors ** 2).sum(dim=-1, keepdim=True)
        
        edge_attr_feats = torch.cat([edge_attr, rel_dist], dim=-1)
        hidden_out, coors_out = self.propagate(edge_index, x=feats, edge_attr=edge_attr_feats,
                                                    coors=coors, rel_coors=rel_coors, 
                                                    batch=batch)
        
        return hidden_out, coors_out

class EGNNNetwork(nn.Module):
    
    def __init__(
        self, 
        n_layers, 
        nf_dim, 
        edge_attr_dim = 2, 
        m_dim=32,
        embedding_nums=list([40, 8, 2, 2, 9]), 
        embedding_idxs=list([1, 4, 5, 7, 8]),
        embedding_dims=list([32, 16, 4, 4, 16]),
        update_coors=True, 
        update_feats=True, 
        norm_feats=True, 
        dropout=0.,
        aggr="sum",
        pooling=pyg.nn.global_add_pool, 
        global_linear_dims=list([128, 64]),
        global_final_dim=64,
        *args,
        **kwargs
    ):
        super(EGNNNetwork, self).__init__(*args, **kwargs)
        self.n_layers = n_layers
        self.emb_idxs = embedding_idxs
        self.nf_dim = nf_dim
        self.nf_dim_emb = nf_dim
        
        # handle embeddings 
        self.emb_layers = nn.ModuleList()
        for emb_num, emb_dim in zip(embedding_nums, embedding_dims):
            self.emb_layers.append(nn.Embedding(emb_num, emb_dim))
            self.nf_dim_emb += emb_dim - 1
            
        self.mp_layers = nn.ModuleList()
        self.edge_attr_dim = edge_attr_dim
        self.norm_feats = norm_feats
        self.update_feats = update_feats
        self.update_coors = update_coors
        self.dropout = dropout
        self.pooling = pooling
        
        # for mpnn
        self.ef_dim = self.nf_dim_emb * 2 + (self.edge_attr_dim + 1)
        for i in range(n_layers):
            layer = EGNNConv(
                nf_dim=self.nf_dim_emb,
                ef_dim=self.ef_dim,
                m_dim=m_dim,
                dropout=dropout,
                aggr=aggr,
                norm_feats=norm_feats
            )
            self.mp_layers.append(layer)
        
        
    def forward(self, data: Data):
        # get the embeddings 
        x, edge_index, edge_attr, coors, batch = data.x, data.edge_index, data.edge_attr, data.pos, data.batch
        x = self.embed(x)
        
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr, fill_value=0.0)
        for i, layer in enumerate(self.mp_layers):
            x, coors = layer(x, coors, edge_index, edge_attr, batch)
        
        return self.pooling(x, batch)
    
    def embed(self, x):
        x_cat = x[:, self.emb_idxs]
        embs = []
        for i, emb in enumerate(self.emb_layers):
            embs.append(emb(x_cat[:, i].long()))
        mask = torch.ones_like(x, dtype=torch.bool)   
        mask[:, self.emb_idxs] = False
        col_mask = mask.any(dim=0)
        x = x[:, col_mask]
        return torch.cat([x, *embs], dim=-1)
    
class SingleGraphEGNN(nn.Module):
    
    def __init__(
        self, 
        n_layers=7, 
        nf_dim=48, 
        edge_attr_dim=2, 
        m_dim=32,
        embedding_nums=list([40, 8, 2, 2, 9]), 
        embedding_idxs=list([1, 4, 5, 7, 8]),
        embedding_dims=list([32, 16, 4, 4, 16]),
        update_coors=True, 
        update_feats=True, 
        norm_feats=True, 
        dropout=0.2,
        aggr="sum", # actually for the forms in the paper this has to be sum
        pooling=pyg.nn.global_add_pool, 
        global_linear_dims=list([128, 64]),
        global_final_dim=5,
        *args, 
        **kwargs
    ):
        super(SingleGraphEGNN, self).__init__(*args, **kwargs)
        self.egnn = EGNNNetwork(
            n_layers=n_layers, 
            nf_dim=nf_dim, 
            edge_attr_dim=edge_attr_dim, 
            m_dim=m_dim,
            embedding_nums=embedding_nums, 
            embedding_idxs=embedding_idxs,
            embedding_dims=embedding_dims,
            update_coors=update_coors, 
            update_feats=update_feats, 
            norm_feats=norm_feats, 
            dropout=dropout,
            aggr=aggr,
            pooling=pooling,
        )
        
        # for global graph info
        self.ffn = nn.Sequential(nn.Linear(self.egnn.nf_dim_emb + 217, global_linear_dims[0]))
        for dim1, dim2 in zip(global_linear_dims[:-1], global_linear_dims[1:]):
            self.ffn.extend([nn.Linear(dim1, dim2), nn.ReLU()])
        
        self.ffn.append(nn.Linear(global_linear_dims[-1], global_final_dim))
        self.global_final_dim = global_final_dim 
        
    
    def forward(self, data, graph_desc):
        egnn_res = self.egnn(data)
        full_inp = torch.cat([egnn_res, graph_desc], dim=-1)
        return self.ffn(full_inp)
    
    
class MultiGraphEGNN(nn.Module):
    
    def __init__(
        self, 
        n_layers=7, 
        nf_dim=48, 
        edge_attr_dim=2, 
        m_dim=32,
        embedding_nums=list([40, 8, 2, 2, 9]), 
        embedding_idxs=list([1, 4, 5, 7, 8]),
        embedding_dims=list([32, 16, 4, 4, 16]),
        update_coors=True, 
        update_feats=True, 
        norm_feats=True, 
        dropout=0.2,
        aggr="sum", # actually for the forms in the paper this has to be sum
        aggr_graphs="lstm", 
        pooling=pyg.nn.global_add_pool, 
        global_linear_dims=list([128, 64]),
        global_final_dim=5,
        *args, 
        **kwargs
    ):
        super(SingleGraphEGNN, self).__init__(*args, **kwargs)
        self.egnn = EGNNNetwork(
            n_layers=n_layers, 
            nf_dim=nf_dim, 
            edge_attr_dim=edge_attr_dim, 
            m_dim=m_dim,
            embedding_nums=embedding_nums, 
            embedding_idxs=embedding_idxs,
            embedding_dims=embedding_dims,
            update_coors=update_coors, 
            update_feats=update_feats, 
            norm_feats=norm_feats, 
            dropout=dropout,
            aggr=aggr,
            pooling=pooling,
        )
        
        assert aggr_graphs in {"lstm", "sum", "concat", "max"}
        
        
        # for global graph info
        self.ffn = nn.Sequential(nn.Linear(self.egnn.nf_dim_emb + 217, global_linear_dims[0]))
        for dim1, dim2 in zip(global_linear_dims[:-1], global_linear_dims[1:]):
            self.ffn.extend([nn.Linear(dim1, dim2), nn.ReLU()])
        
        self.ffn.append(nn.Linear(global_linear_dims[-1], global_final_dim))
        self.global_final_dim = global_final_dim 