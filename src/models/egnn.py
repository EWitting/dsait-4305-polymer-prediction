import torch
import numpy as np
import torch_geometric as pyg
import torch_geometric.nn.aggr as agg

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
        norm_feats=False,
        norm_coors=False,
        aggr='sum', 
        dropout=0.1, 
        edge_mlp_dim=128, 
        node_mlp_dim=128, 
        coors_mlp_dim=32, 
        act=nn.SiLU, 
        *, 
        aggr_kwargs = None, 
        flow = "source_to_target", 
        node_dim = -2, 
        decomposed_layers = 1
    ):
        super().__init__(aggr, aggr_kwargs=aggr_kwargs, flow=flow, node_dim=node_dim, decomposed_layers=decomposed_layers)
        
        self.act = act
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
        m_ij = self.edge_mlp(torch.cat([x_i, x_j, edge_attr.float()], dim=-1))
        return m_ij
    
    def propagate(self, edge_index, size=None, **kwargs):
        size = self._check_input(edge_index, size)
        coll_dict = self._collect(self._user_args, edge_index, size, kwargs)
        msg_kwargs = self.inspector.collect_param_data('message', coll_dict)
        aggr_kwargs = self.inspector.collect_param_data('aggregate', coll_dict)
        update_kwargs = self.inspector.collect_param_data('update', coll_dict)
        
        m_ij = self.message(**msg_kwargs)

        coors_out = kwargs["coors"]
        
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
        edge_attr_dim = 1, 
        m_dim=32,
        embedding_nums=list([119, 11, 12, 8, 2, 9, 2, 9, 5, 7]), 
        embedding_idxs=list([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        embedding_dims=list([32, 8, 8, 8, 8, 8, 8, 8, 8, 8]),
        norm_feats=True, 
        dropout=0.,
        act=nn.SiLU,
        aggr="sum",
        pooling=pyg.nn.global_mean_pool,
        use_embeddings=True,
        *args,
        **kwargs
    ):
        super(EGNNNetwork, self).__init__(*args, **kwargs)
        self.n_layers = n_layers
        self.emb_idxs = embedding_idxs
        self.nf_dim = nf_dim
        self.nf_dim_emb = nf_dim
        self.use_embeddings = use_embeddings
        
        # handle embeddings 
        if self.use_embeddings:
            self.emb_layers = nn.ModuleList()
            for emb_num, emb_dim in zip(embedding_nums, embedding_dims):
                self.emb_layers.append(nn.Embedding(emb_num, emb_dim))
                self.nf_dim_emb += emb_dim - 1
        
        self.mp_layers = nn.ModuleList()
        self.edge_attr_dim = edge_attr_dim
        self.norm_feats = norm_feats
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
                act=act,
                norm_feats=norm_feats
            )
            self.mp_layers.append(layer)
            
        all_idxs = torch.arange(self.nf_dim)
        self.keep_idxs = torch.tensor([i for i in all_idxs if i not in self.emb_idxs])
        
    def forward(self, data: Data):
        # get the embeddings 
        x, edge_index, edge_attr, coors, batch = data.x, data.edge_index, data.edge_attr.float(), data.pos.float(), data.batch
        x = self.embed(x) if self.use_embeddings else x
        
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr, fill_value=0.0)
        for i, layer in enumerate(self.mp_layers):
            x, coors = layer(x, coors, edge_index, edge_attr, batch)
        
        return self.pooling(x, batch)
    
    def embed(self, x):
        embs = []
        for emb, i in zip(self.emb_layers, self.emb_idxs):
            embs.append(emb(x[:, i].long()))
        x = x[:, self.keep_idxs]
        return torch.cat([x, *embs], dim=-1)
    
class SingleGraphEGNN(nn.Module):
    
    def __init__(
        self, 
        n_layers=7, 
        nf_dim=24, 
        edge_attr_dim=1, 
        m_dim=32,
        embedding_nums=list([119, 11, 12, 8, 2, 9, 2, 9, 5, 7]), 
        embedding_idxs=list([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        embedding_dims=list([32, 8, 8, 8, 8, 8, 8, 8, 8, 8]),
        norm_feats=True, 
        dropout=0.2,
        act=nn.SiLU,
        aggr="sum", # actually for the forms in the paper this has to be sum
        pooling='global_mean', 
        global_linear_dims=list([128, 64]),
        use_descs=True,
        use_embeddings=True,
        global_final_dim=5,
        *args, 
        **kwargs
    ):
        super(SingleGraphEGNN, self).__init__(*args, **kwargs)
        
        match pooling:
            case 'global_mean':
                pooling = pyg.nn.global_mean_pool
            case 'global_max':
                pooling = pyg.nn.global_max_pool   
            case 'global_add':
                pooling = pyg.nn.global_add_pool    
                
        self.egnn = EGNNNetwork(
            n_layers=n_layers, 
            nf_dim=nf_dim, 
            edge_attr_dim=edge_attr_dim, 
            m_dim=m_dim,
            embedding_nums=embedding_nums, 
            embedding_idxs=embedding_idxs,
            embedding_dims=embedding_dims,
            norm_feats=norm_feats, 
            dropout=dropout,
            aggr=aggr,
            act=act,
            pooling=pooling,
            use_embeddings=use_embeddings
        )
        
        # for global graph info
        if use_descs:
            self.ffn = nn.Sequential(nn.Linear(self.egnn.nf_dim_emb + 217, global_linear_dims[0]))
        else:
            self.ffn = nn.Sequential(nn.Linear(self.egnn.nf_dim_emb, global_linear_dims[0]))
            
        for dim1, dim2 in zip(global_linear_dims[:-1], global_linear_dims[1:]):
            self.ffn.extend([nn.Linear(dim1, dim2), nn.LayerNorm(dim2), nn.ReLU()])
        
        self.ffn.append(nn.Linear(global_linear_dims[-1], global_final_dim))
        self.global_final_dim = global_final_dim 
        self.use_descs = use_descs
    
    def forward(self, input):
        if self.use_descs:
            data, graph_desc = input
        else:
            data = input
        egnn_res = self.egnn(data)
        if self.use_descs:
            full_inp = torch.cat([egnn_res, graph_desc], dim=-1)
        else:
            full_inp = egnn_res
        return self.ffn(full_inp)
    
    
class MultiGraphEGNN(nn.Module):
    
    def __init__(
        self, 
        n_layers=7, 
        nf_dim=48, 
        edge_attr_dim=2, 
        m_dim=32,
        embedding_nums=list([118, 8, 2, 2, 9]), 
        embedding_idxs=list([1, 4, 5, 7, 8]),
        embedding_dims=list([32, 16, 4, 4, 16]),
        update_coors=True, 
        update_feats=True, 
        norm_feats=True, 
        dropout=0.2,
        aggr="sum", # actually for the forms in the paper this has to be sum
        aggr_graphs="lstm", 
        lstm_kwargs=dict(),
        pooling='global_mean', 
        out_dim=128,
        global_linear_dims=list([128, 64]),
        global_final_dim=5,
        *args, 
        **kwargs
    ):
        super(MultiGraphEGNN, self).__init__(*args, **kwargs)
        
        match pooling:
            case 'global_mean':
                pooling = pyg.nn.global_mean_pool
            case 'global_max':
                pooling = pyg.nn.global_max_pool   
            case 'global_add':
                pooling = pyg.nn.global_add_pool       
                
        self.egnns = nn.ModuleList()
        for i in range(5):
            self.egnns.append(
                EGNNNetwork(
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
            )
        
        assert aggr_graphs in {"lstm", "sum", "concat", "max", "mean"}

        self.aggr_graphs = aggr_graphs
        self.intial_extractor = nn.Sequential()
        match aggr_graphs:
            case "lstm":
                lstm_out = lstm_kwargs['lstm_out']
                del lstm_kwargs['lstm_out']
                self.aggregator = agg.LSTMAggregation(
                    in_channels=nf_dim, 
                    out_channels=lstm_out
                    **lstm_kwargs
                ) # we can do this since we know the graph permutation
                self.intial_extractor.extend([
                    nn.Linear(lstm_kwargs['hidden_dim'], out_dim),
                    nn.ReLU()
                ])
            case "sum": 
                self.aggregator = agg.SumAggregation()
                self.intial_extractor.extend([
                    nn.Linear(self.egnn.nf_dim_emb, out_dim),
                    nn.ReLU()
                ])
            case "max": 
                self.aggregator = agg.MaxAggregation()
                self.intial_extractor.extend([
                    nn.Linear(self.egnn.nf_dim_emb, out_dim),
                    nn.ReLU()
                ])
            case "max": 
                self.aggregator = agg.MeanAggregation()
                self.intial_extractor.extend([
                    nn.Linear(self.egnn.nf_dim_emb, out_dim),
                    nn.ReLU()
                ])
            case "concat": 
                self.aggregator = None
                self.intial_extractor.extend([
                    nn.Linear(self.egnn.nf_dim_emb * 5, out_dim),
                    nn.ReLU()
                ])
        
        # for global graph info
        self.ffn = nn.Sequential()
        self.ffn.extend([nn.Linear(out_dim + 217, global_linear_dims[0]), nn.ReLU()])
        for dim1, dim2 in zip(global_linear_dims[:-1], global_linear_dims[1:]):
            self.ffn.extend([nn.LayerNorm(dim1), nn.Linear(dim1, dim2), nn.ReLU()])
        
        self.ffn.append(nn.Linear(global_linear_dims[-1], global_final_dim))
        self.global_final_dim = global_final_dim 
        
        def forward(self, data: tuple):
            descs = data[-1]
            