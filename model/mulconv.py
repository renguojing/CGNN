import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv.gcn_conv import gcn_norm

class MulConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, normalize: bool = False, bias: bool = False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.lin1 = Linear(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')
        self.lin2 = Linear(in_channels, out_channels, bias=bias,
                          weight_initializer='glorot')
        self.lin3 = Linear(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')
        self.normalize = normalize
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

    def forward(self, x, edge_index, x1=None, edge_weight=None):
        edge_index, edge_weight = gcn_norm(edge_index, edge_weight, x.size(self.node_dim), add_self_loops=False)
        # if edge_weight is None:
        #     edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
        out = self.lin2(out)
        
        out += self.lin1(x)

        if x1 is not None:
            a = (x * x1).sum(dim=-1)
            a = torch.sigmoid(a)
            # print(a.shape, x1.shape, self.lin3(x1).shape)
            out += self.lin3(x1) * a.unsqueeze(-1)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)
        return out

    def message(self, x_j, edge_weight):
        # print('message')
        # print('x_j:', x_j)
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
    
    # def message_and_aggregate(self, adj_t, x):
    #     # print('message_and_aggregate')
    #     return matmul(adj_t, x, reduce=self.aggr)