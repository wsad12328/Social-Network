import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, MessagePassing, GraphNorm

class NeighborhoodAggregation(MessagePassing):
    def __init__(self):
        super(NeighborhoodAggregation, self).__init__(aggr='add')
    
    def forward(self, x, edge_index, norm):
        return self.propagate(edge_index, x=x, norm=norm)
    
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

class DrBC(nn.Module):
    def __init__(self, dim, num_layers, aggregate_type='sum', heads=4):
        super(DrBC, self).__init__()
        self.num_layers = num_layers
        self.aggregate_type = aggregate_type

        if aggregate_type == 'GCN':
            self.convs = nn.ModuleList([GCNConv(dim, dim) for _ in range(num_layers)])
        elif aggregate_type == 'GAT':
            self.convs = nn.ModuleList([GATConv(dim, dim, heads=heads, concat=False, residual=True) for _ in range(num_layers)])
        else:
            self.aggr = NeighborhoodAggregation()

        self.encode_mlp = nn.Linear(6, dim, bias=False)
        
        self.decode_mlp = nn.Linear(dim, dim // 2, bias=False)
        self.decode_mlp2 = nn.Linear(dim // 2, 1, bias=False)
        
        self.gru_cell = nn.GRUCell(dim, dim)
        self.mha = nn.MultiheadAttention(dim, num_heads=heads, batch_first=True, dropout=0.2)
        self.graph_norm = GraphNorm(dim)
        self.dropout = nn.Dropout(0.15)
        self.apply(self.init_weights)

    def forward(self, data, norm=None):
        z = self.encode(data, norm)
        y = self.decode(z, data)
        return y.view(-1)

    def encode(self, data, norm=None):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        h_list = []
        h = F.leaky_relu(self.encode_mlp(x))
        h = self.dropout(h)
        h_list.append(h)

        prev_h = h
        
        for l in range(self.num_layers):

            if self.aggregate_type == 'GCN':
                h = self.convs[l](h, edge_index)    
            elif self.aggregate_type == 'GAT':
                h = self.convs[l](h, edge_index)
            elif self.aggregate_type == 'sum':
                h = self.aggr(h, edge_index, norm)

            h = self.gru_cell(h, prev_h) + prev_h
            h = self.graph_norm(h, batch)
            h_list.append(h)
            prev_h = h

        h_stack = torch.stack(h_list, dim=1)
        h_final = self.mha(h_stack, h_stack, h_stack)[0] + h_stack
        z = torch.sum(h_final, dim=1)
        return z

    def decode(self, z, data):
        y = F.leaky_relu(self.decode_mlp(z))
        y = self.dropout(y)
        y = self.decode_mlp2(y)
        return y
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)


