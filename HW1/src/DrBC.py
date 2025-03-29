import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
import networkx as nx
import torch_geometric.utils as pyg_utils

class NeighborhoodAggregation(MessagePassing):
    def __init__(self):
        super(NeighborhoodAggregation, self).__init__(aggr='add')  # "加法"聚合。

    def forward(self, x, edge_index, norm):
        # x 的形狀為 [N, dim]
        # edge_index 的形狀為 [2, E]
        # 將嵌入傳播到鄰居節點。
        return self.propagate(edge_index, x=x, norm=norm)
    
    def message(self, x_j, norm):
        # x_j 的形狀為 [E, out_channels]
        # norm 的形狀為 [E]
        return norm.view(-1, 1) * x_j
    
class DrBC(nn.Module):
    def __init__(self, dim, num_layers):
        super(DrBC, self).__init__()
        self.layer = num_layers
        self.aggr = NeighborhoodAggregation()
        self.encode_mlp = nn.Linear(3, dim, bias=False)
        self.decode_mlp = nn.Linear(dim, dim//2, bias=False)
        self.decode_mlp2 = nn.Linear(dim//2, 1, bias=False)
        self.GRU_Cell = nn.GRUCell(dim, dim)
        self.apply(self.init_weights) 

    def forward(self, data, norm):
        z = self.encode(data, norm)
        y = self.decode(z)
        return y.view(-1)
    
    def encode(self, data, norm):
        x, edge_index = data.x, data.edge_index
        # x 的形狀為 [N, 3]
        # 初始化 h，記錄每一層的狀態
        h_list = []
        h_prev = F.relu(self.encode_mlp(x))  # 第一層的初始狀態
        h_prev = F.normalize(h_prev, p=2, dim=1)
        h_list.append(h_prev)  # 記錄第一層的狀態

        # 迭代計算每一層的 h
        for _ in range(self.layer - 1):
            # 聚合鄰居信息
            next_h_Nv = self.aggr(h_prev, edge_index, norm)

            # 使用 GRU 更新隱藏狀態
            h_next = self.GRU_Cell(next_h_Nv, h_prev)

            # 歸一化
            h_next = F.normalize(h_next, p=2, dim=1)

            # 記錄當前層的狀態
            h_list.append(h_next)

            # 更新 h_prev 為下一層的輸入
            h_prev = h_next

        # 將所有層的 h 堆疊成一個張量
        h = torch.stack(h_list, dim=1)  # 形狀: [N, layer, 128]
        z = h.max(dim=1)[0]  # 形狀: [N, 128]
        return z
    
    def decode(self, z):
        y = F.relu(self.decode_mlp(z))
        y = self.decode_mlp2(y)
        return y
    
    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
        if type(m) == nn.GRUCell:
            for name, param in m.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                if 'bias' in name:
                    param.data.fill_(0.01)
        return m
    