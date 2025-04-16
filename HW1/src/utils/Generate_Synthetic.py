import torch
import torch_geometric.utils as pyg_utils
import numpy as np
import networkx as nx
import igraph as ig
from torch_geometric.data import Data
from preprocessing import Generate_Feature

def Generate_Synthetic(num_min, num_max, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 隨機生成節點數量
    num_nodes = np.random.randint(num_max - num_min + 1) + num_min
    G = nx.powerlaw_cluster_graph(n=num_nodes, m=4, p=0.05, seed=seed)

    # 計算 Betweenness Centrality
    bc_values = ig.Graph.from_networkx(G).betweenness()
    # 將 bc 值轉換為 torch tensor
    N = len(G.nodes)
    max_bc = (N - 1) * (N - 2) / 2 
    bc_values = torch.tensor(bc_values) / max_bc

    # Generate features
    features = Generate_Feature(G)

    # 節點連接資訊（從網絡圖中提取）
    edge_index = pyg_utils.from_networkx(G).edge_index
    # 將手動特徵和 bc 值打包進 Data
    SyntheticData = Data(x=features, edge_index=edge_index, y=bc_values)
    return SyntheticData
