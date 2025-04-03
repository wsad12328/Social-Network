import torch
import torch_geometric.utils as pyg_utils
import numpy as np
import networkx as nx
import igraph as ig
from torch_geometric.data import Data

def Generate_Synthetic(num_min, num_max, seed, embed_dim=16):
    np.random.seed(seed + 100)
    torch.manual_seed(seed + 100)

    # 隨機生成節點數量
    num_nodes = np.random.randint(num_max - num_min + 1) + num_min
    G = nx.powerlaw_cluster_graph(n=num_nodes, m=4, p=0.05, seed=seed+100)

    # 計算 Betweenness Centrality
    bc_values = ig.Graph.from_networkx(G).betweenness()
    N = len(G.nodes)
    max_bc = (N - 1) * (N - 2) / 2 
    bc_values = torch.tensor(bc_values) / max_bc

    # 計算 eigenvector centrality（需要計算整個圖的特徵向量）
    eigenvector_centrality = nx.eigenvector_centrality_numpy(G)
    eigenvector_centrality_values = torch.tensor(list(eigenvector_centrality.values()))

    # 計算 clustering coefficient
    clustering_coeff = nx.clustering(G)
    clustering_coeff_values = torch.tensor(list(clustering_coeff.values()))

    # 計算 degree（度數）
    degree = torch.tensor([val for (node, val) in G.degree()])

    features = torch.stack([degree.float(), 
                            eigenvector_centrality_values.float(), 
                            clustering_coeff_values.float()], dim=1)

    # 節點連接資訊（從網絡圖中提取）
    edge_index = pyg_utils.from_networkx(G).edge_index

    # 將手動特徵和 bc 值打包進 Data
    SyntheticData = Data(x=features, edge_index=edge_index, y=bc_values)
    return SyntheticData
