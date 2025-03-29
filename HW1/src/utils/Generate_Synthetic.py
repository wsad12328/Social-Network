import igraph as ig
import torch
import networkx as nx
from torch_geometric.data import Data
import torch_geometric.utils as pyg_utils

def Generate_Synthetic(seed):
    # Generate a synthetic graphG = nx.powerlaw_cluster_graph(n=5000, m=4, p=0.05)
    G = nx.powerlaw_cluster_graph(n=5000, m=4, p=0.05, seed=seed+100)
    bc_values = ig.Graph.from_networkx(G).betweenness()
    N = len(G.nodes)
    max_bc = (N - 1) * (N - 2) / 2  # 最大可能值
    bc_values = torch.tensor(bc_values)
    bc_values = bc_values / max_bc
    features = torch.ones(N, 3)
    degree = torch.tensor([val for (node, val) in G.degree()])
    features[:,0] = degree
    SyntheticData = Data(x=features, edge_index=pyg_utils.from_networkx(G).edge_index, y=bc_values)
    return SyntheticData