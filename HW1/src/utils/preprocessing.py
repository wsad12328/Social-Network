import networkx as nx
import torch
import time
import numpy as np

def Generate_Feature(edge_index, num_nodes):
    # 計算 degree（度數）
    row, _ = edge_index
    degree = torch.bincount(row, minlength=num_nodes)

    log_degree = torch.log(degree.float() + 1)
    log_degree_mean = log_degree.mean()
    log_degree_std = log_degree.std()
    log_degree_mean = log_degree_mean.expand(num_nodes)  # Expand to match the number of nodes
    log_degree_std = log_degree_std.expand(num_nodes)    # Expand to match the number of nodes

    # Stack features
    features = torch.stack([log_degree,
                            log_degree_mean,
                            log_degree_std], dim=1)

    return features
