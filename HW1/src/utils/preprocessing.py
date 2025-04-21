import torch

def Generate_Feature(edge_index, num_nodes):
    row, col = edge_index

    degree = torch.bincount(row, minlength=num_nodes).float()

    deg_neighbors = degree[col]
    deg_count = torch.zeros(num_nodes, dtype=torch.float32)
    deg_sum = torch.zeros(num_nodes, dtype=torch.float32)
    deg_sumsq = torch.zeros(num_nodes, dtype=torch.float32)

    ones = torch.ones_like(row, dtype=torch.float32)

    deg_sum.index_add_(0, row, deg_neighbors)
    deg_sumsq.index_add_(0, row, deg_neighbors ** 2)
    deg_count.index_add_(0, row, ones)

    deg_mean = deg_sum / deg_count.clamp(min=1)
    deg_var = deg_sumsq / deg_count.clamp(min=1) - deg_mean ** 2
    deg_std = deg_var.clamp(min=0).sqrt()

    deg_mean[deg_count == 0] = 0.0
    deg_std[deg_count == 0] = 0.0

    # Max neighbor degree
    deg_max = torch.full((num_nodes,), float('-inf'), dtype=torch.float32)
    deg_max = deg_max.index_reduce(0, row, deg_neighbors, reduce='amax')
    deg_max[deg_max == float('-inf')] = 0.0

    relative_deg_ratio = degree / deg_mean.clamp(min=1e-6)

    features = torch.stack([
        degree,
        degree,  # 原本這是 log_degree，但我們等一下整體做 log1p
        deg_mean,
        deg_max,
        deg_std,
        relative_deg_ratio
    ], dim=1)

    # 一次性 log1p 全部特徵（包含 degree 那兩欄）
    features = torch.log1p(features)

    return features
