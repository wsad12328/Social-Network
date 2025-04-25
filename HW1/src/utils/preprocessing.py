import torch

def Generate_Feature(edge_index, num_nodes):
    row, col = edge_index

    degree = torch.bincount(row, minlength=num_nodes).float()

    deg_neighbors = degree[col]
    deg_count = torch.zeros(num_nodes, dtype=torch.float32)
    deg_sum = torch.zeros(num_nodes, dtype=torch.float32)
    deg_sumsq = torch.zeros(num_nodes, dtype=torch.float32)
    deg_sumsq3 = torch.zeros(num_nodes, dtype=torch.float32)  # 用於計算偏度
    ones = torch.ones_like(row, dtype=torch.float32)

    deg_sum.index_add_(0, row, deg_neighbors)
    deg_sumsq.index_add_(0, row, deg_neighbors ** 2)
    deg_sumsq3.index_add_(0, row, deg_neighbors ** 3)  # 計算三階矩
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

    # Min neighbor degree
    deg_min = torch.full((num_nodes,), float('inf'), dtype=torch.float32)
    deg_min = deg_min.index_reduce(0, row, deg_neighbors, reduce='amin')
    deg_min[deg_min == float('inf')] = 0.0

    # 偏度計算（以 standardized skewness 的方式）
    m3 = (deg_sumsq3 / deg_count.clamp(min=1)) - 3 * deg_mean * deg_var - deg_mean ** 3
    deg_skewness = torch.zeros_like(m3)
    valid_mask = deg_std > 1e-6  # 排除 std 為 0 的 case

    deg_skewness[valid_mask] = m3[valid_mask] / (deg_std[valid_mask] ** 3)
    deg_skewness[~valid_mask] = 0.0  # std 為 0 → 偏度 = 0
    deg_skewness -= deg_skewness.min()

    # 組合特徵
    features = torch.stack([
        degree,
        deg_std,
        deg_max,
        deg_min,
        deg_max/(deg_min + 1e-6),
        deg_skewness,
    ], dim=1)

    features = torch.log1p(features)

    return features
