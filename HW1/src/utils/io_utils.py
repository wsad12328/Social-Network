import os
import torch
import pandas as pd
import networkx as nx
from torch_geometric.data import Data

def load_edge_list(file_path):
    """
    讀取 edge list 檔案，返回 NetworkX 無向圖。
    
    Args:
        file_path (str): 邊列表檔案路徑（如 `0.txt`）。
    
    Returns:
        G (networkx.Graph): NetworkX 圖物件
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到檔案: {file_path}")

    G = nx.read_edgelist(file_path, nodetype=int)
    return G

def load_edge_list_pyg(file_path, num_nodes=None, features=None, labels=None):
    """
    讀取 edge list 檔案，轉換為 PyTorch Geometric 的 edge_index 格式。
    
    Args:
        file_path (str): 邊列表檔案路徑（如 `0.txt`）。
        num_nodes (int, optional): 節點總數（如果有）。
        features (torch.Tensor, optional): 節點特徵矩陣 (N, F)。
        labels (torch.Tensor, optional): 節點標籤 (N, )。
    
    Returns:
        data (torch_geometric.data.Data): PyG 格式的圖數據
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到檔案: {file_path}")

    edges = pd.read_csv(file_path, sep="\t", header=None)
    edge_index = torch.tensor(edges.values, dtype=torch.long).t().contiguous()
    
    # 確保 PyG 知道圖的節點數
    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1

    # 創建 PyG Data 物件
    data = Data(edge_index=edge_index, num_nodes=num_nodes)
    
    if features is not None:
        data.x = features
    if labels is not None:
        data.y = labels
    
    return data

def load_score(file_path, num_nodes=None):
    """
    讀取 `_score.txt` 檔案，返回節點的分數（PyTorch Tensor）。
    
    Args:
        file_path (str): 分數檔案路徑（如 `0_score.txt`）。
        num_nodes (int, optional): 如果指定了節點數，會填充未出現的節點。
    
    Returns:
        scores (torch.Tensor): 節點對應的分數 (num_nodes, 1) 的 PyTorch tensor
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到檔案: {file_path}")

    scores_dict = {}
    with open(file_path, "r") as f:
        for line in f:
            node, score = line.strip().split()
            scores_dict[int(node)] = float(score)

    if num_nodes is None:
        num_nodes = max(scores_dict.keys()) + 1

    scores = torch.zeros(num_nodes, 1)  # 預設為 0
    for node, score in scores_dict.items():
        scores[node] = score

    return scores

def save_graph_pyg(data, save_path):
    """
    儲存 PyTorch Geometric 的圖數據到 `.pt` 檔案。
    
    Args:
        data (torch_geometric.data.Data): PyG 格式的圖數據
        save_path (str): 儲存路徑（如 `processed/graph.pt`）
    """
    torch.save(data, save_path)

def load_graph_pyg(load_path):
    """
    讀取 `.pt` 格式的 PyTorch Geometric 圖數據。

    Args:
        load_path (str): `.pt` 檔案的路徑
    
    Returns:
        data (torch_geometric.data.Data): 讀取的 PyG 圖數據
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"找不到檔案: {load_path}")
    
    data = torch.load(load_path)
    return data

def load_graph_with_scores(edge_path, score_path):
    """
    讀取圖的 edge list 和對應的 score，回傳 PyG 的 Data 物件。
    
    Args:
        edge_path (str): 邊列表的檔案路徑（如 `0.txt`）。
        score_path (str): 節點分數檔案（如 `0_score.txt`）。
    
    Returns:
        data (torch_geometric.data.Data): PyG 格式的圖數據（包含 y）
    """
    # 讀取圖結構
    G = load_edge_list(edge_path)
    num_nodes = G.number_of_nodes()
    
    # 讀取 PyG edge_index
    data = load_edge_list_pyg(edge_path, num_nodes=num_nodes)

    # 讀取分數並存入 data.y
    data.y = load_score(score_path, num_nodes=num_nodes)

    return data
