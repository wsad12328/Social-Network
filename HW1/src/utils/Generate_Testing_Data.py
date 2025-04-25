import os
import torch
import argparse
from preprocessing import Generate_Feature
from torch_geometric.data import Data

def read_single_graph(edge_file, score_file):
    # 讀 edge list
    edge_index = []
    delimiter = '\t' if "Synthetic" in edge_file else ' '  # 根據檔案類型選擇分隔符
    with open(edge_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # 跳過空行
            try:
                u, v = map(int, line.split(delimiter))
                edge_index.append((u, v))
                edge_index.append((v, u))  # 無向圖需要雙向邊
            except Exception as e:
                print(f"Error parsing edge line '{line}': {e}")
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # 讀分數檔，支援 tab 或冒號分隔
    score_dict = {}
    with open(score_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # 跳過空行
            try:
                if ':' in line:
                    node_id, score = line.split(':\t')
                elif '\t' in line:
                    node_id, score = line.split('\t')
                else:
                    print(f"Unrecognized score format: {line}")
                    continue
                    
                score_dict[int(node_id.strip())] = float(score.strip())
            except Exception as e:
                print(f"Error parsing line '{line}': {e}")

    # 建 PyG 的資料結構
    num_nodes = max(score_dict.keys()) + 1  # 假設節點編號從 0 開始
    y = torch.zeros(num_nodes, dtype=torch.float)
    for node, score in score_dict.items():
        y[node] = score

    # 生成特徵
    node_features = Generate_Feature(edge_index, num_nodes)

    data = Data(x=node_features, edge_index=edge_index, y=y)
    return data

def load_all_graphs_from_folder(folder_path, num_graphs=30, idx=0):
    all_graphs = []
    print(f"Loading from folder: {folder_path}")
    for i in range(num_graphs):
        if "Synthetic" in folder_path:
            edge_path = os.path.join(folder_path, f"{i}.txt")
            score_path = os.path.join(folder_path, f"{i}_score.txt")
        elif "Real" in folder_path:
            edge_path = os.path.join(folder_path, f"{args.test_dataset[idx]}.txt")
            score_path = os.path.join(folder_path, f"{args.test_dataset[idx]}_score.txt")

        if os.path.exists(edge_path) and os.path.exists(score_path):
            data = read_single_graph(edge_path, score_path)
            all_graphs.append(data)
        else:
            print(f"Missing files: {edge_path} or {score_path}")
    return all_graphs

def load_all_datasets(root_dir, subfolders, num_graphs=30):
    dataset_dict = {}
    folder_path = root_dir
    for idx, folder_name in enumerate(subfolders):
        if "Synthetic" in folder_path:
            folder_path = os.path.join(root_dir, folder_name)
        dataset = load_all_graphs_from_folder(folder_path, num_graphs, idx)
        dataset_dict[folder_name] = dataset
    return dataset_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and save graph datasets")
    parser.add_argument('--test_dataset', nargs='+', required=True, help='List of subdirectories (e.g. 1000 10000)')
    parser.add_argument('--num_graphs', type=int, default=30, help='Number of graphs to load per subdirectory')
    parser.add_argument('--test_dir', type=str, help='Root path to the dataset')

    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.join(root, f'{args.test_dir}')
    print(f"Root path: {root_path}")
    print(f"Subdirs: {args.test_dir}")

    datasets = load_all_datasets(root_path, args.test_dataset, args.num_graphs)

    for key in datasets.keys():
        save_path = os.path.join(root_path, f"{key}.pt")
        torch.save(datasets[key], save_path)
        print(f"Saved {key} dataset to {save_path}")
