import argparse
import torch
from torch_geometric.utils import degree
from torch_geometric.loader import DataLoader
from DrBC import DrBC
from tqdm import tqdm
import numpy as np
from scipy.stats import kendalltau
import os
import csv

def get_args():
    parser = argparse.ArgumentParser(description="Train DrBC model for betweenness centrality prediction.")
    parser.add_argument('--num_min', type=int, default=4000, help='Minimum number of nodes in graphs')
    parser.add_argument('--num_max', type=int, default=5000, help='Maximum number of nodes in graphs')
    parser.add_argument('--dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--layers', type=int, default=5, help='Number of GNN layers')
    parser.add_argument('--aggregate_type', type=str, default='sum', help='Aggregation type for GNN')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--model_dir', type=str, default="../Model", help='Directory to save model')
    parser.add_argument('--test_data_path', type=str, default="../DrBC_Data/Synthetic", help='Path to validation data')
    parser.add_argument('--output_base', type=str, default="../Result", help='Base directory to save evaluation results')
    parser.add_argument('--test_graphs_min', type=int, default=100000, help='Minimum number of nodes in test graphs')
    parser.add_argument('--test_graphs_max', type=int, default=100000, help='Maximum number of nodes in test graphs')
    parser.add_argument('--topk_list', type=int, nargs='+', default=[1, 5, 10], help='List of Top K percentages for evaluation')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer')
    return parser.parse_args()

def evaluate_graphs(y_pred, y_true, batch_indices, k_percentages=[1, 5, 10]):
    """
    針對 batch 內每張圖計算不同的 Top-K 準確率與 Kendall's Tau，並取平均
    :param y_pred: 預測 BC 值 (Tensor)
    :param y_true: 真實 BC 值 (Tensor)
    :param batch_indices: batch 內圖的索引 (Tensor)
    :param k_percentages: List of Top-K percentages
    :return: 各個 Top-K 的平均準確率, Kendall's Tau
    """
    graph_ids = torch.unique(batch_indices)  # 取得 batch 內所有圖 ID
    top_k_acc_dict = {k: [] for k in k_percentages}  # 用字典來存每個 Top-K 準確率
    tau_list = []
    
    for graph_id in graph_ids:
        mask = batch_indices == graph_id  # 找出屬於該圖的節點
        pred_vals = y_pred[mask].cpu().numpy().flatten()  # 取得該圖的預測值
        true_vals = y_true[mask].cpu().numpy().flatten()  # 取得該圖的真實值

        if len(pred_vals) < 2:
            continue  # 避免單節點圖影響計算
        
        for k_percentage in k_percentages:
            k = max(1, int(len(pred_vals) * k_percentage / 100))  # 確保至少選 1 個
            top_k_pred = np.argsort(-pred_vals)[:k]
            top_k_true = np.argsort(-true_vals)[:k]

            top_k_acc = len(set(top_k_pred) & set(top_k_true)) / k
            top_k_acc_dict[k_percentage].append(top_k_acc)

        # 計算 Kendall's Tau-b，避免 NaN
        tau, _ = kendalltau(pred_vals, true_vals, variant="b")
        if not np.isnan(tau):
            tau_list.append(tau)
    # 計算每批次的平均 Top-K 準確率和 Kendall's Tau
    avg_top_k_acc_dict = {k: np.mean(v) if v else 0.0 for k, v in top_k_acc_dict.items()}
    avg_tau = np.mean(tau_list) if tau_list else 0.0
    return avg_top_k_acc_dict, avg_tau

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = f"{base_dir}/{args.model_dir}/{args.aggregate_type}"
    model_path = f"{model_dir}/{args.num_min}_{args.num_max}.pth"

    model = DrBC(dim=args.dim, num_layers=args.layers, aggregate_type=args.aggregate_type)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully.")
    test_graphs = None
    if "Synthetic" in args.test_data_path:
        test_graphs = torch.load(f"{base_dir}/{args.test_data_path}/{args.test_graphs_min}.pt", weights_only=False)

    elif 'Real' in args.test_data_path:
        test_graphs = torch.load(f"{base_dir}/{args.test_data_path}/com-youtube.pt", weights_only=False)

    test_loader = DataLoader(test_graphs, batch_size=1, shuffle=False)

    # Generate output path based on parameters
    if "Synthetic" in args.test_data_path:
        output_dir = f"{base_dir}/{args.output_base}/test_{args.test_graphs_min}"
    elif 'Real' in args.test_data_path:
        output_dir = f"{base_dir}/{args.output_base}/com-youtube"

    os.makedirs(output_dir, exist_ok=True)

    if "Synthetic" in args.test_data_path:
        output_file = f"{output_dir}/{args.num_min}-{args.num_max}.csv"
    elif 'Real' in args.test_data_path:
        output_file = f"{output_dir}/com-youtube.csv"
    
    # Write results to CSV file
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Top-K%', 'Accuracy', 'Kendall\'s Tau'])  # CSV 標題行
        
        total_top_k_acc_dict = {k: 0 for k in args.topk_list}
        total_tau = 0
        num_batches = 0
        with torch.no_grad():
            for batch in tqdm(test_loader):
                # batch = batch
                row, col = batch.edge_index
                deg = degree(row, batch.x.size(0), dtype=batch.x.dtype)
                deg_inv_sqrt = 1.0 / torch.sqrt(deg + 1)
                norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

                pred_bc = model(batch, norm)
                top_k_acc_dict, tau = evaluate_graphs(pred_bc, batch.y, batch.batch, k_percentages=args.topk_list)
                
                for k in args.topk_list:
                    total_top_k_acc_dict[k] += top_k_acc_dict[k]
                total_tau += tau
                num_batches += 1

        # Calculate average results for each Top-K
        avg_top_k_acc_dict = {k: total_top_k_acc_dict[k] / num_batches for k in args.topk_list}
        avg_tau = total_tau / num_batches

        # Write average results to CSV
        for k in args.topk_list:
            writer.writerow([f"Top-{k}%", avg_top_k_acc_dict[k], avg_tau])

    print(f"Evaluation results saved to {output_file}")

if __name__ == "__main__":
    main()
