import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.utils import degree
from torch_geometric.loader import DataLoader
from DrBC import DrBC
from tqdm import tqdm
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_args():
    parser = argparse.ArgumentParser(description="Train DrBC model for betweenness centrality prediction.")
    parser.add_argument('--num_min', type=int, default=4000, help='Minimum number of nodes in graphs')
    parser.add_argument('--num_max', type=int, default=5000, help='Maximum number of nodes in graphs')
    parser.add_argument('--dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--layers', type=int, default=3, help='Number of GNN layers')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--aggregate_type', type=str, default='sum', help='Aggregation type for GNN')
    parser.add_argument('--data_path', type=str, default="../Data/Synthetic_Train", help='Path to training data')
    parser.add_argument('--save_dir', type=str, default="../Model", help='Directory to save model')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer')
    return parser.parse_args()

def compute_ranking_loss(pred_diff, true_diff, mode='hard', margin=0):
    """計算 ranking loss，支持 'hard' 和 'margin' 模式。
    
    Args:
        pred_diff (Tensor): 預測的差距（u, v）。
        true_diff (Tensor): 真實的差距（u, v）。
        mode (str): 訓練模式，'hard' 為 binary loss，'margin' 為 margin ranking loss。
        margin (float): margin ranking loss 的 margin 值。
    Returns:
        Tensor: 計算後的 loss。
    """
    if mode == 'hard':
        # Hard ranking loss
        hard_label = (true_diff > 0).float()  # 大於則為 1，否則為 0
        return F.binary_cross_entropy_with_logits(pred_diff, hard_label)
    
    elif mode == 'margin':
        # Margin ranking loss   
        target = (true_diff > 0).float() * 2 - 1  # +1 / -1
        margin = true_diff.abs()

        sorted_idx = torch.argsort(margin)
        rank = torch.empty_like(margin)
        rank[sorted_idx] = torch.arange(len(margin), dtype=margin.dtype, device=margin.device) + 1
        N = margin.size(0)
        rank = torch.sqrt((N*0.5 - rank).abs() + 1)

        loss = torch.clamp(margin*rank - pred_diff * target, min=0)  # margin ranking loss
        return loss.mean()

    else:
        raise ValueError("Invalid mode. Choose either 'hard' or 'margin'.")

def main():
    args = get_args()
    print("Device:", device)

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # Generate dynamic save path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = f"{base_dir}/{args.save_dir}/{args.aggregate_type}/"
    os.makedirs(save_dir, exist_ok=True)
    data_path = os.path.join(base_dir, f"{args.data_path}/{args.num_min}-{args.num_max}.pt")

    # Initialize the model
    model = DrBC(dim=args.dim, num_layers=args.layers, aggregate_type=args.aggregate_type).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # Load the training graphs
    graphs = torch.load(f"{data_path}", weights_only=False)
    print("Loaded training graphs:", len(graphs))
    
    # Split the graphs into two parts: first 1000 for first 100 epochs, last 1000 for last 100 epochs
    first_half_graphs = graphs[:1000]
    second_half_graphs = graphs[1000:]

    # Create two DataLoaders
    first_half_loader = DataLoader(first_half_graphs, batch_size=args.batch_size, shuffle=True)
    second_half_loader = DataLoader(second_half_graphs, batch_size=args.batch_size, shuffle=True)

    for epoch in range(args.epochs):
        if epoch < args.epochs // 2:
            loader = first_half_loader
        else:
            loader = second_half_loader

        avg_loss = 0

        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            batch = batch.to(device)
            optimizer.zero_grad()

            # Compute normalization factors
            row, col = batch.edge_index
            deg = degree(row, batch.x.size(0), dtype=batch.x.dtype)
            deg_inv_sqrt = 1.0 / torch.sqrt(deg + 1)
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

            # Model prediction
            scores = model(batch, norm)
            real_bc = batch.y
            # z-score normalization real_bc
            real_bc = (real_bc - real_bc.mean()) / real_bc.std()

            graph_ids, num_nodes_per_graph = torch.unique(batch.batch, return_counts=True)
            ranking_losses = []

            for graph_id, num_nodes in zip(graph_ids, num_nodes_per_graph):
                mask = batch.batch == graph_id
                node_indices = torch.where(mask)[0]

                num_pairs = 5 * num_nodes.item()
                if num_nodes.item() < 2:
                    continue

                idx = torch.randint(0, len(node_indices), (num_pairs, 2))
                u, v = node_indices[idx[:, 0]], node_indices[idx[:, 1]]
                pred_diff = scores[u] - scores[v]
                true_diff = real_bc[u] - real_bc[v]

                # Calculate ranking loss using selected mode
                loss = compute_ranking_loss(pred_diff, true_diff)
                ranking_losses.append(loss)

            if len(ranking_losses) > 0:
                ranking_loss = torch.stack(ranking_losses).mean()
            else:
                ranking_loss = torch.tensor(0.0, device=device, requires_grad=True)

            ranking_loss.backward()

            optimizer.step()
            avg_loss += ranking_loss.item()

        avg_loss /= len(loader)

        # Update learning rate
        scheduler.step()
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}")

    # Save the model
    model_save_path = os.path.join(save_dir, f"{args.dim}_{args.layers}.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    print("Training complete.")



if __name__ == "__main__":
    main()
