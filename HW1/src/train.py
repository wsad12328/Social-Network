import argparse
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.utils import degree
from torch_geometric.loader import DataLoader
from DrBC import DrBC
from tqdm import tqdm
import numpy as np
import random
import os

def get_args():
    parser = argparse.ArgumentParser(description="Train DrBC model for betweenness centrality prediction.")
    parser.add_argument('--num_min', type=int, default=4000, help='Minimum number of nodes in graphs')
    parser.add_argument('--num_max', type=int, default=5000, help='Maximum number of nodes in graphs')
    parser.add_argument('--dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--layers', type=int, default=3, help='Number of GNN layers')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=250, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--aggregate_type', type=str, default='sum', help='Aggregation type for GNN')
    parser.add_argument('--data_path', type=str, default="../Data/Synthetic_Train", help='Path to training data')
    parser.add_argument('--save_dir', type=str, default="../Model", help='Directory to save model')
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # Generate dynamic save path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = f"{base_dir}/{args.save_dir}/{args.aggregate_type}/"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"DrBC_{args.num_min}-{args.num_max}_dim{args.dim}_layers{args.layers}.pth")
    data_path = os.path.join(base_dir, f"{args.data_path}/{args.num_min}-{args.num_max}.pt")

    # Initialize the model
    model = DrBC(dim=args.dim, num_layers=args.layers, aggregate_type=args.aggregate_type).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Load the training graphs
    graphs = torch.load(f"{data_path}", weights_only=False)
    print("Loaded training graphs:", len(graphs))
    loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=True)

    for epoch in range(args.epochs):
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

            graph_ids, num_nodes_per_graph = torch.unique(batch.batch, return_counts=True)
            ranking_loss = 0.0

            for graph_id, num_nodes in zip(graph_ids, num_nodes_per_graph):
                mask = batch.batch == graph_id
                node_indices = torch.where(mask)[0]

                num_pairs = 5 * num_nodes.item()
                if num_nodes.item() < 2:
                    continue

                idx = torch.randint(0, len(node_indices), (num_pairs, 2), device=device)
                u, v = node_indices[idx[:, 0]], node_indices[idx[:, 1]]
                s_u, s_v = scores[u], scores[v]
                b_ij = s_u - s_v
                y_ij = (real_bc[u] > real_bc[v]).float()

                loss = F.binary_cross_entropy_with_logits(b_ij, y_ij)
                ranking_loss += loss

            ranking_loss /= args.batch_size
            ranking_loss.backward()
            optimizer.step()
            avg_loss += ranking_loss.item()

        avg_loss /= len(loader)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}")

    # Save the model
    torch.save({'model_state_dict': model.state_dict()}, save_path)
    print(f"Model saved successfully at {save_path}")

if __name__ == "__main__":
    main()
