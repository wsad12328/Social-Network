import argparse
import os
import torch
from Generate_Synthetic import Generate_Synthetic
from tqdm import tqdm
from torch_geometric.data import Data
import networkx as nx
from preprocessing import Generate_Feature
import torch_geometric.utils as pyg_utils

def main(args):
    # Define paths relative to the script's location
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
    train_dir = os.path.join(base_dir, args.train_dir)
    test_dir = os.path.join(base_dir, args.test_dir)
    # Print current settings
    print("Current Configuration:")
    print(f"  Number of nodes: {args.num_min} - {args.num_max}")
    print(f"  Training samples: {args.train_samples}")
    print(f"  Validation samples: {args.test_samples}")
    print(f"  Training directory: {train_dir}")
    print(f"  Validation directory: {test_dir}")
    print("Starting dataset generation...\n")
    
    # Generate and save training graphs with progress bar
    if args.train_samples > 0:
        os.makedirs(train_dir, exist_ok=True)
        print("Generating training graphs...")
        train_graphs = [Generate_Synthetic(args.num_min, args.num_max, seed + 1000) for seed in tqdm(range(args.train_samples), desc="Generating Training Graphs")]
        torch.save(train_graphs, os.path.join(train_dir, f"{args.num_min}-{args.num_max}.pt"))

    if "Synthetic" in args.test_dir:
        os.makedirs(test_dir, exist_ok=True)
        # Generate and save testing graphs with progress bar
        if args.test_samples > 0:
            print("Generating testing graphs...")
            

    elif 'Real' in args.test_dir:
        edge_index = []
        with open(os.path.join(test_dir, f"com-youtube.txt"), 'r') as f:
            for line in f.readlines():
                u, v = map(int, line.split())  # Split by space and convert to integers
                edge_index.append((u, v))
                
        # print(edge_index)

        y = []

        with open(os.path.join(test_dir, f"com-youtube_score.txt"), 'r') as f:
            for line in f.readlines():
                value = float(line.split(':')[1].strip())  # Extract the number after the colon
                y.append(value)
         
        y = torch.tensor(y, dtype=torch.float).view(-1, 1)  # Reshape to (num_nodes, 1)

        G = nx.Graph()
        G.add_edges_from(edge_index)

        # Generate features using the Generate_Feature function
        node_features = Generate_Feature(G)

        edges = pyg_utils.from_networkx(G).edge_index

        # Create a Data object
        data = [Data(x=node_features, edge_index=edges, y=y)]

        # Save the data object
        torch.save(data, os.path.join(test_dir, f"com-youtube.pt"))
        
    
    print("\nDataset generation completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic graph datasets.")
    parser.add_argument("--num_min", type=int, default=4000, help="Minimum number of nodes.")
    parser.add_argument("--num_max", type=int, default=5000, help="Maximum number of nodes.")
    parser.add_argument("--train_samples", type=int, default=0, help="Number of training samples to generate.")
    parser.add_argument("--test_samples", type=int, default=0, help="Number of testing samples to generate.")
    parser.add_argument("--train_dir", type=str, default="../../Data/Synthetic_Train", help="Directory to save training data.")
    parser.add_argument("--test_dir", type=str, default="../../Data/youtube", help="Directory to save validation data.")

    
    args = parser.parse_args()
    main(args)