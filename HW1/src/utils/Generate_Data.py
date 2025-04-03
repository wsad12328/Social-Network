import argparse
import os
import torch
from Generate_Synthetic import Generate_Synthetic
from tqdm import tqdm

def main(args):
    # Define paths relative to the script's location
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
    train_dir = os.path.join(base_dir, args.train_dir)
    validation_dir = os.path.join(base_dir, args.validation_dir)
    
    # Print current settings
    print("Current Configuration:")
    print(f"  Number of nodes: {args.num_min} - {args.num_max}")
    print(f"  Training samples: {args.train_samples}")
    print(f"  Validation samples: {args.test_samples}")
    print(f"  Training directory: {train_dir}")
    print(f"  Validation directory: {validation_dir}")
    print("Starting dataset generation...\n")
    
    # Ensure the directories exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)
    
    # Generate and save training graphs with progress bar
    if args.train_samples > 0:
        print("Generating training graphs...")
        train_graphs = [Generate_Synthetic(args.num_min, args.num_max, seed + 1000) for seed in tqdm(range(args.train_samples), desc="Generating Training Graphs")]
        torch.save(train_graphs, os.path.join(train_dir, f"{args.num_min}-{args.num_max}.pt"))
    
    # Generate and save testing graphs with progress bar
    if args.test_samples > 0:
        print("Generating testing graphs...")
        test_graphs = [Generate_Synthetic(args.num_min, args.num_max, seed + 2000) for seed in tqdm(range(args.test_samples), desc="Generating Testing Graphs")]
        torch.save(test_graphs, os.path.join(validation_dir, f"{args.num_min}-{args.num_max}.pt"))
    
    print("\nDataset generation completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic graph datasets.")
    parser.add_argument("--num_min", type=int, default=100, help="Minimum number of nodes.")
    parser.add_argument("--num_max", type=int, default=200, help="Maximum number of nodes.")
    parser.add_argument("--train_samples", type=int, default=100, help="Number of training samples to generate.")
    parser.add_argument("--test_samples", type=int, default=10, help="Number of testing samples to generate.")
    parser.add_argument("--train_dir", type=str, default="../../Data/Synthetic_Train", help="Directory to save training data.")
    parser.add_argument("--validation_dir", type=str, default="../../Data/Synthetic_Validation", help="Directory to save validation data.")
    
    args = parser.parse_args()
    main(args)