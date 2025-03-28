import torch
from src.utils.Generate_Synthetic import Generate_Synthetic

def print_memory_usage(step):
    print(f"{step} - Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB, "
          f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 示例：在关键步骤插入
print_memory_usage("Before loading data")
Synthetic_Data = Generate_Synthetic()
Synthetic_Data = Synthetic_Data.to(device)
print_memory_usage("After loading data")