import torch
import numpy as np
import random


def get_device() -> torch.device:
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA built with PyTorch: {torch.version.cuda if torch.version.cuda else 'No'}")
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"CUDA available! Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        return device
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("MPS (Apple Silicon) available! Using GPU")
        return device
    else:
    
        return torch.device('cpu')


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Set CUDA seeds if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False