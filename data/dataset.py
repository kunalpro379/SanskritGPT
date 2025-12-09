import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple
from data.data_utils import get_real_length
class SanskritDataset(Dataset):
    def __init__(self, data_path: str, config):
        self.data = np.load(data_path)
        self.config = config
        self.N = len(self.data)
        self.real_lengths = np.sum(self.data != config.pad_id, axis=1)

    def __len__(self) -> int:
        return self.N


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:

        seq = self.data[idx].astype(np.int64)
        real_len = self.real_lengths[idx]
        return torch.from_numpy(seq), real_len


        