import numpy as np
import torch
from typing import Tuple


def get_real_length(sequence: np.ndarray, pad_id: int) -> int:
    return int(np.sum(sequence != pad_id))


def preprocess_sequence(sequence: np.ndarray, max_len: int, pad_id: int) -> np.ndarray:
    if len(sequence) > max_len:
        return sequence[:max_len]
    elif len(sequence) < max_len:
        padded = np.full(max_len, pad_id, dtype=sequence.dtype)
        padded[:len(sequence)] = sequence
        return padded
    return sequence


def validate_sequence(sequence: torch.Tensor, config) -> bool:
    if sequence[0].item() != config.bos_id:
        return False
    
    # Check for EOS somewhere
    has_eos = (sequence == config.eos_id).any().item()
    if not has_eos:
        return False
    
    return True