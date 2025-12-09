import torch
import torch.nn as nn
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)


class PositionEmbedding(nn.Module):
    def __init__(self, seq_len: int, d_model: int):
        super().__init__()
        self.embed = nn.Embedding(seq_len, d_model)
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return self.embed(positions)


class TimestepEmbedding(nn.Module):
    def __init__(self, d_model: int, T: int):
        super().__init__()
        # +1 for t=0 (optional, for edge cases)
        self.embed = nn.Embedding(T + 1, d_model)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.embed(t)