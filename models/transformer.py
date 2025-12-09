import torch
import torch.nn as nn
from typing import Optional


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        dropout: float = 0.1
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(
        self, 
        x:torch.Tensor,
        src_key_padding_mask:Optional[torch.Tensor]=None
    )->torch.Tensor:
        return self.transformer(x, src_key_padding_mask=src_key_padding_mask)

