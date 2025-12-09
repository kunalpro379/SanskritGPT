import torch
import torch.nn as nn
from typing import Optional

from .embeddings import TokenEmbedding, PositionEmbedding, TimestepEmbedding
from .transformer import TransformerEncoder

class TransformerEncoderDiffusion(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        # mapp token ids to vectors
        self.token_embed=TokenEmbedding(config.vocab_size, config.d_model)
        # //positional embeddings
        self.pos_embed=PositionEmbedding(config.seq_len, config.d_model)
        #Timestemp embeddings
        self.timestep_embed=TimestepEmbedding(config.d_model, config.T)
        #Transformer encoder
        self.transformer=TransformerEncoder(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            n_layers=config.n_layers,
            dropout=config.dropout
        )
        # Layer normalization before output
        self.ln_out=nn.LayerNorm(config.d_model)
        self.head=nn.Linear(config.d_model, config.vocab_size)
        self._init_weights()#initialize weights

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:  # Only initialize 2D+ tensors (not biases)
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x:torch.Tensor,
        t:torch.Tensor,
        attention_mask:Optional[torch.Tensor]=None
    )-> torch.Tensor:
        batch_size, seq_len = x.shape
        # Token embeddings: [B, L] -> [B, L, D]
        token_emb = self.token_embed(x)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embed(positions)

        # Timestep embeddings: [B] -> [B, D] -> [B, 1, D]
        t_emb = self.timestep_embed(t).unsqueeze(1)

        # Combine all embeddings
        h = token_emb + pos_emb + t_emb  # [B, L, D]
        if attention_mask is None:
            attention_mask = (x != self.config.pad_id).float()
        pad_mask = (attention_mask == 0)

        h = self.transformer(h, src_key_padding_mask=pad_mask)  # [B, L, D]

        # Layer normalization
        h = self.ln_out(h)  # [B, L, D]
        # Predict token logits
        logits = self.head(h)  # [B, L, vocab_size]
        
        return logits



