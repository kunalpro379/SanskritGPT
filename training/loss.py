import torch
import torch.nn.functional as F


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    config
) -> torch.Tensor:
    batch_size, seq_len, vocab_size = logits.shape
# Only compute loss on:
    # 1. Positions that were masked (mask == True)
    # 2. Positions that are not padding (targets != pad_id)
    loss_mask = mask & (targets != config.pad_id)
# Reshape logits: [B, L, V] -> [M, V] where M = number of masked positions
    logits_flat = logits[loss_mask]  # [M, vocab_size]
    # Reshape targets: [B, L] -> [M]
    targets_flat = targets[loss_mask]  # [M]
    
    # ========== COMPUTE CROSS-ENTROPY LOSS ==========
    # Label smoothing helps with generalization
    ce_loss = F.cross_entropy(
        logits_flat, 
        targets_flat, 
        label_smoothing=0.1  # Smooth labels by 10%
    )
    
    return ce_loss


    