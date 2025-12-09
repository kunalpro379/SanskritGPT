import torch
from typing import Tuple

def get_mask_ratio(t: int, config) -> float:
    return config.mask_start + (config.mask_end - config.mask_start) * (t - 1) / (config.T - 1)


def forward_mask(
    x0: torch.Tensor, 
    real_len: int, 
    t: int, 
    config
) -> Tuple[torch.Tensor, torch.Tensor]:
    seq_len = x0.size(0)
    xt = x0.clone() 
    mask = torch.zeros(seq_len, dtype=torch.bool, device=x0.device)
    if real_len <= 2:
        return xt, mask

    special_tokens = torch.tensor([
        config.pad_id, 
        config.bos_id, 
        config.eos_id
    ], device=x0.device)

    special_mask = torch.isin(x0, special_tokens)
    position_mask = torch.arange(seq_len, device=x0.device) < real_len
    eligible = ~special_mask & position_mask
    eligible_indices = torch.where(eligible)[0]
    if len(eligible_indices) == 0:
        return xt, mask

#calculate mask 
    mask_ratio = get_mask_ratio(t, config)
    num_mask = max(1, int(mask_ratio * len(eligible_indices)))
    num_mask = min(num_mask, len(eligible_indices)) 
    perm = torch.randperm(len(eligible_indices), device=x0.device)
    masked_idx = eligible_indices[perm[:num_mask]]
    xt[masked_idx] = config.mask_id
    
    # Mark positions as masked
    mask[masked_idx] = True
    
    return xt, mask