
import torch
import torch.nn.functional as F


def top_p_filtering(
    logits: torch.Tensor, 
    top_p: float = 0.9, 
    temperature: float = 1.0
) -> torch.Tensor:

    # Apply temperature scaling
    logits = logits / temperature
    
    # Sort tokens by probability (descending)
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    
    # Compute cumulative probabilities
    probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(probs, dim=-1)
    
    # Find tokens to remove (cumulative prob > top_p)
    sorted_indices_to_remove = cumulative_probs > top_p
    
    # Shift right to keep first token above threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    # Map back to original indices
    indices_to_remove = sorted_indices_to_remove.scatter(
        -1, 
        sorted_indices, 
        sorted_indices_to_remove
    )
    
    logits[indices_to_remove] = -float('Inf')
    
    return logits