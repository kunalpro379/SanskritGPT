import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple
import time

from utils.sampling import top_p_filtering
from inference.decoder import decode_tokens

def generate_unconditional(
    model, 
    device, 
    config, 
    steps: Optional[int] = None,
    show_steps: bool = True
) -> Tuple[List[int], List[str]]:
    model.eval()
    if steps is None:
        steps = config.T

    x = torch.full((1, config.seq_len), config.mask_id, device=device)
    x[0, 0] = config.bos_id
    x[0, -1] = config.eos_id

    fixed = torch.zeros_like(x, dtype=torch.bool)
    fixed[:, 0] = True
    fixed[:, -1] = True
    thresholds = torch.linspace(0.6, 0.9, steps)

    step_outputs = []
    
    with torch.no_grad():
        for step_idx in range(steps):
            t = config.T - step_idx
            if t < 1:
                t = 1
            t_batch = torch.tensor([t], device=device)
            attn_mask = (x != config.pad_id).float()
            logits = model(x, t_batch, attn_mask)
            
            for pos in range(config.seq_len):
                if not fixed[0, pos]:
                    filtered_logits = top_p_filtering(
                        logits[0, pos], 
                        config.top_p, 
                        config.temp
                    )
                    probs = F.softmax(filtered_logits, dim=-1)
                    max_prob = probs.max().item()
                    
                    if max_prob > thresholds[step_idx]:
                        token = probs.argmax().item()
                        x[0, pos] = token
                        fixed[0, pos] = True
                    else:
                        token = torch.multinomial(probs, 1).item()
                        x[0, pos] = token

            eos_pos = (x == config.eos_id).nonzero(as_tuple=True)
            if len(eos_pos[0]) > 0:
                first_eos = eos_pos[1][0].item()
                if first_eos < config.seq_len - 1:
                    x[0, first_eos+1:] = config.pad_id
                    fixed[0, first_eos+1:] = True

            if show_steps:
                current_tokens = x[0].cpu().tolist()
                try:
                    decoded = decode_tokens(current_tokens, 'sanskrit_spm.model')
                    step_outputs.append(decoded)
                    print(f"Step {step_idx + 1}/{steps} (t={t}): {decoded[:100]}...")
                    time.sleep(0.1)
                except:
                    step_outputs.append(f"Step {step_idx + 1} tokens: {current_tokens[:20]}...")

    final_tokens = x[0].cpu().tolist()
    return final_tokens, step_outputs
