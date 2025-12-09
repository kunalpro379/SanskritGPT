import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Tuple

from training.masking import forward_mask
from training.loss import compute_loss


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    config,
    epoch: int
) -> Tuple[float, float]:

    model.train()  # Set model to training mode
    total_loss = 0
    total_correct = 0
    total_masked = 0
    #PROGRESS bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for x0, real_lens in pbar:
        batch_size = x0.size(0)
        x0 = x0.to(device)
        t = torch.randint(1, config.T + 1, (batch_size,), device=device)
        # Mask each sequence according to its timestep
        xt_list = []
        mask_list = []
        
        for i in range(batch_size):
            xt_i, mask_i = forward_mask(
                x0[i], 
                real_lens[i].item(), 
                t[i].item(), 
                config
            )
            xt_list.append(xt_i)
            mask_list.append(mask_i)
        
        # Stack into batches
        xt = torch.stack(xt_list).to(device)  # [B, L]
        mask = torch.stack(mask_list).to(device)  # [B, L]
        # Mark padding positions (for transformer)
        attn_mask = (x0 != config.pad_id).float()
        
        # Predict tokens at masked positions
        logits = model(xt, t, attn_mask)  # [B, L, vocab_size]
        
        loss = compute_loss(logits, x0, mask, config)
        
        # Skip if no masked positions
        if loss.item() == 0:
            continue
        # Compute accuracy Only on masked positions
        loss_mask = mask & (x0 != config.pad_id)
        if loss_mask.sum() > 0:
            logits_flat = logits[loss_mask]
            targets_flat = x0[loss_mask]
            preds = logits_flat.argmax(-1)
            correct = (preds == targets_flat).sum().item()
            total_correct += correct
            total_masked += loss_mask.sum().item()


        #Backward pass
        optimizer.zero_grad()  # Clear gradients
        loss.backward() 
         # Gradient clipping (prevent exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        # Update parameters
        optimizer.step()
        scheduler.step()  # Update learning rate
        total_loss += loss.item()
        
        # Update progress bar
        current_acc = correct / max(1, loss_mask.sum().item()) if loss_mask.sum() > 0 else 0
        pbar.set_postfix({
            'loss': loss.item(),
            'acc': current_acc,
            'lr': scheduler.get_last_lr()[0]
        })

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_correct / max(1, total_masked)
    
    return avg_loss, avg_acc

