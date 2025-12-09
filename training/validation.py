import torch
from tqdm import tqdm
from training.masking import forward_mask
def validate_epoch(model, dataloader, device, config) -> float:
    model.eval()
    val_correct = 0
    val_masked = 0
    with torch.no_grad():
        for x0, real_lens in tqdm(dataloader, desc="Validating"):
            x0 = x0.to(device)
            batch_size = x0.size(0)
            t = torch.full((batch_size,), config.T // 2, device=device)
            # Apply forward masking
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

            xt = torch.stack(xt_list).to(device)
            mask = torch.stack(mask_list).to(device)
            attn_mask = (x0 != config.pad_id).float()
            # Model prediction
            logits = model(xt, t, attn_mask)
            # Compute accuracy on masked positions
            loss_mask = mask & (x0 != config.pad_id)
            if loss_mask.sum() > 0:
                preds = logits[loss_mask].argmax(-1)
                targets = x0[loss_mask]
                val_correct += (preds == targets).sum().item()
                val_masked += loss_mask.sum().item()

    val_acc = val_correct / max(1, val_masked)
    return val_acc