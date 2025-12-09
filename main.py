import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os

from config import Config
from data import SanskritDataset
from models import TransformerEncoderDiffusion
from training import train_epoch
from training.validation import validate_epoch
from inference import generate_unconditional, decode_tokens
from utils import get_device, set_seed, create_scheduler

def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_acc = checkpoint.get('val_acc', 0.0)
        train_acc = checkpoint.get('train_acc', 0.0)
        
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Checkpoint loaded successfully!")
        print(f"  Epoch: {checkpoint.get('epoch', 0)}")
        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Validation Accuracy: {best_val_acc:.4f}")
        return start_epoch, best_val_acc
    return 1, 0.0

def save_checkpoint(epoch, model, optimizer, scheduler, train_acc, val_acc, best_val_acc, config, is_best=False):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_acc': train_acc,
        'val_acc': val_acc,
        'best_val_acc': best_val_acc,
        'config': config.__dict__
    }
    
    torch.save(checkpoint, 'checkpoint_latest.pt')
    
    if is_best:
        torch.save(checkpoint, 'best_model.pt')
        print(f"  Saved best model (acc: {val_acc:.4f})")
    
    if epoch % 5 == 0:
        torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pt')

def main():

    set_seed(42)

    device = get_device()
    print(f"Using device: {device}")

    config = Config()
    full_dataset = SanskritDataset("sanskrit_dataset.npy", config)
    train_size = int(0.95 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size]
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )
    print("Initializing model...")
    model = TransformerEncoderDiffusion(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    scheduler = create_scheduler(optimizer, config)
    
    start_epoch = 1
    best_val_acc = 0.0
    
    checkpoint_path = 'checkpoint_latest.pt'
    if os.path.exists(checkpoint_path):
        start_epoch, best_val_acc = load_checkpoint(
            checkpoint_path, model, optimizer, scheduler, device
        )
    else:
        print("No checkpoint found. Starting training from scratch.")
    
    print("Starting training loop...")
    num_epochs = config.total_steps // len(train_loader) + 1
    
    for epoch in range(start_epoch, num_epochs + 1):
        train_loss, train_acc=train_epoch(
            model, train_loader, optimizer, scheduler, device,config, epoch
        )
        val_acc=validate_epoch(model, val_loader, device, config)
        print(f"\nEpoch {epoch}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Acc: {val_acc:.4f}")


        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
        
        save_checkpoint(
            epoch, model, optimizer, scheduler, 
            train_acc, val_acc, best_val_acc, config, is_best
        )

        if epoch%5==0:
            tokens, step_outputs = generate_unconditional(model, device, config, steps=config.T, show_steps=True)
            text = decode_tokens(tokens, 'sanskrit_spm.model')
            print(f"\nFinal Generated: {text}")
            print("="*80)

    print("\nFinal generation with step-by-step unmasking:")
    tokens, step_outputs = generate_unconditional(model, device, config, show_steps=True)
    text = decode_tokens(tokens, 'sanskrit_spm.model')
    print("="*80)
    print(f"\nFinal output: {text}")



if __name__ == "__main__":
    main()



