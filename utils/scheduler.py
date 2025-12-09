import math
import torch.optim as optim


def create_scheduler(optimizer, config):
    def lr_lambda(step):
        # linear increase
        if step < config.warmup_steps:
            return step / config.warmup_steps
        
        # Decay phase: cosine decay
        progress = (step - config.warmup_steps) / (config.total_steps - config.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)