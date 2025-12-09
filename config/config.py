from .special_tokens import SpecialTokens
class Config:
    seq_len = 400     # Maximum sequence length
    vocab_size = 24000
    pad_id = SpecialTokens.PAD_ID
    bos_id = SpecialTokens.BOS_ID
    eos_id = SpecialTokens.EOS_ID
    mask_id = SpecialTokens.MASK_ID
    d_model = 768      
    n_layers = 6       # Number of transformer layers
    n_heads = 12       # Number of attention heads
    d_ff = 3072        # Feed-forward dimension (usually 4 * d_model)
    dropout = 0.1
    T = 20             # Number of diffusion timesteps
    mask_start = 0.1   # Initial masking ratio (at t=1)
    mask_end = 0.8
    batch_size = 64    
    lr = 3e-4          
    weight_decay = 0.01 # L2 regularization weight
    warmup_steps = 500 
    total_steps = 20000 # Total training steps
    grad_clip = 1.0    # Gradient clipping threshold
    
    top_p = 0.9        
    temp = 1.0 