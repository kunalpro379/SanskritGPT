# Sanskrit Masked Diffusion Model(Transformer Denoiser)
![Generation Animation](generation_1.gif)

## Training Process

The training follows a masked diffusion approach for Sanskrit text generation:

1. Load the preprocessed Sanskrit dataset from numpy array format
2. Split the dataset into training and validation sets with 95-5 ratio
3. Initialize the TransformerEncoderDiffusion model with configurable parameters
4. Set up AdamW optimizer with learning rate and weight decay
5. Create a learning rate scheduler with warmup steps
6. Load checkpoint if available to resume training from previous state
7. For each epoch, process batches of Sanskrit sequences
8. Randomly sample diffusion timesteps for each sequence in the batch
9. Apply forward masking where tokens are masked based on timestep and mask ratio
10. Pass masked sequences through the transformer encoder model
11. Compute cross-entropy loss only on masked token positions
12. Calculate accuracy by comparing predicted tokens with original tokens at masked positions
13. Perform backpropagation with gradient clipping to prevent exploding gradients
14. Update model parameters and learning rate scheduler
15. Validate the model on validation set after each training epoch
16. Save checkpoints periodically and save best model based on validation accuracy
17. Generate sample texts every 5 epochs to monitor training progress
18. Use the trained model for unconditional Sanskrit text generation through iterative unmasking

## Running Training

```bash
python main.py
```

## Running Inference

```bash
python infer.py
```

