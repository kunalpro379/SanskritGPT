import torch
from config import Config
from models import TransformerEncoderDiffusion
from inference import generate_unconditional, decode_tokens
from utils import get_device, set_seed

def main():
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    config = Config()
    model = TransformerEncoderDiffusion(config).to(device)

    checkpoint = torch.load("best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokens, step_outputs = generate_unconditional(model, device, config, steps=config.T, show_steps=True)
    text = decode_tokens(tokens, "sanskrit_spm.model")
    print("="*80)
    print("Final output:")
    print(text)

if __name__ == "__main__":
    main()