import argparse
import torch

# Get paths
parser = argparse.ArgumentParser(description="Denoise a latent image using a model.")
parser.add_argument("latent", type=str, help="Path to the latent image file (.pt)")
parser.add_argument("conditioning", type=str, help="Path to the conditioning file (.pt)")
parser.add_argument("model", type=str, help="Path to the model file (.safetensors)")
args = parser.parse_args()
latent = args.latent
conditioning = args.conditioning
model = args.model

# Load image
latent = torch.load(latent)

# Save final image
torch.save(latent, "denoised.pt")