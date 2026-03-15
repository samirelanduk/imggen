import argparse
import torch

# Get paths
parser = argparse.ArgumentParser(description="Denoise a latent image using a model.")
parser.add_argument("latent", type=str, help="Path to the latent image file (.pt)")
parser.add_argument("conditioning", type=str, help="Path to the conditioning file (.pt)")
parser.add_argument("model", type=str, help="Path to the model file (.safetensors)")
parser.add_argument("--steps", type=int, default=20, help="Number of denoising steps")
args = parser.parse_args()
latent_path = args.latent
conditioning_path = args.conditioning
model_path = args.model
steps = args.steps

# Load image
latent = torch.load(latent_path)

# Denoise in steps
for step in range(1, steps + 1):
    print(f"Step {step} of {steps}")


# Save final image
torch.save(latent, "denoised.pt")