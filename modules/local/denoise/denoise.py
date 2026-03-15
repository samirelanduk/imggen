import argparse
import safetensors
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

# Load model
with safetensors.safe_open(model_path, framework="pt", device="cpu") as f:
    tensors = {
        "blocks": {
            "input": {},
            "middle": {},
            "output": {},
        }
    }
    for key in f.keys():
        if key.startswith("model.diffusion_model."):
            name = key[len("model.diffusion_model."):]
            if name.startswith("input_blocks."):
                name = name[len("input_blocks."):]
                tensors["blocks"]["input"][name] = f.get_tensor(key)
            elif name.startswith("middle_block."):
                name = name[len("middle_block."):]
                tensors["blocks"]["middle"][name] = f.get_tensor(key)
            elif name.startswith("output_blocks."):
                name = name[len("output_blocks."):]
                tensors["blocks"]["output"][name] = f.get_tensor(key)


# Load image
latent = torch.load(latent_path)

# Denoise in steps
for step in range(1, steps + 1):
    print(f"Step {step} of {steps}")


# Save final image
torch.save(latent, "denoised.pt")