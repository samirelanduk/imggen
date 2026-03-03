#! /usr/bin/env python3

import re
import sys
import argparse
import safetensors
import torch

# Get paths
parser = argparse.ArgumentParser(description="Encode embeddings using a pretrained CLIP model.")
parser.add_argument("embeddings_path", type=str, help="Path to the embeddings file (.pt)")
parser.add_argument("model_path", type=str, help="Path to the CLIP model file (.safetensors)")
args = parser.parse_args()
embeddings_path = args.embeddings_path
model_path = args.model_path

# Load tensors
tensors = {}
norm_tensors = {"weight": None, "bias": None}
with safetensors.safe_open(model_path, framework="pt", device="cpu") as f:
    for key in f.keys():
        if key.endswith("final_layer_norm.weight"):
            norm_tensors["weight"] = f.get_tensor(key)
        elif key.endswith("final_layer_norm.bias"):
            norm_tensors["bias"] = f.get_tensor(key)
        layer_number_match = re.search(r"layers\.(\d+)\.(.*)", key)
        if not layer_number_match or "cond_stage_model" not in key.lower():
            continue
        layer_number = int(layer_number_match.group(1))
        post_layer_key = layer_number_match.group(2)
        if layer_number not in tensors:
            tensors[layer_number] = {}
        tensors[layer_number][post_layer_key] = f.get_tensor(key)
for layer_number in sorted(tensors.keys()):
    if len(tensors[layer_number]) != 16:
        print(f"Layer {layer_number} has {len(tensors[layer_number])} tensors, expected 16")
        sys.exit(1)
if any(value is None for value in norm_tensors.values()):
    print("Missing norm tensors")
    sys.exit(1)

# Load embeddings
conditioning = torch.load(embeddings_path)

for layer_number in sorted(tensors.keys()):
    pass

# Perform final normalization
norm_layer = torch.nn.LayerNorm(norm_tensors["weight"].shape[0], device="cpu")
norm_layer.load_state_dict(norm_tensors)
conditioning = norm_layer(conditioning)

# Output conditioning
with open("conditioning.pt", "wb") as f:
    torch.save(conditioning, f)


