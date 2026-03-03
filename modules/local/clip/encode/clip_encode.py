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

def load_tensors(model_path):
    with safetensors.safe_open(model_path, framework="pt", device="cpu") as f:
        tensors = {"final_norm": {"weight": None, "bias": None}, "layers": {}}
        layer_numbers = set()
        for key in f.keys():
            if "cond_stage_model" not in key.lower(): continue
            layer_number_match = re.search(r"layers\.(\d+)", key)
            if not layer_number_match: continue
            layer_number = int(layer_number_match.group(1))
            layer_numbers.add(layer_number)
        layer_numbers = sorted(layer_numbers)
        for layer_number in layer_numbers:
            tensors["layers"][layer_number] = {
                "norm1_weight": None,
                "norm1_bias": None,
                "norm2_weight": None,
                "norm2_bias": None,
            }
        for key in f.keys():
            if "cond_stage_model" not in key.lower(): continue
            layer_number_match = re.search(r"layers\.(\d+)\.(.*)", key)
            if not layer_number_match:
                if key.endswith("final_layer_norm.weight"):
                    tensors["final_norm"]["weight"] = f.get_tensor(key)
                elif key.endswith("final_layer_norm.bias"):
                    tensors["final_norm"]["bias"] = f.get_tensor(key)
                continue
            layer_number = int(layer_number_match.group(1))
            if key.endswith("norm1.weight"):
                tensors["layers"][layer_number]["norm1_weight"] = f.get_tensor(key)
            elif key.endswith("norm1.bias"):
                tensors["layers"][layer_number]["norm1_bias"] = f.get_tensor(key)
            elif key.endswith("norm2.weight"):
                tensors["layers"][layer_number]["norm2_weight"] = f.get_tensor(key)
            elif key.endswith("norm2.bias"):
                tensors["layers"][layer_number]["norm2_bias"] = f.get_tensor(key)
        for layer_number in layer_numbers:
            for key in tensors["layers"][layer_number]:
                if tensors["layers"][layer_number][key] is None:
                    print(f"Layer {layer_number} {key} is None")
                    sys.exit(1)
        for key in tensors["final_norm"]:
            if tensors["final_norm"][key] is None:
                print(f"Final norm {key} is None")
                sys.exit(1)
        return tensors

# Load tensors
tensors = load_tensors(model_path)

# Load embeddings
conditioning = torch.load(embeddings_path)

for i, layer_number in enumerate(sorted(tensors["layers"].keys())):
    print(f"Layer {i + 1} of {len(tensors['layers'])}...")
    # Normalization 1
    norm1_layer = torch.nn.LayerNorm(tensors["layers"][layer_number]["norm1_weight"].shape[0], device="cpu")
    norm1_layer.weight = torch.nn.Parameter(tensors["layers"][layer_number]["norm1_weight"])
    norm1_layer.bias = torch.nn.Parameter(tensors["layers"][layer_number]["norm1_bias"])
    conditioning = norm1_layer(conditioning)

    # Attention

    # Normalization 2
    norm2_layer = torch.nn.LayerNorm(tensors["layers"][layer_number]["norm2_weight"].shape[0], device="cpu")
    norm2_layer.weight = torch.nn.Parameter(tensors["layers"][layer_number]["norm2_weight"])
    norm2_layer.bias = torch.nn.Parameter(tensors["layers"][layer_number]["norm2_bias"])
    conditioning = norm2_layer(conditioning)

    # NN
    pass

# Perform final normalization
norm_layer = torch.nn.LayerNorm(tensors["final_norm"]["weight"].shape[0], device="cpu")
norm_layer.load_state_dict(tensors["final_norm"])
conditioning = norm_layer(conditioning)

# Output conditioning
with open("conditioning.pt", "wb") as f:
    torch.save(conditioning, f)


