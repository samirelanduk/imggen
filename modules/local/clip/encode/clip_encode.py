#! /usr/bin/env python3

import re
import sys
import argparse
import safetensors
import torch

HEADS = 12

# Get paths
parser = argparse.ArgumentParser(description="Encode embeddings using a pretrained CLIP model.")
parser.add_argument("embeddings_path", type=str, help="Path to the embeddings file (.pt)")
parser.add_argument("model_path", type=str, help="Path to the CLIP model file (.safetensors)")
args = parser.parse_args()
embeddings_path = args.embeddings_path
model_path = args.model_path

def load_tensors(model_path):
    with safetensors.safe_open(model_path, framework="pt", device="cpu") as f:
        get_tensor = lambda key: f.get_tensor(key).float()
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
                "mlp1_weight": None,
                "mlp1_bias": None,
                "mlp2_weight": None,
                "mlp2_bias": None,
                "attn_q_weight": None,
                "attn_k_weight": None,
                "attn_v_weight": None,
                "attn_out_weight": None,
                "attn_q_bias": None,
                "attn_k_bias": None,
                "attn_v_bias": None,
                "attn_out_bias": None,
            }
        for key in f.keys():
            if "cond_stage_model" not in key.lower(): continue
            layer_number_match = re.search(r"layers\.(\d+)\.(.*)", key)
            if not layer_number_match:
                if key.endswith("final_layer_norm.weight"):
                    tensors["final_norm"]["weight"] = get_tensor(key)
                elif key.endswith("final_layer_norm.bias"):
                    tensors["final_norm"]["bias"] = get_tensor(key)
                continue
            layer_number = int(layer_number_match.group(1))
            if key.endswith("norm1.weight"):
                tensors["layers"][layer_number]["norm1_weight"] = get_tensor(key)
            elif key.endswith("norm1.bias"):
                tensors["layers"][layer_number]["norm1_bias"] = get_tensor(key)
            elif key.endswith("norm2.weight"):
                tensors["layers"][layer_number]["norm2_weight"] = get_tensor(key)
            elif key.endswith("norm2.bias"):
                tensors["layers"][layer_number]["norm2_bias"] = get_tensor(key)
            elif key.endswith("mlp.fc1.weight"):
                tensors["layers"][layer_number]["mlp1_weight"] = get_tensor(key)
            elif key.endswith("mlp.fc1.bias"):
                tensors["layers"][layer_number]["mlp1_bias"] = get_tensor(key)
            elif key.endswith("mlp.fc2.weight"):
                tensors["layers"][layer_number]["mlp2_weight"] = get_tensor(key)
            elif key.endswith("mlp.fc2.bias"):
                tensors["layers"][layer_number]["mlp2_bias"] = get_tensor(key)
            elif key.endswith("attn.q_proj.weight"):
                tensors["layers"][layer_number]["attn_q_weight"] = get_tensor(key)
            elif key.endswith("attn.q_proj.bias"):
                tensors["layers"][layer_number]["attn_q_bias"] = get_tensor(key)
            elif key.endswith("attn.k_proj.weight"):
                tensors["layers"][layer_number]["attn_k_weight"] = get_tensor(key)
            elif key.endswith("attn.k_proj.bias"):
                tensors["layers"][layer_number]["attn_k_bias"] = get_tensor(key)
            elif key.endswith("attn.v_proj.weight"):
                tensors["layers"][layer_number]["attn_v_weight"] = get_tensor(key)
            elif key.endswith("attn.v_proj.bias"):
                tensors["layers"][layer_number]["attn_v_bias"] = get_tensor(key)
            elif key.endswith("attn.out_proj.weight"):
                tensors["layers"][layer_number]["attn_out_weight"] = get_tensor(key)
            elif key.endswith("attn.out_proj.bias"):
                tensors["layers"][layer_number]["attn_out_bias"] = get_tensor(key)
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


def quick_gelu(x):
    return x * torch.sigmoid(1.702 * x)

# Load tensors
tensors = load_tensors(model_path)

# Load embeddings
conditioning = torch.load(embeddings_path).float()

# Create mask
mask = torch.full(
    (conditioning.shape[1], conditioning.shape[1]),
    -torch.finfo(conditioning.dtype).max,
    device="cpu"
).triu_(1)

for i, layer_number in enumerate(sorted(tensors["layers"].keys())):
    print(f"Layer {i + 1} of {len(tensors['layers'])}...")
    # Normalization 1
    norm1_layer = torch.nn.LayerNorm(tensors["layers"][layer_number]["norm1_weight"].shape[0], device="cpu")
    norm1_layer.weight = torch.nn.Parameter(tensors["layers"][layer_number]["norm1_weight"])
    norm1_layer.bias = torch.nn.Parameter(tensors["layers"][layer_number]["norm1_bias"])
    attn_conditioning = norm1_layer(conditioning)

    # Attention Q
    attn_q_layer = torch.nn.Linear(1, 1, device="cpu")
    attn_q_layer.weight = torch.nn.Parameter(tensors["layers"][layer_number]["attn_q_weight"])
    attn_q_layer.bias = torch.nn.Parameter(tensors["layers"][layer_number]["attn_q_bias"])
    Q = attn_q_layer(attn_conditioning)

    # Attention K
    attn_k_layer = torch.nn.Linear(1, 1, device="cpu")
    attn_k_layer.weight = torch.nn.Parameter(tensors["layers"][layer_number]["attn_k_weight"])
    attn_k_layer.bias = torch.nn.Parameter(tensors["layers"][layer_number]["attn_k_bias"])
    K = attn_k_layer(attn_conditioning)

    # Attention V
    attn_v_layer = torch.nn.Linear(1, 1, device="cpu")
    attn_v_layer.weight = torch.nn.Parameter(tensors["layers"][layer_number]["attn_v_weight"])
    attn_v_layer.bias = torch.nn.Parameter(tensors["layers"][layer_number]["attn_v_bias"])
    V = attn_v_layer(attn_conditioning)

    # Apply attention
    head_dim = Q.shape[-1] // HEADS
    batch = Q.shape[0]
    seq_len = Q.shape[1]
    Q = Q.view(batch, seq_len, HEADS, head_dim).transpose(1, 2)
    K = K.view(batch, seq_len, HEADS, head_dim).transpose(1, 2)
    V = V.view(batch, seq_len, HEADS, head_dim).transpose(1, 2)
    scores = Q @ K.transpose(-2, -1) / (Q.shape[-1] ** 0.5)
    scores = scores + mask
    attn_output = torch.softmax(scores, dim=-1) @ V
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
    attn_out_layer = torch.nn.Linear(1, 1, device="cpu")
    attn_out_layer.weight = torch.nn.Parameter(tensors["layers"][layer_number]["attn_out_weight"])
    attn_out_layer.bias = torch.nn.Parameter(tensors["layers"][layer_number]["attn_out_bias"])
    attn_output = attn_out_layer(attn_output)

    # Add to conditioning
    conditioning += attn_output

    # Normalization 2
    norm2_layer = torch.nn.LayerNorm(tensors["layers"][layer_number]["norm2_weight"].shape[0], device="cpu")
    norm2_layer.weight = torch.nn.Parameter(tensors["layers"][layer_number]["norm2_weight"])
    norm2_layer.bias = torch.nn.Parameter(tensors["layers"][layer_number]["norm2_bias"])
    mlp_conditioning = norm2_layer(conditioning)

    # MLP 1
    mlp1_layer = torch.nn.Linear(1, 1, device="cpu")
    mlp1_layer.weight = torch.nn.Parameter(tensors["layers"][layer_number]["mlp1_weight"])
    mlp1_layer.bias = torch.nn.Parameter(tensors["layers"][layer_number]["mlp1_bias"])
    mlp_conditioning = mlp1_layer(mlp_conditioning)

    # Activation
    mlp_conditioning = quick_gelu(mlp_conditioning)

    # MLP 2
    mlp2_layer = torch.nn.Linear(1, 1, device="cpu")
    mlp2_layer.weight = torch.nn.Parameter(tensors["layers"][layer_number]["mlp2_weight"])
    mlp2_layer.bias = torch.nn.Parameter(tensors["layers"][layer_number]["mlp2_bias"])
    mlp_conditioning = mlp2_layer(mlp_conditioning)

    # Add to conditioning
    conditioning += mlp_conditioning

# Perform final normalization
norm_layer = torch.nn.LayerNorm(tensors["final_norm"]["weight"].shape[0], device="cpu")
norm_layer.load_state_dict(tensors["final_norm"])
conditioning = norm_layer(conditioning)

# Output conditioning
with open("conditioning.pt", "wb") as f:
    torch.save(conditioning, f)


