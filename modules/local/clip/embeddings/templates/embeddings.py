#! /usr/bin/env python3

import json
import torch
import sys
import safetensors.torch

model_path = "${clip_model}"
tokens_path = "${tokens}"

def get_tensors_by_key(st):
    """Safetensors files have no fixed nomenclature for the tensors, so we need
    to check a few different possibilities to extract the relebant tensors."""
    
    tensors = {
        "token_embedding": None,
        "position_embedding": None,
    }
    for key, value in st.items():
        if key.endswith("token_embedding.weight"):
            tensors["token_embedding"] = value
        elif key.endswith("position_embedding.weight"):
            tensors["position_embedding"] = value
    missing = [key for key, value in tensors.items() if value is None]
    if len(missing) > 0:
        print(f"Missing tensors: {missing}")
        sys.exit(1)
    return tensors

# Load tokens
with open(tokens_path, "r") as f:
    tokens = json.load(f)
tokens_tensor = torch.tensor(tokens)

# Open the model
device = torch.device("cpu")
model = safetensors.torch.load_file(model_path, device="cpu")

# Get relevant tensors
tensors = get_tensors_by_key(model)

# Create token embedding
token_embedding = torch.nn.Embedding(
    num_embeddings=tensors["token_embedding"].shape[0],
    embedding_dim=tensors["token_embedding"].shape[1]
).from_pretrained(tensors["token_embedding"])
token_vectors = token_embedding(tokens_tensor)

# Create position embedding
position_embedding = torch.nn.Embedding(
    num_embeddings=tensors["position_embedding"].shape[0],
    embedding_dim=tensors["position_embedding"].shape[1]
).from_pretrained(tensors["position_embedding"])
position_vectors = position_embedding(torch.arange(tokens_tensor.shape[1]))

# Create final embeddings
embeddings = token_vectors + position_vectors

# Save embeddings
torch.save(embeddings, "embeddings.pt")
