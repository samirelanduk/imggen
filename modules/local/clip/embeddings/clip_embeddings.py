#! /usr/bin/env python3

import argparse
import json
import torch
import sys
import safetensors

if __name__ == "__main__":
    # Get path to tokens and to model
    parser = argparse.ArgumentParser(description="Compute CLIP embeddings using given model and token files.")
    parser.add_argument("tokens_path", type=str, help="Path to the tokens file (JSON)")
    parser.add_argument("model_path", type=str, help="Path to the model file (.safetensors)")
    args = parser.parse_args()
    tokens_path = args.tokens_path
    model_path = args.model_path

    # Load tokens
    with open(tokens_path, "r") as f:
        tokens = json.load(f)
    tokens_tensor = torch.tensor(tokens)

    # Load tensors
    tensors = {"token_embedding": None, "position_embedding": None}
    with safetensors.safe_open(model_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.endswith("token_embedding.weight"):
                tensors["token_embedding"] = f.get_tensor(key)
            elif key.endswith("position_embedding.weight"):
                tensors["position_embedding"] = f.get_tensor(key)
    if any(value is None for value in tensors.values()):
        print("Missing tensors")
        sys.exit(1)

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
