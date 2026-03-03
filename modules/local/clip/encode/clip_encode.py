#! /usr/bin/env python3

import argparse

# Get paths
parser = argparse.ArgumentParser(description="Encode embeddings using a pretrained CLIP model.")
parser.add_argument("embeddings_path", type=str, help="Path to the embeddings file (.pt)")
parser.add_argument("clip_model_path", type=str, help="Path to the CLIP model file (.safetensors)")
args = parser.parse_args()
embeddings_path = args.embeddings_path
clip_model_path = args.clip_model_path