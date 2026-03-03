#! /usr/bin/env python3

import argparse
import csv
import json
from transformers import CLIPTokenizer

MAX_LENGTH = 77

# Get paths
parser = argparse.ArgumentParser(description="Compute CLIP embeddings using given model and token files.")
parser.add_argument("text_path", type=str, help="Path to the text file (txt)")
parser.add_argument("clip_tokenizer_path", type=str, help="Path to the CLIP tokenizer directory")
args = parser.parse_args()
text_path = args.text_path
clip_tokenizer_path = args.clip_tokenizer_path

# Get text
with open(text_path) as f:
    data = f.read()

# Create correct tokenizer class
tokenizer = CLIPTokenizer.from_pretrained(clip_tokenizer_path)

# Get list of tokens
all_tokens = tokenizer.encode(data)

# Break up into lists of length MAX_LENGTH
tokens = [all_tokens[i:i+MAX_LENGTH] for i in range(0, len(all_tokens), MAX_LENGTH)]
if len(tokens[-1]) < MAX_LENGTH:
    tokens[-1] += [tokenizer.pad_token_id] * (MAX_LENGTH - len(tokens[-1]))

# Get mapping of tokens to strings
mappings = []
for sub_list in tokens:
    strings = tokenizer.convert_ids_to_tokens(sub_list)
    mappings.append([(s, t) for t, s in zip(sub_list, strings)])

# Save tokens as JSON
with open("tokens.json", "w") as f:
    json.dump(tokens, f, indent=4)

# Save mapping as CSV
with open("mapping.csv", "w", newline="") as f:
    writer = csv.writer(f)
    for i, mapping in enumerate(mappings):
        writer.writerows(mapping)
        if i < len(mappings) - 1:
            writer.writerow([])