#! /usr/bin/env python3

import csv
import json
from transformers import CLIPTokenizer

with open("${text}") as f:
    data = f.read()

# Create correct tokenizer class
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

# Get list of tokens
tokens = tokenizer.encode(data)

# Get mapping of tokens to strings
strings = tokenizer.convert_ids_to_tokens(tokens)
mapping = [(s, t) for t, s in zip(tokens, strings)]

# Save tokens as JSON
with open("tokens.json", "w") as f:
    json.dump(tokens, f, indent=4)

# Save mapping as CSV
with open("mapping.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["string", "token"])
    writer.writerows(mapping)