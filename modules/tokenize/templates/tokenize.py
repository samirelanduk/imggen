#! /usr/bin/env python3

import json
from transformers import CLIPTokenizer

with open("${text}") as f:
    data = f.read()

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
tokens = tokenizer.encode(data)

with open("tokens.json", "w") as f:
    json.dump(tokens, f, indent=4)