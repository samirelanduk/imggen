#! /usr/bin/env python3

import argparse
import csv
import json
from transformers import CLIPTokenizer

MAX_LENGTH = 77

if __name__ == "__main__":
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

    # Get token IDs for markers
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id

    # Get list of tokens
    all_tokens = tokenizer.encode(data)
    if all_tokens[0] == bos:
        all_tokens.pop(0)
    if all_tokens[-1] == eos:
        all_tokens.pop(-1)

    # Break up into lists of length MAX_LENGTH - 2 (for BOS and EOS)
    tokens = [all_tokens[i:i+MAX_LENGTH-2] for i in range(0, len(all_tokens), MAX_LENGTH-2)]
    tokens = [[bos] + t + [eos] for t in tokens]
    if len(tokens[-1]) < MAX_LENGTH:
        tokens[-1].pop(-1)
        tokens[-1] += [pad] * (MAX_LENGTH - len(tokens[-1]))

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