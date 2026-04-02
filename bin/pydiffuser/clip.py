import json
import csv
from transformers import CLIPTokenizer

MAX_LENGTH = 77

def tokenize(
    text: str,
    clip_tokenizer_path: str,
    tokens_path: str = "tokens.json",
    mappings_path: str = "mappings.csv",
) -> None:
    """Tokenizes the given text as a list of lists, and outputs them to JSON.
    It also creates a human-readable mapping of token strings to token IDs."""

    tokenizer = CLIPTokenizer.from_pretrained(clip_tokenizer_path)
    tokens = _text_to_tokens(text, tokenizer)
    tokens = _break_up_tokens(tokens, tokenizer)
    mappings = _create_token_string_mapping(tokens, tokenizer)
    with open(tokens_path, "w") as f:
        json.dump(tokens, f, indent=4)
    with open(mappings_path, "w", newline="") as f:
        writer = csv.writer(f)
        for i, mapping in enumerate(mappings):
            writer.writerows(mapping)
            if i < len(mappings) - 1:
                writer.writerow([])
    
def _text_to_tokens(text: str, clip_tokenizer: CLIPTokenizer) -> list[int]:
    """Creates a list of tokens IDs from the given text. It is a single flat
    list regardless of the length of the text, and the start and end tokens are
    removed."""

    all_tokens = clip_tokenizer.encode(text)
    if all_tokens[0] == clip_tokenizer.bos_token_id:
        all_tokens.pop(0)
    if all_tokens[-1] == clip_tokenizer.eos_token_id:
        all_tokens.pop(-1)
    return all_tokens

def _break_up_tokens(
    tokens: list[int],
    clip_tokenizer: CLIPTokenizer,
    max_length: int=MAX_LENGTH,
) -> list[list[int]]:
    """Breaks a list of tokens into a list of lists of tokens, where each
    sublist is of length max_length, and the start and end tokens are added."""

    bos = clip_tokenizer.bos_token_id
    eos = clip_tokenizer.eos_token_id
    pad = clip_tokenizer.pad_token_id
    tokens = [tokens[i:i+max_length-2] for i in range(0, len(tokens), max_length-2)]
    tokens = [[bos] + t + [eos] for t in tokens]
    if len(tokens[-1]) < max_length:
        tokens[-1].pop(-1)
        tokens[-1] += [pad] * (max_length - len(tokens[-1]))
    return tokens

def _create_token_string_mapping(tokens: list[int], clip_tokenizer: CLIPTokenizer) -> list[list[tuple[str, int]]]:
    """Creates a mapping of token values to token integers."""

    mappings = []
    for sub_list in tokens:
        strings = clip_tokenizer.convert_ids_to_tokens(sub_list)
        mappings.append([(s, t) for t, s in zip(sub_list, strings)])
    return mappings