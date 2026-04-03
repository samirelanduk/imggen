import json
import csv
import torch
import safetensors
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

def embed(tokens_path: str, model_path: str, output_path: str = "embeddings.pt") -> None:
    """Takes a set of tokens and maps them to the correct embedding vectors for
    this model."""

    with open(tokens_path) as f:
        tokens = json.load(f)
    tokens_tensor = torch.tensor(tokens)
    with safetensors.safe_open(model_path, framework="pt", device="cpu") as f:
        token_vectors = _create_token_embedding(f, tokens_tensor)
        position_vectors = _create_position_embedding(f, tokens_tensor)
    embedding = token_vectors + position_vectors
    torch.save(embedding, output_path)
    
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

def _create_token_embedding(
    tensors: safetensors.safe_open,
    tokens_tensor: torch.Tensor,
) -> torch.Tensor:
    """Finds the correct token embedding tensor in a model, and runs the tokens
    through it."""

    for key in tensors.keys():
        if key.endswith("token_embedding.weight"):
            tensor = tensors.get_tensor(key)
            token_embedding = torch.nn.Embedding(
                num_embeddings=tensor.shape[0],
                embedding_dim=tensor.shape[1]
            ).from_pretrained(tensor)
            return token_embedding(tokens_tensor)
    else:
        raise ValueError("Token embedding tensor not found in model")

def _create_position_embedding(
    tensors: safetensors.safe_open,
    tokens_tensor: torch.Tensor,
) -> torch.Tensor:
    """Finds the correct position embedding tensor in a model, and runs each of
    the positions from 0 to whatever the maximum position is through it."""
    
    for key in tensors.keys():
        if key.endswith("position_embedding.weight"):
            tensor = tensors.get_tensor(key)
            position_embedding = torch.nn.Embedding(
                num_embeddings=tensor.shape[0],
                embedding_dim=tensor.shape[1]
            ).from_pretrained(tensor)
            return position_embedding(torch.arange(tokens_tensor.shape[1]))
    else:
        raise ValueError("Position embedding tensor not found in model")

