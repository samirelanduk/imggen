import torch

def save_blank_latent(path: str, width: int, height: int):
    """Creates a blank latent tensor of zeros with the given width and
    height."""

    latent = torch.zeros(1, 4, height // 8, width // 8)
    torch.save(latent, path)