import torch

def linear(weight: torch.Tensor, bias: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
    """Applies a linear transformation to the incoming data. The weight must be
    a 2D tensor of dimensions (output_dim, input_dim), and the bias must be a 1D
    tensor of dimensions (output_dim)."""
    
    layer = torch.nn.Linear(weight.shape[0], weight.shape[1], device="cpu")
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    layer.bias = torch.nn.Parameter(bias, requires_grad=False)
    return layer(input)