import torch

def create_gaussian_perturbation(
    num_patch_positions: int,
    hidden_size: int = 5120,
    mean: float = 0.0,
    std: float = 0.01,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Generates a Gaussian perturbation tensor of shape [num_patch_positions, hidden_size].
    
    Args:
        num_patch_positions: Number of token positions being patched.
        hidden_size: Dimension of the residual stream (5120 for Qwen3-14B).
        mean: Center of the normal distribution (default: 0.0).
        std: Standard deviation/scale of perturbation (default: 0.01).
        dtype: Model tensor precision (torch.bfloat16, torch.float16, or torch.float32).
        device: Target hardware device ('cuda' or 'cpu').
        
    Returns:
        torch.Tensor: Perturbation tensor of shape [num_patch_positions, hidden_size].
    """
    # Sample from standard normal N(0, I) in float32 for numerical stability before casting
    perturbation = torch.randn(
        num_patch_positions, 
        hidden_size, 
        dtype=torch.float32, 
        device=device
    ) * std + mean

    return perturbation.to(dtype=dtype)
