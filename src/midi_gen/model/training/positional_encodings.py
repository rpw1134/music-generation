from typing import Tuple

import torch

def init_cos_sin_table(seq_len, d_head, base):
    i = torch.arange(0, d_head, 2) # indices for angle computations
    freqs = 1.0 / (base ** (i / d_head)) # thetas for every index
    positions = torch.arange(seq_len) # 0-S indexes. essentially the m's we need for the rotations
    angles = torch.outer(positions, freqs) # gives an (m,k), one angle for each possible m,k pair
    angles = torch.cat([angles, angles], dim=-1) # expand along dim 1 to get (m, 2k) or (m, d_head)
    cos = angles.cos() # cosine angles for every theta
    sin = angles.sin() # sine angles for every theta
    return cos, sin

def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2] # if batched, keeps all dims the same except takes the elements in the last dim up to d_head//2. if not batched, just takes the first half of the vector
    x2 = x[..., x.shape[-1] // 2:] # opposite, only second half of the vector
    return torch.cat([-x2, x1], dim=-1) # reverses and negates for use in angle calculation

def apply_rope_transformations(Q: torch.Tensor, K: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to Q and K using precomputed cos/sin tables.

    Args:
        Q, K: (batch, n_heads, seq_len, d_head)
        cos, sin: (1, 1, seq_len, d_head) — sliced from the buffer stored on the model
    """
    Q_rotated = Q * cos + rotate_half(Q) * sin
    K_rotated = K * cos + rotate_half(K) * sin
    return Q_rotated, K_rotated
