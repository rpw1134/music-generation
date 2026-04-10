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

def apply_rope_transformations(Q: torch.Tensor, K: torch.Tensor, base=10000) -> Tuple[torch.Tensor, torch.Tensor]:
    # Q and K are (batch, seq_len, n_heads, d_head)
    seq_len, d_head = Q.shape[1], Q.shape[2]
    # TODO: Call this in the model ONCE, not every transformation
    cos, sin = init_cos_sin_table(seq_len, d_head, base) # (seq_len, d_head)
    cos = cos.unsqueeze(0).unsqueeze(2).to(Q.device) # (1, seq_len, 1, d_head)
    sin = sin.unsqueeze(0).unsqueeze(2).to(Q.device) # (1, seq_len, 1, d_head)
    Q_rotated = Q * cos + rotate_half(Q) * sin
    K_rotated = K * cos + rotate_half(K) * sin
    return Q_rotated, K_rotated
