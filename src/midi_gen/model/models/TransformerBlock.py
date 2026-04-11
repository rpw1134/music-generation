import torch
import torch.nn.functional as F
from torch import nn

from midi_gen.model.training.positional_encodings import apply_rope_transformations


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim_ratio=4, dropout=0.1, causal=True):
        super().__init__()
        self.num_heads = num_heads
        self.causal = causal

        self.query_proj = nn.Linear(d_model, d_model, bias=False)
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)

        self.lin1 = nn.Linear(d_model, d_model * ff_dim_ratio)
        self.lin2 = nn.Linear(d_model * ff_dim_ratio, d_model)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        batch_size, seq_len, d_model = x.shape
        d_head = d_model // self.num_heads

        # Attention
        residual = x
        x = self.layer_norm_1(x)

        Q = self.query_proj(x)  # (batch, seq_len, d_model)
        K = self.key_proj(x)
        V = self.value_proj(x)

        Q = Q.view(batch_size, seq_len, self.num_heads, d_head).transpose(1, 2)  # (batch, n_heads, seq_len, d_head)
        K = K.view(batch_size, seq_len, self.num_heads, d_head).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, d_head).transpose(1, 2)

        Q, K = apply_rope_transformations(Q, K)

        scores = Q @ K.transpose(-2, -1) / (d_head ** 0.5)  # (batch, n_heads, seq_len, seq_len)

        if self.causal:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)

        out = scores @ V  # (batch, n_heads, seq_len, d_head)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        out = self.out_proj(out)

        x = residual + out

        # FFN (pre-norm)
        residual = x
        x = self.layer_norm_2(x)
        x = self.lin2(self.gelu(self.lin1(x)))
        x = residual + self.dropout(x)

        return x
