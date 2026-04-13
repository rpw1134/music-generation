import torch
import torch.nn.functional as F
from torch import nn

from midi_gen.model.training.positional_encodings import apply_rope_transformations, init_cos_sin_table


class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, num_heads=4, ff_dim_ratio=4, dropout=0.1, causal=True, max_seq_len=1024):
        super().__init__()
        self.num_heads = num_heads
        self.causal = causal

        self.d_head = d_model // num_heads
        cos, sin = init_cos_sin_table(max_seq_len, self.d_head, base=10000)  # (max_seq_len, d_head)
        self.register_buffer("rope_cos", cos.unsqueeze(0).unsqueeze(0))  # (1, 1, max_seq_len, d_head)
        self.register_buffer("rope_sin", sin.unsqueeze(0).unsqueeze(0))

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

    def forward(self, x, use_cache=False, kv_cache=None):
        # x: (batch, seq_len, d_model)
        #   training / prefill: seq_len = full sequence length
        #   decode:             seq_len = 1 (single new token)
        batch_size, seq_len, d_model = x.shape
        d_head = d_model // self.num_heads

        # Attention
        residual = x
        x = self.layer_norm_1(x)

        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)

        Q = Q.view(batch_size, seq_len, self.num_heads, d_head).transpose(1, 2)  # (batch, heads, seq_len, d_head)
        K = K.view(batch_size, seq_len, self.num_heads, d_head).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, d_head).transpose(1, 2)

        if kv_cache is not None:
            # prefill or decode: write K/V into pre-allocated buffer, attend over filled region
            K_buf, V_buf, past_len = kv_cache
            new_len = past_len + seq_len

            cos = self.rope_cos[:, :, past_len:new_len, :]
            sin = self.rope_sin[:, :, past_len:new_len, :]
            Q, K = apply_rope_transformations(Q, K, cos, sin)

            # in-place write — no allocation
            K_buf[:, :, past_len:new_len, :] = K
            V_buf[:, :, past_len:new_len, :] = V

            # causal mask only needed when processing multiple tokens (prefill)
            # for single-token decode, Q attends to all cache positions — is_causal=False is correct
            out = F.scaled_dot_product_attention(
                Q,
                K_buf[:, :, :new_len, :],
                V_buf[:, :, :new_len, :],
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=(seq_len > 1),
            )
        else:
            # training: no cache, standard full-sequence causal attention
            cos = self.rope_cos[:, :, :seq_len, :]
            sin = self.rope_sin[:, :, :seq_len, :]
            Q, K = apply_rope_transformations(Q, K, cos, sin)

            out = F.scaled_dot_product_attention(Q, K, V, dropout_p=self.dropout.p if self.training else 0.0, is_causal=self.causal)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        out = self.out_proj(out)

        x = residual + out

        # FFN (pre-norm)
        residual = x
        x = self.layer_norm_2(x)
        x = self.lin2(self.gelu(self.lin1(x)))
        x = residual + self.dropout(x)

        if use_cache:
            return x, (K_buf, V_buf, new_len)
        return x

if __name__ == "__main__":
    model = TransformerBlock()
    print(sum(p.numel() for p in model.parameters()))