# Inference Optimizations

## KV Caching

### Measured Baseline (pre-cache, MPS, GPTMidiV1 10-epoch checkpoint)

| Metric | Value |
|---|---|
| Samples | 5 |
| Max length | 1024 tokens |
| Avg tokens generated | 890 |
| Avg total time | 57.57s |
| Avg time per token | 59.0 ms |
| Avg tokens per second | 16.9 |

Per-sample breakdown:

| Sample | Generated | Time (s) | ms/token |
|---|---|---|---|
| 1 | 1024 | 46.39 | 45.3 |
| 2 | 1024 | 85.25 | 83.3 |
| 3 | 1024 | 79.69 | 77.8 |
| 4 | 352  | 7.54  | 21.4 |
| 5 | 1024 | 68.98 | 67.4 |

Sample 4 hit EOS early (352 tokens), which is why its ms/token is lower — it spent less time in the expensive late-sequence steps where the O(n²) cost is highest. The variance across full-length samples (45–83 ms/token) reflects this: each step gets slower as the sequence grows because the model is reprocessing an ever-longer context on every forward pass.

At the target sequence length of 2048 tokens, this cost roughly quadruples. KV caching reduces it to O(n) — constant ms/token regardless of sequence length.

---

### Results (post-cache, MPS, GPTMidiV1 10-epoch checkpoint)

| Metric | Before | After | Improvement |
|---|---|---|---|
| Avg tokens generated | 890 | 843 | — |
| Avg total time | 57.57s | 4.55s | **12.7x faster** |
| Avg time per token | 59.0 ms | 6.3 ms | **9.4x faster** |
| Avg tokens per second | 16.9 | 157.6 | **9.3x faster** |

Two changes drove this:
1. **KV caching** — K/V projections reduced from O(t) per step to O(1). Past token representations are computed once and reused.
2. **Pre-allocated static buffers** — replaced `torch.cat` (which allocates a new tensor every step) with fixed `(K_buf, V_buf, length)` buffers written in-place. This eliminated 6×1024 tensor allocations per generation and gave Metal a fixed attention shape to compile a single shader for, rather than recompiling for every unique sequence length.

---

### The Problem

During autoregressive generation, the model produces one token at a time. At each step t, the current implementation passes the entire accumulated sequence to the model:

```
tokens = [s0, s1, s2, ..., s_{t-1}, s_t]  # full sequence so far
logits = model(tokens)                      # processes ALL t+1 tokens
next   = sample(logits[0, -1, :])          # only uses the LAST position's output
```

Inside each TransformerBlock, every token in the sequence gets embedded, projected into Query, Key, and Value vectors, and run through attention. This means the work done for s0..s_{t-1} is identical to the previous step — we are recomputing the same K and V vectors over and over again, throwing the results away, and recomputing them on the next step.

The cost compounds across the generation loop: at step 1 you process 2 tokens, at step 2 you process 3, and so on. Total work is proportional to `1 + 2 + 3 + ... + n = O(n²)`. At n=2048 tokens that is over 2 million token-steps of redundant projection work.

KV caching eliminates this. After computing K and V for a token, we store them. On the next step, we retrieve the cached K/V for all past tokens and only project the single new token. Total projection work becomes O(n) across the full generation loop.

---

### Why K and V, Not Q?

This is the key insight behind the optimization.

In attention, each token produces three vectors:
- **Q (Query)**: "what information am I looking for from other tokens?"
- **K (Key)**: "what information do I contain that others can query?"
- **V (Value)**: "what do I contribute when someone attends to me?"

The output at position t is: Q_t attends over K_{0..t}, takes a weighted sum of V_{0..t}. In other words, the output at position t depends on Q_t and on the K/V of every token before it.

When generating the next token, we only need the output at the final position. So we only need to compute Q for the new token. But we do need K and V for every past position — that's what we cache. The Q vectors for past tokens are never needed again once we've moved past them.

---

### Two-Phase Inference

KV caching splits generation into two distinct phases:

**Phase 1 — Prefill**

Process the seed sequence all at once, exactly as today, but also collect the K and V tensors from every layer:

```
seed = [SOS, s1, s2, ..., s_100]
logits, kv_caches = model(seed, use_cache=True)
```

After the prefill, `kv_caches` is a list of length `num_layers`. Each entry is a `(K, V)` tuple with shape `(1, num_heads, seed_len, d_head)`. This is the model's "memory" of the seed.

**Phase 2 — Decode**

For each new token, pass only the single most recent token to the model:

```
last_token = seed[:, -1:]   # shape (1, 1)

loop:
    logits, kv_caches = model(last_token, kv_caches=kv_caches, use_cache=True)
    next_token = sample(logits[0, -1, :])
    last_token = next_token.view(1, 1)
    # kv_caches grows by 1 position each step
```

The model receives a single token. Inside each TransformerBlock, it computes Q/K/V only for that one token, appends the new K/V to the stored cache, and attends over the full cache. The output is identical to the non-cached version — we are just skipping the redundant re-projection of past tokens.

---

### TransformerBlock Changes in Detail

**Current `forward(self, x)`:**

```python
Q = query_proj(x)   # (1, seq_len, d_model)
K = key_proj(x)
V = value_proj(x)

Q = Q.view(batch, seq_len, heads, d_head).transpose(1, 2)
K = K.view(...)
V = V.view(...)

cos = self.rope_cos[:, :, :seq_len, :]
sin = self.rope_sin[:, :, :seq_len, :]
Q, K = apply_rope_transformations(Q, K, cos, sin)

out = scaled_dot_product_attention(Q, K, V, is_causal=True)
```

**New `forward(self, x, kv_cache=None, use_cache=False)`:**

Decode path (`kv_cache` is a `(K_past, V_past)` tuple, `x` is a single new token):

```python
Q = query_proj(x)   # (1, 1, d_model) — only the new token
K = key_proj(x)
V = value_proj(x)

Q = Q.view(batch, 1, heads, d_head).transpose(1, 2)   # (1, heads, 1, d_head)
K = K.view(...)
V = V.view(...)

K_past, V_past = kv_cache
past_len = K_past.shape[2]   # number of tokens already in the cache

# Apply RoPE at the correct position
cos = self.rope_cos[:, :, past_len:past_len+1, :]
sin = self.rope_sin[:, :, past_len:past_len+1, :]
Q, K = apply_rope_transformations(Q, K, cos, sin)

# Grow the cache
K = torch.cat([K_past, K], dim=2)   # (1, heads, past_len+1, d_head)
V = torch.cat([V_past, V], dim=2)

# Attend — Q is (1, heads, 1, d_head), K/V are (1, heads, past_len+1, d_head)
# No causal mask: the cache is already in causal order and Q has only one position
out = scaled_dot_product_attention(Q, K, V, is_causal=False)
```

Training/prefill path (`kv_cache=None`): identical to today, zero change.

The `use_cache` flag controls the return value:
- `use_cache=False` (training default): return `x` only. The training loop is completely unaffected.
- `use_cache=True`: return `(x, (K, V))` so the caller can store and pass the cache forward.

---

### The RoPE Offset — Why It's Critical

RoPE (Rotary Position Embedding) encodes position information by rotating Q and K vectors. The rotation angle depends on the position of the token in the sequence. If token s_t is at position 5, its K vector is rotated by an angle derived from position 5. Later, when a new query at position 6 attends to that cached K, the dot product between them correctly encodes the relative distance of 1 (6 - 5 = 1). This is how the model knows how far apart two tokens are.

If we naively applied RoPE from position 0 on every decode step (by slicing `rope_cos[:, :, :1, :]`), every new token would be treated as if it were at position 0. Its Q would be rotated by the angle for position 0, while the cached K vectors were rotated for positions 1, 2, 3, and so on. The relative position encoding would be completely wrong — the model would have no idea where in the sequence it is, and coherence would break down entirely.

The fix is straightforward: `past_len = K_past.shape[2]` gives us the current position automatically — it equals the number of tokens already in the cache. We slice `rope_cos[:, :, past_len:past_len+1, :]` to get the rotation for the correct position. This works because the RoPE buffer was pre-allocated up to `max_seq_len` in `TransformerBlock.__init__`, so any valid position is already there.

---

### GPTMidiV1 Changes

Minimal — just threading the per-layer cache list through the block loop:

```python
def forward(self, x, kv_caches=None, use_cache=False):
    x = self.embedding(x)
    new_caches = []

    for i, block in enumerate(self.transformer_blocks):
        past_cache = kv_caches[i] if kv_caches is not None else None
        if use_cache:
            x, layer_cache = block(x, kv_cache=past_cache, use_cache=True)
            new_caches.append(layer_cache)
        else:
            x = block(x)   # training path: zero change

    x = self.layer_norm(x)
    logits = self.out_proj(x)

    if use_cache:
        return logits, new_caches
    return logits
```

---

### Files Changed

| File | What changes |
|---|---|
| `src/midi_gen/model/models/TransformerBlock.py` | `forward` gains `kv_cache`, `use_cache` params; RoPE slicing uses `past_len` offset; `is_causal=False` during decode |
| `src/midi_gen/model/models/GPTMidiV1.py` | `forward` gains `kv_caches`, `use_cache` params; threads cache list through block loop |
| `src/midi_gen/model/inference/base_inference.py` | `create_sample_tokens` runs prefill pass, then decode loop passing single tokens |
| `src/midi_gen/model/training/training_loop.py` | **unchanged** |
| `src/midi_gen/model/training/positional_encodings.py` | **unchanged** |
