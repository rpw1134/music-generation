# RoPE and YaRN

## The Problem with Standard Positional Encodings

Learned absolute positional encodings assign a fixed vector to each position index. The model
learns to use these, but they generalize poorly: position 1500 was never seen during training on
1024-length sequences, so the model has no reliable representation for it.

Sinusoidal encodings (original Transformer) are deterministic but still added to the token
embedding — they don't directly interact with the attention mechanism, and extrapolation beyond
training length degrades badly.

Both approaches encode *absolute* position. What attention actually needs is *relative* position:
not "I am at step 47" but "this token is 12 steps before that one."

---

## RoPE — Rotary Position Embedding

**Core idea**: encode position by *rotating* the query and key vectors in 2D subspaces before
computing attention. The rotation angle is a function of position. Because attention scores
depend on the dot product Q·K, and rotating both Q and K by position-dependent angles causes
the dot product to depend only on the *difference* in positions, relative position is baked in
for free.

### Mathematical Motivation

Split an embedding vector of dimension `d` into `d/2` pairs. For each pair at index `k`,
define a rotation frequency:

```
θ_k = base^(-2k / d)        # base is typically 10000
```

For a token at position `m`, rotate each 2D pair of its query (and key) by angle `m * θ_k`:

```
Rotate([x1, x2], angle) = [x1·cos(angle) - x2·sin(angle),
                            x1·sin(angle) + x2·cos(angle)]
```

The dot product between a query at position `m` and a key at position `n` then becomes a
function of `(m - n)` only — the absolute positions cancel out. The model naturally learns
relative relationships without ever being told what "relative" means.

### Pseudo-algorithm

```
for each attention head:
    for each 2D pair (i, i+1) in Q and K:
        freq = base^(-2i / d_head)
        q_rotated[i], q_rotated[i+1] = rotate(Q[i], Q[i+1], pos * freq)
        k_rotated[i], k_rotated[i+1] = rotate(K[i], K[i+1], pos * freq)
    attention_score = dot(q_rotated, k_rotated) / sqrt(d_head)
```

V vectors are not rotated — position only needs to influence which tokens attend to which,
not the content of what gets aggregated.

---

## YaRN — Yet Another RoPE extensioN

**Core idea**: extend the effective context window at inference time beyond what the model was
trained on, without retraining. YaRN modifies the RoPE frequencies and attention temperature
so that positions outside the training range produce coherent (not garbage) attention patterns.

### Mathematical Motivation

Naively scaling positions beyond the training window means the model sees rotation angles it
never encountered in training. YaRN addresses this in two ways:

**1. Frequency interpolation (NTK-aware scaling)**
Different frequency components of RoPE extrapolate differently. High-frequency pairs (small
`θ_k`, high `k`) handle fine-grained local structure — they saturate quickly and break first.
Low-frequency pairs handle long-range structure — they extrapolate more gracefully. YaRN
interpolates the frequencies non-uniformly: high-frequency components are interpolated
aggressively (compressed into the trained range), low-frequency components are left mostly
alone.

```
For each frequency θ_k:
    if θ_k is in the "high frequency" band:  interpolate (scale down)
    if θ_k is in the "low frequency" band:   leave unchanged
    else:                                     ramp between the two
```

**2. Attention temperature correction**
Interpolating positions reduces the variance of attention logits, making the softmax too
"flat" (over-attending). YaRN compensates by scaling the attention scores by a temperature
factor `t > 1` derived from the interpolation ratio, sharpening the distribution back to
what the model expects.

### Pseudo-algorithm (inference only)

```
scale_factor = target_length / train_length

for each frequency θ_k:
    θ_k_yarn = interpolate(θ_k, scale_factor, high_freq_threshold, low_freq_threshold)

for each token at position m:
    effective_pos = m  # no change to positions themselves
    rotate Q and K using θ_k_yarn instead of θ_k

scale all attention logits by temperature t before softmax
```

YaRN is applied at inference only — training uses standard RoPE. This makes it a drop-in
extension: train with RoPE on 1024 tokens, generate with YaRN at 4096+ tokens.
