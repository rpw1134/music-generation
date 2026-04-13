# Iteration Log

## Round 1 — Initial Training Run (10 epochs)

### What was built
Full pipeline from scratch: MIDI parsing → event tokenization → GPT decoder transformer → training → autoregressive inference. Model is a 6-layer decoder-only transformer (d_model=512, 8 heads) with RoPE positional encodings, causal attention, weight-tied output projection, and pre-norm. Trained on the full MAESTRO v3 dataset (90/10 split) for 10 epochs with AdamW + linear warmup + cosine annealing, mixed precision (fp16).

### Results
- Token validity was good: ~4 decode errors per 400 notes (ON without OFF, etc.)
- Chords were musical and sounded reasonable
- No musical coherence overall — output sounded random, not like music
- Seeding with 100 real tokens from the dataset reduced token errors further but had zero effect on musical quality

### Diagnosis
Training curves showed val loss still declining steadily at epoch 10 (3.66 → 2.73), with no sign of plateauing. Train loss ≈ val loss throughout — no overfitting, clean underfitting. Final perplexity ~15 means the model is choosing between ~15 equally likely tokens at each step, which is too uncertain for coherent musical output to emerge.

The seeding result is the key signal: the model IS using context for token grammar (ON/OFF pairing improved) but is NOT using context for musical content (pitch, rhythm, phrasing unchanged). This confirms the model has learned the token structure but not musical structure — a perplexity problem, not an architectural one.

The cosine LR schedule fully annealed to `eta_min=1e-5` over 10 epochs, so continuing from the checkpoint would train at a near-dead LR. A fresh run is required.

### Inference-time experiments tried
- **top-p (nucleus) sampling at p=0.9** — no improvement on melody/coherence
- **repetition penalty** — made things worse: increased token errors and degraded sound quality. Root cause: the penalty indiscriminately penalises structural tokens (TIME_SHIFT, VELOCITY) that legitimately repeat, breaking the token grammar

---

## Round 2 — Extended Training Run (planned)

### Changes
- `num_epochs`: 10 → 30
- `eta_min` in cosine schedule: `1e-5` → `1e-6` (more room to decay; final epochs remain useful)

### Motivation
The loss curve had no plateau at epoch 10 and the model is clearly underfit. 30 epochs gives the schedule room to properly anneal and should bring perplexity down enough for musical structure to emerge.

---

## Code changes made between Round 1 and Round 2

### RoPE table moved into model (`TransformerBlock`, `positional_encodings.py`)
**Problem:** `apply_rope_transformations` was calling `init_cos_sin_table` on every forward pass, recomputing the same deterministic table every step.

**Fix:** `TransformerBlock.__init__` now computes the cos/sin table once and stores it as a registered buffer (`rope_cos`, `rope_sin`). `forward` slices to the actual `seq_len`. `apply_rope_transformations` now takes precomputed cos/sin rather than building them internally. `GPTMidiV1` exposes `max_seq_len` (default 1024) and passes it through.

**Loading note:** Old checkpoints don't contain the buffer keys. Use `strict=False` when loading — the buffers are deterministic and always correctly initialised at construction time.

### top-p added to inference (`base_inference.py`, `inference/testing.py`)
Added `top_p` parameter to `_sample_next_token` and `create_sample_tokens`. Temperature is now applied once upfront before nucleus/top-k filtering to avoid double-scaling.
