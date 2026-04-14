# Iteration Log — MAESTRO

> **Dataset retired.** After Round 1 and planning for Round 2, we shifted away from MAESTRO v3 to the
> Lakh Clean dataset. MAESTRO is high-quality but too narrow — solo classical piano only — for the goal
> of general musical generation. See `lakh_model_iterating.md` for the continuation.
> This file is kept as a record of the MAESTRO experiments.

---

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
- `num_epochs`: 10 → 25
- `eta_min` in cosine schedule: `1e-5` → `1e-6` (more room to decay; final epochs remain useful)
- `seq_length`: 1024 → 2048 (dataset must be re-tokenized and re-uploaded to Kaggle)
- `max_seq_len=2048` passed to `GPTMidiV1` and through to `TransformerBlock` (RoPE buffer pre-allocated at this size)

### Motivation
**More epochs:** Loss curve had no plateau at epoch 10, model clearly underfit. 25 epochs is estimated to fit within the 12-hour Kaggle limit (~2x time per epoch due to longer sequences, ~5x total vs. the original run).

**Longer sequences:** 1024-token windows cover roughly 2-3 seconds of dense piano music — not enough to contain a full musical phrase (typically 4-8 bars). The model cannot learn phrase-level structure from windows that cut through phrases arbitrarily. Seeding with 100 real tokens produced zero improvement in musical coherence, confirming the model isn't learning to use long-range context. Doubling to 2048 gives the model a fighting chance to see and learn full phrases.

**Flash Attention:** Already in use, so the O(n²) memory concern for longer sequences doesn't apply. Batch size stays at 32, no gradient accumulation needed.

---

## Round 3 — Candidates (pending Round 2 results)

### Register imbalance (investigate first)
Generated output showed heavy upper-register bias: 84% of notes ≥ C5 vs. 34% in ground truth, mean pitch 76.2 vs. 65.6. However, this may be a consequence of the oscillating-note generation failure rather than a systematic model bias — the histogram just reflects whatever notes the model loops on. Re-run the pitch histogram diagnostic after Round 2. If the imbalance persists with otherwise better output, pursue register-aware tokenization (separate `<MELODY_ON/OFF>` and `<BASS_ON/OFF>` token types split at a pitch threshold, e.g. MIDI 60).

---

## Code changes made between Round 1 and Round 2

### RoPE table moved into model (`TransformerBlock`, `positional_encodings.py`)
**Problem:** `apply_rope_transformations` was calling `init_cos_sin_table` on every forward pass, recomputing the same deterministic table every step.

**Fix:** `TransformerBlock.__init__` now computes the cos/sin table once and stores it as a registered buffer (`rope_cos`, `rope_sin`). `forward` slices to the actual `seq_len`. `apply_rope_transformations` now takes precomputed cos/sin rather than building them internally. `GPTMidiV1` exposes `max_seq_len` (default 1024) and passes it through.

**Loading note:** Old checkpoints don't contain the buffer keys. Use `strict=False` when loading — the buffers are deterministic and always correctly initialised at construction time.

### top-p added to inference (`base_inference.py`, `inference/testing.py`)
Added `top_p` parameter to `_sample_next_token` and `create_sample_tokens`. Temperature is now applied once upfront before nucleus/top-k filtering to avoid double-scaling.
