# Iteration Log — Lakh Clean

Continuation from `maestro_model_iterating.md`. Architecture and training code carry over from the
MAESTRO work; this log tracks changes made specifically for the Lakh dataset and any new experiments.

---

## Dataset Decision — Lakh Clean (piano-only subset)

### Why we switched
MAESTRO v3 is ~200 hours of solo classical piano performed at a professional level — high quality, but
a single genre and instrument. After Round 1 the model learned token grammar (valid ON/OFF pairs) but
showed no musical coherence. The dataset was identified as a contributing constraint: the model has no
exposure to diverse rhythmic, harmonic, or structural patterns beyond classical phrasing.

Lakh Clean (~45k files, multi-genre: pop, rock, jazz, electronic, folk) was recommended as a better
fit for the goal of general musical generation.

### Piano-only filter
Lakh files are predominantly multi-instrument. The current tokenizer has no instrument identity — all
notes are treated as the same voice. Training on mixed-instrument files would produce dense, incoherent
token sequences the model cannot learn from.

**Decision:** Filter to files where every instrument is a GM piano-family program (0–7) and no drum
tracks are present (`is_drum=False`). This keeps the tokenizer unchanged and avoids instrument
confusion. A minimum of 50 notes per file is required to exclude near-empty files.

**Outcome:** 557 files scanned, 557 qualifying. Smaller than MAESTRO in file count but more diverse in style.

**Output rendered as grand piano (program 0) regardless of source instrument.** The tokenizer produces
no instrument tokens, so inference output is always reconstructed as Acoustic Grand Piano. This is
intentional for now.

### Filter implementation
`src/midi_gen/data_management/lakh_filter.py` — scans a directory, writes qualifying paths to a text
file. Run with:
```
uv run python -m midi_gen.data_management.lakh_filter --src data/lakh_clean --out data/lakh_piano_files.txt
```

---

## Dataset Statistics

`src/midi_gen/exploration/lakh_piano_stats.ipynb` — full results:

| Stat | Mean | Median | Min | Max | p95 |
|---|---|---|---|---|---|
| Duration (s) | 219.2 | 199.1 | 31.4 | 2290.6 | 394.5 |
| Note count | 1997 | 1612 | 193 | 18054 | 4202 |
| Density (notes/s) | 9.1 | 8.4 | 1.6 | 32.4 | 16.4 |
| Token estimate | 7986 | 6447 | 771 | 72215 | 16805 |
| Pitch mean | 62.6 | 62.9 | 47.0 | 79.2 | 68.1 |
| Note dur mean (s) | 0.4 | 0.3 | 0.1 | 3.1 | 0.7 |
| Gap mean (s) | 0.1 | 0.1 | 0.0 | 0.6 | 0.2 |
| Gap max (s) | 1.8 | 1.4 | 0.2 | 15.2 | 4.0 |
| Large gap % (>1s) | 0.6 | 0.1 | 0.0 | 11.7 | 2.6 |
| Poly mean | 3.6 | 3.2 | 1.0 | 16.6 | 5.5 |
| Poly max | 8 | 8 | 2 | 29 | 13 |

**Dataset totals:** 33.9 hours, ~4.45M tokens, **2,327 sequences @ seq_length=2048**

### Key findings
- **TIME_SHIFT bins are fine** — mean gap 0.1s, large gaps (>1s) only 0.6% on average; existing 0.01s–1.0s log-scale bins cover the distribution well
- **Pitch is well-centered** — mean ~62.6 (near middle C), no upper-register bias seen in MAESTRO
- **2,327 sequences is too small to train on directly** — addressed with pitch transposition augmentation

---

## Pitch Transposition Augmentation

2,327 sequences is too small (~62M tokens) for a 20M parameter model. Pitch transposition is
musically lossless — shifting all notes by N semitones preserves every interval, chord, rhythm, and
phrase. The piece sounds identical in a different key.

**Implementation:** `augment_pitch()` in `tokenizing.py`. Operates on the already-tokenized `(N, seq_len)`
array. For each shift, sequences where any note would fall outside MIDI pitch range [1, 128] are
skipped (no clamping). Default: ±6 semitones (12 shifts).

**Result:** 2,327 → 30,251 sequences (13x increase). Saved to `data/lakh_tokenized_augmented.npy`.

The original unaugmented dataset is always the first block — slice `arr[:2327]` to recover it.

---

## Round 1 — First Lakh Training Run ✓

### Changes from MAESTRO
- **Dataset:** MAESTRO v3 → Lakh Clean piano-only, pitch-augmented (`lakh_tokenized_augmented.npy`)
- **Model size reduced:** 6L/d=512/8h (~20M params) → 4L/d=384/8h (~7M params)
  - Motivation: 62M unique tokens is too small for 20M params (Chinchilla: ~3M optimal).
    7M is a safer fit while still having capacity for musical structure.
- **seq_length, all other hyperparameters:** unchanged (2048, AdamW lr=3e-4, 25 epochs, warmup 200 steps)
- **Checkpoint:** `lakh_piano_v1_best.pt`
- **Hardware:** 2× GPU (DataParallel), 851 steps/epoch

### Results

| Epoch | Train Loss | Val Loss | Accuracy | Perplexity |
|---|---|---|---|---|
| 1  | 20.03 | 3.34 | 0.212 | 28.13 |
| 5  | 2.72  | 2.62 | 0.323 | 13.80 |
| 10 | 2.45  | 2.39 | 0.366 | 10.95 |
| 15 | 2.30  | 2.26 | —     | —     |
| 20 | 2.25  | 2.21 | 0.400 | 9.15  |
| 25 | 2.23  | 2.20 | 0.403 | 9.04  |

**vs. MAESTRO Round 1 (epoch 10):** val loss 2.73 → 2.20, perplexity ~15 → ~9. Meaningful improvement.

### Diagnosis
- **Train loss ≈ val loss throughout** — no overfitting. Clean generalization, model is still underfitting slightly.
- **Loss still slowly declining at epoch 25** (epochs 20–25: 2.21→2.20, ~0.001/epoch). Nearly plateaued. More epochs would yield diminishing returns.
- **Perplexity 9** — model chooses between ~9 equally likely tokens at each step. Better than MAESTRO's ~15, but musical coherence still needs work.
- **With temp=1.1, top_p=0.9:** produces 2048-token samples with relatively few decode errors. Seeding helps.

### Observed output quality
- Token grammar: good — few ON/OFF errors
- Pitch distribution: well-centered, no upper-register bias seen in MAESTRO
- **Issues:**
  - Lack of rhythmic consistency
  - Velocity drift — samples become too quiet over time
  - Gets stuck in repeating patterns
  - Limited long-range musical structure

---

## Round 2 — Candidates

### Repetition / stuck patterns
The model loops on short motifs rather than developing phrases. Possible approaches:
- **Repetition penalty scoped to pitch tokens only** — MAESTRO experiments showed blanket repetition penalty broke token grammar by penalising structural TIME_SHIFT/VELOCITY tokens. A pitch-only penalty (applied only to ON tokens) avoids this. Worth trying at inference time first (no retraining needed).
- **More data / more diverse data** — only 557 source files. Relaxing the piano-only filter to include near-piano programs or extracting piano tracks from multi-instrument files would increase diversity significantly.

### Velocity drift
Generated samples trend quieter over time. The model may be learning a bias toward low velocity bins from the Lakh data, or the autoregressive distribution drifts as generation length increases. Investigate:
- Velocity histogram of generated samples vs. ground truth (add to `stats.py`)
- If systematic bias: consider velocity-aware sampling or a post-processing normalisation pass

### More training
Loss was still declining at epoch 25. Diminishing returns expected, but running to 40 epochs costs little on Kaggle and may push perplexity below 8.

### Larger model / more data
If Round 2 experiments don't resolve musical coherence, the next meaningful step is more data — either relaxing the piano-only filter or expanding to multi-instrument with instrument tokens added to the vocabulary.
