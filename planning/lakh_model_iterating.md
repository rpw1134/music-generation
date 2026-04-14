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

**Outcome:** ~557 files scanned, ~600 qualifying (exact count pending notebook run). This is a smaller
dataset than MAESTRO in file count but more diverse in style.

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

## Dataset Statistics (pending notebook results)

`src/midi_gen/exploration/lakh_piano_stats.ipynb` — scans the file list and computes per-file stats
(duration, note count, density, pitch distribution, polyphony, time gaps, token estimates).

Key questions to answer before tokenizing:
- Total token count and number of 2048-token sequences available for training
- Whether the existing TIME_SHIFT bin range (0.01s–1.0s, 157 bins) covers the gap distribution
- Average polyphony — informs whether the model needs to handle denser simultaneous-note patterns than MAESTRO
- How many files fall below 2048 tokens (less than one full training sequence)

*Update this section once the notebook has run.*

---

## Round 1 — First Lakh Training Run (planned)

### Changes from MAESTRO Round 2 plan
- Dataset: MAESTRO v3 → Lakh Clean piano-only
- Tokenization: re-run `tokenize_dataset` over the new file list → `data/lakh_tokenized_dataset.npy`
- Architecture, training hyperparameters, and seq_length (2048) carry over unchanged pending stats review

### Open questions before training
- Does the token estimate per file suggest the dataset produces enough sequences to train on, or do we
  need to lower the minimum-notes threshold or reconsider seq_length?
- Are there tokenizer parameters (TIME_SHIFT bins, velocity bins) that should be adjusted given the
  different musical style of the Lakh files vs. MAESTRO?

*Update with results once training completes.*
