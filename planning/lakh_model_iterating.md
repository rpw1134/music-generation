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

## Round 2 — Candidates (superseded by Round 2 plan below)

The Round 1 post-mortem identified four candidates: repetition penalty tuning, velocity drift
investigation, more training epochs, and more/better data. All four are addressed by the Round 2
architectural plan, which treats them together rather than piecemeal.

---

## Round 2 — Plan

### Motivation
Round 1 established that the model learns token grammar well but lacks musical structure: it loops on
short motifs, has no sense of bar or phrase, and drifts in velocity over long sequences. These are
symptoms of a single root cause — the model has no structural representation of music. It predicts
the next token from local context only, with no notion of where it is in a bar, what the rhythmic
grid looks like, or what kind of passage it is generating.

Two complementary changes address this directly:

1. **REMI+ tokenization** — add explicit bar and position tokens so the model's input sequence
   encodes musical structure rather than requiring the model to infer it from time-shift patterns.

2. **Expert description conditioning** — prepend per-bar feature tokens (note density, pitch range,
   velocity, polyphony, time signature) before each bar's notes. The model learns to generate music
   that matches its description, forcing it to reason about bar-level structure. Inspired by FIGARO
   (Rütte et al., 2022), but using expert descriptions only — the latent description pathway from
   FIGARO requires a separate VQ-VAE encoder and is not justified at this data scale.

---

### Dataset Upgrade — ADL Piano MIDI

The Lakh piano-only subset (557 files) was too small for the original 7M-param model and is clearly
insufficient for a more expressive architecture. For Round 2 we switch to the
**ADL Piano MIDI** dataset (https://github.com/lucasnfe/adl-piano-midi):

- ~9,021 piano MIDI files from the ADL dataset combined with ~2,065 files scraped from
  publicly-available sources; all de-duplicated by MD5 checksum (~11k files total).
- Piano-only by construction — no instrument filter required beyond a minimum-note threshold.
- Substantially more style diversity than Lakh piano (which was incidentally piano due to the
  GM program filter, not by curation).

This supersedes Lakh Clean as the training corpus. The existing `lakh_filter.py` and
`lakh_piano_files.txt` are retained for reference but are no longer the active data pipeline.
A new filter/scan script will be written for ADL, analogous to `lakh_filter.py`.

---

### Tokenizer Changes — REMI+ (`tokenizing_remi.py`)

The current tokenizer (`tokenizing.py`) will remain unchanged for reference. A new file
`src/midi_gen/data_management/tokenizing_remi.py` will implement the REMI+ encoding.

**New structural tokens added to the vocabulary:**

| Token | Purpose |
|---|---|
| `<BAR>` | Marks the start of a new bar. Emitted once per bar, before its notes. |
| `<POSITION_0>` … `<POSITION_15>` | Onset position within bar at 16th-note resolution (16 bins). Fixed grid regardless of time signature. |
| `<TIME_SIG_4_4>`, `<TIME_SIG_3_4>`, `<TIME_SIG_6_8>`, `<TIME_SIG_2_4>`, `<TIME_SIG_OTHER>` | Active time signature at the bar start. Exact set determined by ADL corpus survey (see exploration notebook). |

**Token sequence structure per bar (REMI+ body):**
```
<BAR> <TIME_SIG_4_4>
<POSITION_0> <VELOCITY_12> <ON_60> <ON_64> <ON_67>
<POSITION_4> <VELOCITY_10> <OFF_60> <VELOCITY_14> <ON_62>
...
```

Note: `OFF` tokens in REMI+ are placed at the position of the note-off event, not grouped with
the note-on. This keeps the sequence causal and avoids look-ahead. Velocity precedes each `ON`
event as in the original tokenizer.

**Implementation notes:**
- `pm.get_downbeats()` gives bar start times directly from pretty_midi; no manual tempo math.
- Note-to-bar assignment: `np.searchsorted(downbeats, note_start_times, side='right') - 1`.
- Position within bar: `int((t - bar_start) / bar_dur * 16)`, clamped to `[0, 15]`.
- `pm.time_signature_changes` is never empty (pretty_midi inserts 4/4 default).
- Bars with duration < 0.1s or zero notes are skipped.

**Vocab size:** ~448 + 1 (BAR) + 16 (POSITION) + 5 (TIME_SIG) = **~470 tokens** before expert
description tokens.

---

### Expert Description Tokens

Per-bar feature tokens are prepended to each bar's note sequence, between `<BAR>` and the first
`<POSITION_*>`. The model sees what a bar should contain before it generates the notes, analogous
to a language model being told the topic of the next paragraph.

**Description token groups (data-driven bin boundaries from ADL corpus):**

| Group | Tokens | Captures |
|---|---|---|
| `<DENSITY_0>` … `<DENSITY_7>` | 8 bins | Notes per second (bar busyness) |
| `<PITCH_LOW_0>` … `<PITCH_LOW_7>` | 8 bins | Lowest pitch in bar (register floor) |
| `<PITCH_HIGH_0>` … `<PITCH_HIGH_7>` | 8 bins | Highest pitch in bar (register ceiling) |
| `<VEL_MEAN_0>` … `<VEL_MEAN_7>` | 8 bins | Mean velocity (dynamics) |
| `<POLY_0>` … `<POLY_3>` | 4 bins | Mean polyphony: 1, 2, 3, 4+ simultaneous notes |

Bin boundaries will be set from percentiles of the ADL corpus (not uniform), computed in
`src/midi_gen/exploration/remi_bar_exploration.ipynb` section 9.

**Token sequence structure per bar (with description):**
```
<BAR> <TIME_SIG_4_4> <DENSITY_4> <PITCH_LOW_3> <PITCH_HIGH_5> <VEL_MEAN_6> <POLY_2>
<POSITION_0> <VELOCITY_12> <ON_60> ...
```

**Total vocab size:** ~470 + 36 (description groups) = **~506 tokens**.

At inference time, expert descriptions can be:
- **Extracted from a seed MIDI** — condition generation on the features of an existing piece.
- **Specified manually** — dial in desired density, register, and dynamics.
- **Sampled from a prior** — fit a simple per-feature distribution over the training corpus and
  sample from it, giving unconditioned generation with structural coherence.

---

### Architecture Changes — Cross-Attention for Description Conditioning

The description tokens could be handled two ways:

**Option A: Prefix tokens (simpler)** — Description tokens are prepended to the bar's note tokens
and attended to via the standard causal self-attention. No architecture change. The model sees
descriptions as part of the sequence. This is the starting point.

**Option B: Cross-attention encoder-decoder (FIGARO-style)** — A separate encoder processes the
description sequence; the decoder attends to encoder outputs via cross-attention at each layer.
Cleaner separation between description and generation, more expressive, but requires rewriting
`TransformerBlock` and `GPTMidiV1`.

**Plan:** Start with Option A (prefix tokens). If training shows the model ignores or confuses
description tokens with note tokens, move to Option B. Cross-attention changes needed for Option B:

- `TransformerBlock`: add a `CrossAttentionBlock` after self-attention (pre-norm, same d_model/heads).
  Accepts encoder hidden states as key/value. KV-cache for encoder states is static per generation
  (descriptions don't change), so no cache invalidation issue.
- `GPTMidiV1`: add a small description encoder (2-layer transformer or even just an embedding +
  mean-pool) whose output is passed to each decoder block's cross-attention.
- `GenerateRequest` schema: add `description` field (list of per-bar feature dicts or pre-encoded
  token list).

Cross-attention is deferred until Option A is evaluated.

---

### Implementation Order

1. **Explore** — `remi_bar_exploration.ipynb`: verify bar parsing, survey ADL time signatures,
   compute bin boundaries. ✓ (notebook written)
2. **Expert description extractor** — `data_management/expert_descriptions.py`: function taking a
   PrettyMIDI object → list of per-bar feature dicts + vocab for description tokens.
   ✓ (complete)
3. **REMI+ tokenizer** — `data_management/tokenizing_remi.py`: new `tokenize_file_remi()`,
   `tokenize_dataset_remi()`, updated `reconstruct_notes_remi()`. Vocab includes BAR, POSITION,
   TIME_SIG, and description tokens.
4. **ADL filter script** — `data_management/adl_filter.py`: scan ADL directory, write qualifying
   paths to `data/adl_piano_files.txt`.
5. **Tokenize ADL corpus** — produce `data/adl_tokenized_remi.npy`. Decide whether pitch
   augmentation is still needed given the larger corpus size.
6. **Train** — same GPTMidiV1 architecture with updated vocab size. Evaluate whether prefix
   conditioning (Option A) is sufficient or cross-attention (Option B) is needed.
7. **Inference updates** — `base_inference.py`: accept per-bar description list or a seed MIDI
   to extract descriptions from. Update `GenerateRequest` schema.

---

### Data Pipeline — End-to-End Flow

This section describes the full pipeline from raw MIDI files to a tokenized training array,
covering how all the new components fit together.

```
Raw MIDI files (ADL corpus, ~11k files)
          │
          ▼
  adl_filter.py
  ─────────────
  Scan directory, apply min-note threshold (≥50 notes), write qualifying paths to
  data/adl_piano_files.txt. No instrument filter needed — ADL is piano-only by construction.

          │  data/adl_piano_files.txt
          ▼
  expert_descriptions.fit_boundaries(file_list)
  ──────────────────────────────────────────────
  Load each file via pretty_midi. For every bar (via pm.get_downbeats()), extract:
    note_density, pitch_min, pitch_max, vel_mean, poly_mean
  Compute percentile-based bin edges for each feature across all bars in the corpus.
  Save to data/adl_description_boundaries.npy (or hardcode in tokenizing_remi.py).

          │  boundaries dict
          ▼
  tokenizing_remi.tokenize_dataset_remi(file_list, boundaries)
  ────────────────────────────────────────────────────────────
  For each file:
    1. pm.get_downbeats()  →  bar start times
    2. expert_descriptions.compute_bar_features(pm)  →  list[dict], one per bar
    3. expert_descriptions.bar_features_to_tokens(bar_features, vocab, boundaries)
         →  per-bar description token lists
    4. For each bar, emit the full REMI+ token sequence:
         <BAR>
         <TIME_SIG_4_4> <DENSITY_4> <PITCH_LOW_3> <PITCH_HIGH_5> <VEL_MEAN_6> <POLY_1>
         <POSITION_0> <VELOCITY_12> <ON_60> <ON_64>
         <POSITION_2> <VELOCITY_10> <OFF_60>
         <POSITION_4> <VELOCITY_14> <ON_62>
         ...
    5. Prepend <SOS>, append <EOS>.
  Concatenate all file token arrays, pad to seq_length, reshape to (N, seq_length).
  Save to data/adl_tokenized_remi.npy.

          │  data/adl_tokenized_remi.npy  shape: (N, seq_length)
          ▼
  [optional] augment_pitch()
  ───────────────────────────
  Pitch transposition still works on REMI+ tokens — ON/OFF token indices shift by semitone
  exactly as before. Description tokens (PITCH_LOW, PITCH_HIGH) also need to be re-quantized
  after shifting, or pitch augmentation is skipped and the larger raw corpus relied on instead.
  Decision deferred until corpus size is known.

          │
          ▼
  training_loop.py  (unchanged)
  ──────────────────────────────
  GPTMidiV1 with vocab_size updated to ~506. All other hyperparameters carry over from Round 1
  as the starting point; adjust after reviewing first-epoch loss.
```

**Key files and their roles:**

| File | Role |
|---|---|
| `data_management/expert_descriptions.py` | Bar feature extraction, description vocab, token conversion, boundary fitting |
| `data_management/tokenizing_remi.py` | REMI+ tokenizer: BAR/POSITION tokens + description prefix, full vocab, reconstruct |
| `data_management/adl_filter.py` | Scan ADL corpus, write `adl_piano_files.txt` |
| `exploration/remi_bar_exploration.ipynb` | Interactive exploration; run §8–9 on ADL to get time sig coverage and bin boundaries |
| `data/adl_piano_files.txt` | Corpus file list (generated) |
| `data/adl_tokenized_remi.npy` | Final training array (generated) |

**Vocab layout (REMI+):**

```
Index range    Group              Count
───────────────────────────────────────
0              <PAD>              1
1              <SOS>              1
2              <EOS>              1
3              <BAR>              1
4–8            <TIME_SIG_*>       5  (4/4 3/4 6/8 2/4 2/2 + OTHER = 6)
9–24           <POSITION_0–15>    16
25–181         <TIME_SHIFT_0–156> 157
182–309        <ON_1–128>         128
310–437        <OFF_1–128>        128
438–469        <VELOCITY_1–32>    32
470–505        Description tokens 36  (8+8+8+8+4)
───────────────────────────────────────
Total                             ~506
```

Note: exact offsets depend on final time sig count. The tokenizer keeps the layout
explicit (as `tokenizing.py` does) so index ranges can be read off at a glance.

---

## Round 2 — Revised Plan (supersedes "Implementation Order" above)

After further design review, Round 2 is expanding in scope. The original plan (Option A:
prefix tokens in self-attention) is being skipped in favour of going directly to the full
FIGARO-inspired architecture. The revised plan has three concrete phases:

1. Extend the expert description vocabulary (BAR position tokens + chord tokens)
2. Write the REMI+ tokenizer
3. New encoder-decoder architecture with cross-attention conditioning

Each phase is documented below. **Implement in this order — each phase depends on the previous.**

---

### Phase 1 — Extended Expert Description Vocabulary

The existing description tokens (DENSITY, PITCH_LOW, PITCH_HIGH, VEL_MEAN, POLY, TIME_SIG)
are kept. Two new token groups are added.

#### 1a. BAR_i — Bar position within the training window

**What it is:** A token indicating which bar-within-the-window this is. Gives the model a
sense of local position in musical time (bar 0 of this sequence, bar 3, bar 11, etc.).

**Why not absolute bar position in the piece:** Pieces vary wildly in length. The model would
see high bar numbers rarely, and absolute position would not generalise across pieces of
different lengths. Window-relative position (always starting at 0) is consistent across all
training examples.

**How to determine the range:** At seq_length=2048, how many bars fit in one window depends
on tempo and note density. A typical bar at 120 BPM in 4/4 generates roughly 40–80 REMI+
tokens (depending on note density). That gives ~25–50 bars per window in the median case.
Survey the ADL corpus to find the 99th-percentile bars-per-window count — that becomes
MAX_BARS_PER_WINDOW. All sequences will have at most that many BAR_i tokens.

**Token names:** `<BAR_0>`, `<BAR_1>`, ..., `<BAR_{MAX_BARS_PER_WINDOW - 1}>`

**Where it appears in the sequence:** As the first token in each bar's description header,
before TIME_SIG and the other feature tokens:
```
<BAR_3> <TIME_SIG_4_4> <DENSITY_4> <PITCH_LOW_3> <PITCH_HIGH_5> <VEL_MEAN_6> <POLY_2> <CHORD_G_MAJ>
<POSITION_0> <VELOCITY_12> <ON_60> ...
```

**Survey function needed:** `survey_bars_per_window(file_list, seq_length=2048) -> int`
in `expert_descriptions.py`. Tokenises a sample of ADL files with the REMI+ tokenizer,
counts how many bars each window contains, returns the 99th-percentile value.

#### 1b. CHORD tokens — Chord quality per bar

**What it is:** A token describing the dominant chord in each bar. Gives the model harmonic
context — the description tells it "this bar is over a G major chord" before it generates
the notes.

**Chord representation:** Root (pitch class) + quality. Not Roman numeral / key-relative,
because key detection from MIDI is error-prone and adds a hard dependency. Absolute pitch
class is robust.

- **Root:** C, C#, D, D#, E, F, F#, G, G#, A, A#, B — 12 values
- **Quality:** MAJ, MIN, DOM7, DIM, AUG, OTHER — 6 values
- **Special:** `<CHORD_NONE>` for empty bars

Total: 12 × 6 + 1 = **73 chord tokens**

**Detection method:** Template matching against the bar's pitch-class histogram.
For each candidate (root, quality) pair, compute a dot product of the pitch-class
histogram against the chord template (interval set). Pick the highest-scoring pair.
This is fast, requires no external library, and degrades gracefully to OTHER for
ambiguous bars. Implementation goes in `expert_descriptions.py`.

**Chord templates (intervals from root, as pitch-class sets):**

| Quality | Intervals |
|---------|-----------|
| MAJ     | {0, 4, 7} |
| MIN     | {0, 3, 7} |
| DOM7    | {0, 4, 7, 10} |
| DIM     | {0, 3, 6} |
| AUG     | {0, 4, 8} |

If the best-scoring template scores below a confidence threshold (empirically tuned on ADL),
the chord is labelled OTHER.

**Survey function needed:** Run chord detection across the ADL corpus and print the
distribution of chord types. This validates that the 5 qualities + OTHER cover the
distribution well, and that `<CHORD_NONE>` (empty bars) is rare enough to ignore.

#### 1c. Updated vocab layout

```
Index range    Group                    Count
──────────────────────────────────────────────
0              <PAD>                    1
1              <SOS>                    1
2              <EOS>                    1
3              <BAR>                    1     ← structural separator (unchanged)
4–9            <TIME_SIG_*>             6     ← 5 known + OTHER
10–25          <POSITION_0–15>          16
26–182         <TIME_SHIFT_0–156>       157
183–310        <ON_1–128>               128
311–438        <OFF_1–128>              128
439–470        <VELOCITY_1–32>          32
471–478        <DENSITY_0–7>            8     ┐
479–486        <PITCH_LOW_0–7>          8     │
487–494        <PITCH_HIGH_0–7>         8     │ description
495–502        <VEL_MEAN_0–7>           8     │ tokens
503–506        <POLY_0–3>               4     │
507–579        <CHORD_*>                73    │ (12 roots × 6 qualities + NONE)
580–6xx        <BAR_0–MAX>              ~32   ┘ (exact count from survey)
──────────────────────────────────────────────
Total                                   ~613 (exact count after surveys)
```

Exact offsets are written explicitly in `tokenizing_remi.py` so they can be read off at
a glance, following the same convention as `tokenizing.py`.

---

### Phase 2 — REMI+ Tokenizer (`tokenizing_remi.py`)

A new file, leaving `tokenizing.py` untouched for reference.

#### Token sequence structure (one bar)

```
<BAR>
<BAR_i> <TIME_SIG_4_4> <DENSITY_4> <PITCH_LOW_3> <PITCH_HIGH_5> <VEL_MEAN_6> <POLY_2> <CHORD_G_MAJ>
<POSITION_0> <VELOCITY_12> <ON_60> <ON_64> <ON_67>
<POSITION_4> <VELOCITY_10> <OFF_60> <VELOCITY_14> <ON_62>
<POSITION_8> <OFF_64> <OFF_67>
...
```

The `<BAR>` structural token is the separator the decoder learns to generate. The description
header (`<BAR_i>` through `<CHORD_*>`) is consumed by the encoder (Phase 3) and never
generated by the decoder during inference. During training, both appear in the sequence and
the loss is masked to zero on description tokens — the model is only trained to predict
note tokens and the structural `<BAR>` token.

#### OFF token placement

OFF tokens are placed at their actual note-off time (the position in the bar where the note
ends), not grouped with the note-on. This keeps the sequence causal and requires no
lookahead. A note that starts in bar 3 and ends in bar 4 emits its OFF token at the
appropriate POSITION within bar 4.

#### Functions to implement

| Function | Purpose |
|---|---|
| `build_vocab_remi()` | Returns `(vocab: dict, inverse: list)` for the full ~613-token vocabulary |
| `tokenize_file_remi(path, vocab, boundaries) -> list[int]` | Single file → flat token list |
| `tokenize_dataset_remi(file_list, vocab, boundaries, seq_length) -> np.ndarray` | Corpus → (N, seq_length) array |
| `reconstruct_notes_remi(tokens, inverse) -> list[tuple]` | Token list → (start, end, pitch, velocity) tuples |

#### Loss masking

During training the loss is computed only on non-description tokens. Description tokens are
inputs (consumed as context) not targets (things the model learns to generate). A boolean
mask is constructed from the token indices: any index in the description range is masked out
before the cross-entropy loss is summed. This is handled in `training_loop.py` — the
tokenizer exports `DESCRIPTION_TOKEN_RANGE = (start_idx, end_idx)` so the training loop can
build the mask without hardcoding offsets.

---

### Phase 3 — Encoder-Decoder Architecture

#### Overview

The existing `GPTMidiV1` decoder-only architecture is extended with a small bidirectional
encoder that processes each bar's description header. The decoder generates note tokens
using cross-attention to the encoder's output.

```
Description tokens (per bar)          Note tokens (autoregressive)
  <BAR_i> <TIME_SIG> <DENSITY>  →  Bidirectional    →  cross-attn  →  Decoder  →  logits
  <PITCH_LOW> ... <CHORD_G_MAJ>     Encoder                            (causal)
```

#### Bidirectional Encoder

- Processes the description tokens for **one bar at a time** using full (non-causal)
  self-attention. No causal mask — it can attend to all description tokens in both directions.
- Small: 2 transformer layers, same `d_model=384` and `num_heads=8` as the decoder.
  The description sequence is short (7–8 tokens per bar), so depth beyond 2 layers adds
  little and increases training cost.
- Output: a sequence of `d_model`-dimensional vectors, one per description token. The decoder
  cross-attends to this sequence.
- The encoder is run **once per bar** before generating that bar's notes. Its output is fixed
  for the duration of that bar's generation — encoder states do not change while the decoder
  is producing notes for bar N.

#### Decoder

- Same causal self-attention as `GPTMidiV1`, N layers (starting point: 4 layers, same as
  Round 1; tune after first training run).
- Each decoder layer gains a **cross-attention sub-layer** inserted between self-attention
  and the FFN, following the standard encoder-decoder layout:
  ```
  LayerNorm → Self-Attention (causal) → residual
  LayerNorm → Cross-Attention (to encoder states) → residual
  LayerNorm → FFN → residual
  ```
- Cross-attention keys and values come from the current bar's encoder output. Queries come
  from the decoder hidden states. This is identical to the T5 / BART decoder block.

#### KV-Cache Strategy

Two separate caches:

1. **Self-attention KV-cache** (existing): grows token-by-token as the decoder generates
   note tokens. Same pre-allocated buffer as before.
2. **Encoder state cache** (new): stores the encoder output for the current bar. Because
   the encoder output is fixed for the entire bar, this is computed once at the `<BAR>` token
   and reused for every note token in that bar. When the decoder emits the next `<BAR>` token,
   the encoder is re-run on the next bar's description and the cache is refreshed.

This means inference has two phases per bar:
1. **Encode**: run the bidirectional encoder on the description tokens → store encoder states.
2. **Decode**: run the causal decoder token-by-token, cross-attending to the stored states.

#### New files / changes

| File | Change |
|---|---|
| `model/models/TransformerBlock.py` | Add optional `CrossAttentionBlock` after self-attention. Controlled by `use_cross_attention=True` flag. No change to existing self-attention path. |
| `model/models/DescriptionEncoder.py` | New file. Small 2-layer bidirectional transformer. Takes a description token sequence, returns encoder hidden states `(seq_len, d_model)`. |
| `model/models/GPTMidiV2.py` | New file. Wraps `DescriptionEncoder` + N decoder `TransformerBlock`s with cross-attention. Handles the encode-once-per-bar logic during inference. |
| `model/training/training_loop.py` | Add description loss mask. Pass encoder states to decoder during forward pass. |
| `model/inference/base_inference.py` | Two-phase inference: encode description, then decode notes bar by bar. |
| `serve/schemas/generate.py` | Add optional `descriptions` field (list of per-bar feature dicts or pre-encoded tokens). |

#### Training forward pass (pseudocode)

```python
# x shape: (batch, seq_len) — full REMI+ token sequence including description tokens
# description_mask: (batch, seq_len) bool — True where token is a description token

for each bar b in sequence:
    desc_tokens = extract_description_tokens(x, b)          # (batch, 7)
    encoder_states[b] = encoder(desc_tokens)                # (batch, 7, d_model)

# Decoder forward — cross-attends to encoder_states[current_bar] at each position
logits = decoder(x, encoder_states)                         # (batch, seq_len, vocab)

# Mask loss on description tokens — we only train the model to predict note tokens
loss = cross_entropy(logits, targets, mask=~description_mask)
```

#### Why skip Option A

Option A (prefix tokens in plain self-attention) was originally the starting point. It is
simpler but has a structural problem: the causal mask prevents note tokens from attending
back to description tokens that appear earlier in the same bar once the sequence has grown
long. In a 2048-token window this is usually fine, but for longer sequences (or when
description tokens are far in the past) the conditioning signal can degrade. The encoder-
decoder design avoids this entirely — encoder states are always immediately accessible via
cross-attention regardless of sequence position.

---

### Revised Implementation Order

1. ✓ `remi_bar_exploration.ipynb` — bar parsing confirmed, `fit_boundaries()` updated to accept glob
2. ✓ `expert_descriptions.py` — bar feature extraction complete
3. Survey functions (add to `expert_descriptions.py`):
   - `survey_bars_per_window()` → determines MAX_BARS_PER_WINDOW for BAR_i vocab
   - Chord distribution survey → validates chord token taxonomy
4. `tokenizing_remi.py` — full REMI+ tokenizer with extended vocab
5. `adl_filter.py` — scan ADL, write `adl_piano_files.txt`
6. Tokenize ADL corpus → `data/adl_tokenized_remi.npy`
7. `model/models/DescriptionEncoder.py` — bidirectional encoder
8. `model/models/TransformerBlock.py` — add CrossAttentionBlock
9. `model/models/GPTMidiV2.py` — full encoder-decoder model
10. `model/training/training_loop.py` — description loss masking, encoder state passing
11. Train Round 2
12. `model/inference/base_inference.py` + schema — two-phase inference
