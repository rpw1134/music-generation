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

## Round 2 — Revised Plan (supersedes all sections above from "Implementation Order" onward)

---

### Architecture — Encoder-Decoder with Cross-Attention

This is not a flat decoder-only transformer. It has two separate components that process
two separate inputs. Here is the high-level picture:

```
  ENCODER INPUT                          DECODER INPUT / OUTPUT
  (description tokens, per bar)          (note tokens, autoregressive)

  <TIME_SIG_4_4>                         <BAR>
  <DENSITY_4>          ──► Bidirectional  <POSITION_0>
  <PITCH_LOW_3>            Encoder    ──► <VELOCITY_12>  ◄── cross-attention
  <PITCH_HIGH_5>           (no causal     <ON_60>             at every layer
  <VEL_MEAN_6>              mask)         <ON_64>
  <POLY_2>                               <POSITION_4>
  <CHORD_C_MAJ>                          <OFF_60>
                                         ...
                                         next <BAR> → re-encode next bar's description
```

The encoder and decoder are completely separate. Description tokens never appear in the
decoder's input. Note tokens never appear in the encoder's input. The only connection
between them is the cross-attention mechanism inside each decoder layer.

---

### The Encoder

The encoder is a small bidirectional transformer. "Bidirectional" means it has no causal
mask — every description token can attend to every other description token in both directions.
This is appropriate because the description for a bar is a fixed, complete set of facts
(density, pitch range, chord, etc.) that are all known at the same time. There is no
sequential dependency between them that would require causal ordering.

- Input: the 7 description tokens for one bar (TIME_SIG, DENSITY, PITCH_LOW, PITCH_HIGH,
  VEL_MEAN, POLY, CHORD)
- Architecture: 2 transformer layers, `d_model=384`, 8 heads — same dimensions as the decoder
- Output: 7 hidden state vectors of size `d_model`, one per input token. These are the
  "encoded description" that the decoder will cross-attend to.

The encoder is run **once per bar**, before the decoder starts generating that bar's notes.
Its output is fixed and does not change while the decoder generates.

---

### The Decoder

The decoder is a causal transformer — identical in structure to GPTMidiV1, but with one
extra sub-layer added to each transformer block: cross-attention to the encoder's output.

Each decoder layer now has three sub-layers instead of two:

```
  (before)                      (after)
  ─────────────────────         ──────────────────────────────────────
  LayerNorm                     LayerNorm
  Self-Attention (causal)       Self-Attention (causal)
  + residual                    + residual
                                LayerNorm
  LayerNorm                     Cross-Attention → encoder hidden states
  FFN                           + residual
  + residual
                                LayerNorm
                                FFN
                                + residual
```

The self-attention sub-layer is causal — note token at position i can only attend to
positions 0 through i. This is required for autoregressive generation (you can't look ahead).

The cross-attention sub-layer has no causal restriction — every note token can attend to
all 7 encoder hidden states freely. The encoder hidden states are the keys and values;
the decoder hidden state at the current position is the query. This is how the decoder
reads the description: it asks "given what I'm generating right now, what parts of this
bar's description are relevant?"

---

### Why not just put description tokens in the decoder sequence?

If description tokens were in the decoder's input sequence (the flat approach), the causal
mask would still work — note tokens come after description tokens, so they can attend back
to them. But there are two problems:

1. **The description tokens themselves would need to be predicted.** At every position the
   model produces a distribution over the full vocabulary. If description tokens are in the
   sequence, the model is being asked to predict them as well as note tokens, which wastes
   capacity and pollutes the loss signal (even if you mask them out, they still occupy
   attention bandwidth as the sequence grows longer).

2. **The description tokens get processed causally, which is wrong.** In a flat sequence,
   when the model reads `<DENSITY_4>` it has not yet seen `<CHORD_C_MAJ>` (which comes
   later in the description). A causal mask prevents later description tokens from informing
   earlier ones. The encoder has no such restriction — all 7 description tokens see each
   other fully, producing a richer representation of the bar's character.

The cross-attention design cleanly separates these two concerns: the encoder reads the full
description with full attention, and the decoder generates notes with access to that full
representation at every step.

---

### Training: what gets stored and how the forward pass works

**Dataset format**

The tokenizer produces two parallel arrays per training window, not one flat array:

```
note_tokens:  (N, seq_len)          — decoder input/output
              BAR, POSITION_*, VELOCITY_*, ON_*, OFF_* tokens only
              padded to seq_len=2048, sliced at BAR boundaries

desc_tokens:  (N, max_bars, 7)      — encoder input
              one row of 7 description tokens per bar in the window
              max_bars = maximum bars that fit in seq_len (empirically determined)

bar_map:      (N, seq_len)          — integer array
              bar_map[i, t] = which bar index the note token at position t belongs to
              used to route cross-attention: token at position t attends to
              encoder_states[bar_map[i, t]]
```

These three arrays are saved together and loaded as a unit by the dataset class.

**Forward pass**

```python
# desc_tokens: (batch, max_bars, 7)
# note_tokens: (batch, seq_len)
# bar_map:     (batch, seq_len)

# Step 1 — encode all bars in the batch at once
# reshape to (batch * max_bars, 7), encode, reshape back
encoder_states = encoder(desc_tokens)          # (batch, max_bars, 7, d_model)

# Step 2 — for each decoder position, gather the right encoder states
# bar_map tells us which bar index to use at each position
# result: (batch, seq_len, 7, d_model) — each token gets its bar's encoder states
per_position_states = encoder_states[bar_map]  # indexed by bar_map

# Step 3 — decoder forward, cross-attending at each layer to per_position_states
logits = decoder(note_tokens, per_position_states)   # (batch, seq_len, decoder_vocab_size)

# Step 4 — loss on note tokens only (BAR counts as a note token here)
loss = cross_entropy(logits, targets)
```

There is no loss mask needed because description tokens are never in the decoder's input
or output — they only exist in `desc_tokens` which goes to the encoder.

---

### Separate vocabularies

Because the encoder and decoder process different token types, they have separate
vocabularies and separate embedding tables.

**Encoder vocabulary** (~42 tokens):
```
TIME_SIG_4_4, TIME_SIG_3_4, TIME_SIG_6_8, TIME_SIG_2_4, TIME_SIG_2_2, TIME_SIG_OTHER  (6)
DENSITY_0 … DENSITY_7          (8)
PITCH_LOW_0 … PITCH_LOW_7      (8)
PITCH_HIGH_0 … PITCH_HIGH_7    (8)
VEL_MEAN_0 … VEL_MEAN_7        (8)
POLY_0 … POLY_3                (4)
CHORD_*                        (62: 12 roots × 5 qualities + OTHER + NONE)
─────────────────────────────────
Total: ~104 tokens
```

**Decoder vocabulary** (~468 tokens):
```
PAD, SOS, EOS                  (3)
BAR                            (1)
POSITION_0 … POSITION_15       (16)
ON_1 … ON_128                  (128)
OFF_1 … OFF_128                (128)
VELOCITY_1 … VELOCITY_32       (32)
─────────────────────────────────
Total: 308 tokens
```

TIME_SHIFT tokens from the old tokenizer are not included — REMI+ replaces absolute time
shifts with BAR + POSITION. The old tokenizer and vocabulary are kept in `tokenizing.py`
untouched for reference.

The encoder has its own `nn.Embedding(104, d_model)`. The decoder has its own
`nn.Embedding(308, d_model)` with weight-tied output projection as before.

---

### Inference

At inference you provide a list of bar descriptions — one per bar you want to generate.
Each description is the 7-token sequence: TIME_SIG, DENSITY, PITCH_LOW, PITCH_HIGH,
VEL_MEAN, POLY, CHORD. These come from a seed MIDI (extract features from an existing
piece) or are specified manually.

The generation loop:

```
generated = [<SOS>, <BAR>]
for each bar b in the target description list:
    encoder_states_b = encoder(desc_tokens[b])    # (7, d_model) — run once
    while True:
        next_token = decoder.sample(generated, encoder_states_b)
        if next_token == <BAR> or next_token == <EOS>:
            generated.append(next_token)
            break
        generated.append(next_token)

notes = reconstruct_notes_remi(generated)
```

The decoder generates one token at a time, using its KV-cache for the self-attention
(same as before). The encoder states for the current bar are fixed and passed to every
cross-attention call. When the decoder emits `<BAR>`, the loop encodes the next bar's
description and continues.

---

### Chord tokens

A chord is multiple notes played at the same time. In western music, chords are described
by two things:

- **Root** — the "home" note, one of the 12 pitch classes: C, C#, D, D#, E, F, F#, G, G#, A, A#, B
- **Quality** — the pattern of intervals above the root:
  - **Major** — bright. Root + 4 semitones + 7 semitones (e.g. C E G)
  - **Minor** — dark. Root + 3 semitones + 7 semitones (e.g. C Eb G)
  - **Dominant 7th** — tense, wants to resolve. Root + 4 + 7 + 10 semitones (e.g. C E G Bb)
  - **Diminished** — very tense. Root + 3 + 6 semitones (e.g. C Eb Gb)
  - **Augmented** — unstable, strange. Root + 4 + 8 semitones (e.g. C E G#)

To detect the chord in a bar: count how many times each of the 12 pitch classes appears
across all notes in the bar (ignoring octave). Score every (root, quality) pair by how
well the quality's interval pattern matches the histogram. Pick the best-scoring pair.
If even the best score is low — the notes don't fit any clean pattern — label it OTHER.
Empty bars are NONE.

Total chord tokens: 12 roots × 5 qualities + OTHER + NONE = **62**.

---

### New files and changes

| File | What changes |
|---|---|
| `data_management/expert_descriptions.py` | Add `detect_chord()` for chord detection per bar |
| `data_management/tokenizing_remi.py` | New file. REMI+ tokenizer. Produces `note_tokens`, `desc_tokens`, `bar_map` arrays. Two separate vocab dicts. |
| `data_management/adl_filter.py` | New file. Scan ADL corpus, write `adl_piano_files.txt`. |
| `model/models/DescriptionEncoder.py` | New file. 2-layer bidirectional transformer encoder. |
| `model/models/TransformerBlock.py` | Add cross-attention sub-layer (optional, off by default for backwards compat). |
| `model/models/GPTMidiV2.py` | New file. Decoder wrapping TransformerBlocks with cross-attention enabled. Handles bar-level encoder state routing. |
| `model/training/data.py` | Update dataset class to load and return `(note_tokens, desc_tokens, bar_map)` triples. |
| `model/training/training_loop.py` | Update forward pass to run encoder, gather per-position states, pass to decoder. |
| `model/inference/base_inference.py` | Bar-by-bar generation loop: encode description, decode notes, repeat. |
| `serve/schemas/generate.py` | Add `descriptions` field: list of per-bar feature dicts or seed MIDI path. |

---

### Implementation order

1. ✓ `remi_bar_exploration.ipynb` — bar parsing confirmed, `fit_boundaries()` updated to accept glob
2. ✓ `expert_descriptions.py` — bar feature extraction, boundary fitting complete
3. Add `detect_chord()` and chord survey to `expert_descriptions.py`
4. `tokenizing_remi.py` — REMI+ tokenizer producing the three-array output format
5. `adl_filter.py` — scan and filter ADL corpus
6. Tokenize ADL corpus → `data/adl_remi_notes.npy`, `data/adl_remi_desc.npy`, `data/adl_remi_barmap.npy`
7. `model/models/DescriptionEncoder.py`
8. `model/models/TransformerBlock.py` — add cross-attention sub-layer
9. `model/models/GPTMidiV2.py`
10. `model/training/data.py` — update dataset class
11. `model/training/training_loop.py` — update forward pass
12. Train Round 2
13. `model/inference/base_inference.py` + schema updates
