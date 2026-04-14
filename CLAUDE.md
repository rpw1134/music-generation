## Project
Small GPT-style transformer trained on MIDI sequences for music generation. The model treats
music as a sequence of discrete tokens (ON, OFF, TIME_SHIFT, VELOCITY) and learns
next-token prediction — identical to a language model.

Served via a FastAPI backend (not yet built) and simple web UI with audio playback and piano roll visualization (not yet built).

## Future Extension
Vision-to-music: condition the model on CLIP image embeddings via cross-attention to generate
music matching the "vibe" of an image or slideshow. Build the base music model first.

## Stack
- **Python 3.12**, managed with `uv`
- **PyTorch** — model and training
- **pretty_midi** — MIDI parsing and reconstruction
- **midi2audio** + FluidSynth (system) — MIDI → WAV
- **FastAPI** — serving (planned)

## Structure
```
src/midi_gen/
├── data_management/
│   ├── midi_io.py       # MIDI → note matrix (pretty_midi); notes → PrettyMIDI → write MIDI
│   ├── tokenizing.py    # vocab, tokenization, event encoding/decoding, reconstruct_notes
│   └── testing.py
├── model/
│   ├── models/
│   │   ├── GPTMidiV1.py      # GPT decoder: embedding → N×TransformerBlock → LayerNorm → out_proj
│   │   └── TransformerBlock.py  # pre-norm, RoPE, Flash Attention, FFN; KV-cache support
│   ├── training/
│   │   ├── training_loop.py  # training + validation loops, AdamW + warmup/cosine LR, checkpointing
│   │   ├── data.py           # PyTorch Dataset over tokenized windows
│   │   └── positional_encodings.py  # RoPE cos/sin table + apply_rope_transformations
│   └── inference/
│       ├── base_inference.py # create_sample_tokens: prefill + KV-cache decode loop
│       └── testing.py
├── exploration/
│   └── midi_test.py
└── main.py
```

## Vocabulary
448 tokens total:
- 3 special: `<PAD>=0`, `<SOS>=1`, `<EOS>=2`
- 157 `<TIME_SHIFT_i>` — log-scale bins, 0.01s–1.0s; values > 1s split into multiple tokens
- 128 `<ON_i>` — MIDI pitch (1-indexed)
- 128 `<OFF_i>` — MIDI pitch (1-indexed)
- 32 `<VELOCITY_i>` — 32 quantized bins (1-indexed)

## Model Architecture
- **GPTMidiV1**: 6-layer decoder-only transformer, `d_model=512`, 8 heads, `ff_dim_ratio=4`, `dropout=0.1`
- **Positional encoding**: RoPE (cos/sin table stored as registered buffer in each TransformerBlock)
- **Attention**: Flash Attention (`F.scaled_dot_product_attention`), causal mask during training
- **KV cache**: pre-allocated fixed-size buffers; prefill + single-token decode loop
- **Output**: weight-tied projection (`out_proj.weight = embedding.weight`), pre-norm

## Training
- AdamW (`lr=3e-4`, `weight_decay=0.1`, betas=(0.9, 0.95))
- Linear warmup (200 steps) → cosine annealing (`eta_min=1e-6`)
- Mixed precision fp16 (`torch.amp.GradScaler`)
- Multi-GPU via `DataParallel` when available
- `seq_length=2048`, `batch_size=32`
- Trained on Kaggle (GPU)

## Data
MAESTRO v3 — https://magenta.tensorflow.org/datasets/maestro
Uses GeneralUser-GS soundfont for MIDI → audio rendering.
Tokenized dataset saved as `.npy` arrays in `data/`.

## Planning Docs
`planning/` contains design notes for future/in-progress work:
- `maestro_model_iterating.md` — iteration log for MAESTRO experiments (retired dataset)
- `lakh_model_iterating.md` — **active iteration log** for Lakh Clean experiments (update after every arch/training/inference change)
- `clip_and_cross_attention.md` — vision-to-music conditioning
- `rope_and_yarn.md` — positional encoding extensions
- `inference_optimizations.md` — inference perf work
- `seeding.md` — seed/prompt strategies
- `diffusion.md` — diffusion-based generation notes
- `remote_training.md` — Kaggle/remote training setup

## Iteration Log
Whenever changes are made to the model architecture, training configuration, tokenization, or inference pipeline — document them in `planning/lakh_model_iterating.md`. Include what changed, what problem motivated the change, and what the outcome was (once known).