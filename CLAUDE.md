## Project
Small GPT-style transformer trained on MIDI sequences for music generation. The model treats
music as a sequence of discrete tokens (NOTE_ON, NOTE_OFF, TIME_SHIFT, VELOCITY) and learns
next-token prediction — identical to a language model.

Served via a FastAPI backend and simple web UI with audio playback and piano roll visualization.

## Future Extension
Vision-to-music: condition the model on CLIP image embeddings via cross-attention to generate
music matching the "vibe" of an image or slideshow. Build the base music model first.

## Stack
- **Python 3.11+**, managed with `uv`
- **PyTorch** — model and training
- **pretty_midi** — MIDI parsing
- **midi2audio** + FluidSynth (system) — MIDI → WAV
- **FastAPI** — serving

## Structure
```
src/midi_gen/
├── data/
│   ├── parse.py        # MIDI → event list
│   ├── tokenize.py     # events → token sequences + vocab
│   └── dataset.py      # PyTorch Dataset
├── model/
│   ├── transformer.py  # GPT decoder, built from scratch
│   └── config.py       # ModelConfig dataclass
├── train.py
├── generate.py         # autoregressive sampling
├── midi_utils.py       # token sequence → MIDI → WAV
└── serve/
    └── api.py
```

## Data
MAESTRO v3 — https://magenta.tensorflow.org/datasets/maestro
Start with a small subset (100–200 files) for fast iteration.<br>
Uses GeneralUser-GS soundfont