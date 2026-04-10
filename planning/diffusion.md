# Diffusion Models

## Core Idea

Diffusion models learn to generate data by learning the reverse of a noise-adding process.
The training procedure has two phases:

**Forward process (fixed, not learned)**: Gradually corrupt a real data sample by adding
Gaussian noise over T steps until the sample is indistinguishable from pure noise.

**Reverse process (learned)**: Train a neural network to predict, at each step, how much
noise was added — effectively learning to "denoise" a slightly noisy sample. At inference,
start from pure noise and apply the learned denoiser T times to recover a coherent sample.

The network never sees the final data directly. It learns the *score* — the gradient of the
data distribution with respect to the noisy input — which is sufficient to navigate from
noise back toward real data.

---

## Why It Produces High-Quality Output

Unlike autoregressive models (one token at a time) or GANs (single-pass generator), diffusion
models refine the entire output over many steps. Each denoising step can correct global
structure (step 1: rough shape) and then local detail (step T: fine texture). This iterative
refinement is why diffusion models produce photorealistic images and high-fidelity audio.

---

## Diffusion for Audio: AudioLDM / MusicLDM

Raw audio waveforms are extremely high-dimensional (44100 floats per second). Diffusion
directly in waveform space is computationally prohibitive. The practical solution:

1. Encode audio into a compact **latent representation** using a pretrained VAE or codec.
2. Run diffusion in **latent space** (much smaller).
3. Decode the denoised latent back to audio.

This is Latent Diffusion (LDM). Conditioning on text or images is done by injecting an
embedding (e.g., CLIP or a text encoder) into the denoising network via cross-attention at
each diffusion step — exactly the same cross-attention mechanism described in
`clip_and_cross_attention.md`.

---

## Why It Doesn't Fit This Project

This project uses **discrete tokens** (NOTE_ON, NOTE_OFF, etc.) rather than continuous audio.
Diffusion is fundamentally a continuous process — it requires adding and removing real-valued
Gaussian noise. Discrete diffusion exists but is immature and considerably more complex.

More importantly, the autoregressive token approach has significant advantages for this use case:
- Directly interpretable and editable output (MIDI events)
- Natural fit for sequence continuation and prompting
- Simpler architecture with a clearer path to cross-attention conditioning
- Much faster iteration at small scale

Diffusion would become relevant if the goal shifted to generating raw audio or mel spectrograms
rather than symbolic MIDI — and even then, it would sit *on top of* a different representation,
not replace the transformer architecture.
