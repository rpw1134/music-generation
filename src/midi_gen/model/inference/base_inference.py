import torch
from torch.nn import functional as F
from pathlib import Path

# ---------------------------------------------------------------------------
# Vocabulary index ranges — must match create_vocabulary() in tokenizing.py
# Layout: PAD SOS EOS | TIME_SHIFT×157 | ON×128 | OFF×128 | VELOCITY×32
# ---------------------------------------------------------------------------
_BINS      = 157
_PITCHES   = 128
_ON_START  = 3 + _BINS                   # 160 — first ON token (pitch 1)
_ON_END    = _ON_START + _PITCHES - 1    # 287 — last  ON token (pitch 128)
_OFF_START = _ON_END + 1                 # 288 — first OFF token
_OFF_END   = _OFF_START + _PITCHES - 1  # 415 — last  OFF token


def _apply_pitch_penalty(
    logits: torch.Tensor,
    recent_tokens: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """Penalise ON pitch tokens that appear in recent_tokens.

    Only ON tokens are penalised — TIME_SHIFT, VELOCITY, and OFF tokens are
    left untouched. This avoids the failure mode of blanket repetition penalty,
    which previously broke token grammar by suppressing structural tokens that
    legitimately repeat (e.g. TIME_SHIFT, VELOCITY).

    Standard penalty formula (Keskar et al.):
        logit = logit / penalty   if logit > 0
        logit = logit * penalty   if logit < 0

    A penalty of 1.0 is a no-op. Values in the range 1.1–1.5 are reasonable
    starting points; above ~2.0 the effect becomes very aggressive.

    Args:
        logits:        1-D logit tensor over the full vocabulary (448,).
        recent_tokens: 1-D tensor of the last N generated token indices.
        penalty:       multiplicative penalty factor (>= 1.0).

    Returns:
        Modified logits tensor (same shape, in-place modification).
    """
    if penalty == 1.0:
        return logits

    # Collect unique ON pitches that have appeared in the recent window
    on_mask = (recent_tokens >= _ON_START) & (recent_tokens <= _ON_END)
    seen_on = recent_tokens[on_mask].unique()

    if seen_on.numel() == 0:
        return logits

    # Apply penalty in-place
    logits[seen_on] = torch.where(
        logits[seen_on] > 0,
        logits[seen_on] / penalty,
        logits[seen_on] * penalty,
    )
    return logits


def _sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int = 0,
    top_p: float = 0.0,
    pitch_penalty: float = 1.0,
    recent_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample a single token index from logits.

    Args:
        logits:         Raw logits over the full vocabulary (448,).
        temperature:    Sampling temperature. 0.0 = greedy.
        top_k:          If > 0, restrict to the k highest-probability tokens.
        top_p:          If > 0.0, nucleus sampling over cumulative probability.
        pitch_penalty:  Repetition penalty applied only to ON pitch tokens.
                        1.0 = disabled. Try 1.1–1.5 to reduce note looping.
        recent_tokens:  Token history to check for repeats. Required when
                        pitch_penalty > 1.0; ignored otherwise.
    """
    if temperature == 0.0:
        return logits.argmax(dim=-1)

    # Pitch-only repetition penalty — applied before temperature scaling so
    # the effect is consistent regardless of temperature
    if pitch_penalty > 1.0 and recent_tokens is not None:
        logits = _apply_pitch_penalty(logits, recent_tokens, pitch_penalty)

    logits = logits / temperature

    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        logits[logits < values[-1]] = float('-inf')

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        sorted_indices_to_remove = (cumulative_probs - F.softmax(sorted_logits, dim=-1)) > top_p
        sorted_logits[sorted_indices_to_remove] = float('-inf')
        logits = torch.zeros_like(logits).scatter_(0, sorted_indices, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _build_seed(seed: torch.Tensor | None, device: torch.device) -> torch.Tensor:
    """Return a (1, seq_len) long tensor, prepending <SOS> if needed."""
    if seed is None:
        return torch.tensor([[1]], dtype=torch.long, device=device)
    if seed[0, 0] != 1:
        sos = torch.ones(1, 1, dtype=torch.long, device=device)
        seed = torch.cat([sos, seed], dim=1)
    return seed


def create_sample_tokens(
    model,
    max_length: int,
    seed: torch.Tensor | None = None,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    pitch_penalty: float = 1.0,
    pitch_penalty_window: int = 64,
) -> torch.Tensor:
    """Autoregressively sample a token sequence from the model.

    Args:
        model:                Trained GPTMidiV1, already on the correct device.
        max_length:           Maximum number of tokens to generate.
        seed:                 Optional (1, seq_len) long tensor to condition on.
        temperature:          Sampling temperature. 0.0 = greedy.
        top_k:                If > 0, restrict to the top-k tokens.
        top_p:                If > 0.0, nucleus sampling threshold.
        pitch_penalty:        Repetition penalty for ON pitch tokens only.
                              1.0 = disabled (default). Try 1.1–1.5 to reduce
                              note looping without breaking token grammar.
        pitch_penalty_window: Number of most-recently generated tokens to scan
                              for repeated pitches. Smaller = more local penalty
                              (only recent notes); larger = longer memory.
                              Default: 64 (~a few bars of typical density).

    Returns:
        (1, seq_len) long tensor of token indices including the seed.
    """
    device = next(model.parameters()).device
    tokens = _build_seed(seed, device)

    m = model.module if hasattr(model, 'module') else model
    num_layers = len(m.transformer_blocks)
    num_heads  = m.transformer_blocks[0].num_heads
    d_head     = m.transformer_blocks[0].d_head
    dtype      = next(m.parameters()).dtype

    kv_caches = [
        (
            torch.zeros(1, num_heads, max_length, d_head, dtype=dtype, device=device),
            torch.zeros(1, num_heads, max_length, d_head, dtype=dtype, device=device),
            0,
        )
        for _ in range(num_layers)
    ]

    model.eval()
    with torch.no_grad():
        logits, kv_caches = model(tokens, use_cache=True, kv_caches=kv_caches)
        recent = tokens[0, -pitch_penalty_window:] if pitch_penalty > 1.0 else None
        next_token = _sample_next_token(logits[0, -1, :], temperature, top_k, top_p, pitch_penalty, recent)
        tokens = torch.cat([tokens, next_token.view(1, 1)], dim=1)

        while tokens.shape[1] < max_length and tokens[0, -1] != 2:
            logits, kv_caches = model(tokens[:, -1:], use_cache=True, kv_caches=kv_caches)
            recent = tokens[0, -pitch_penalty_window:] if pitch_penalty > 1.0 else None
            next_token = _sample_next_token(logits[0, -1, :], temperature, top_k, top_p, pitch_penalty, recent)
            tokens = torch.cat([tokens, next_token.view(1, 1)], dim=1)

    return tokens


def generate_sample(
    model,
    midi_out: str,
    wav_out: str,
    max_length: int = 1024,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    pitch_penalty: float = 1.0,
    pitch_penalty_window: int = 64,
    seed: torch.Tensor | None = None,
) -> tuple[list[int], list[tuple], int]:
    """Generate a music sample from a pre-loaded model.

    Args:
        model:                GPTMidiV1, already on the correct device and in eval mode.
        midi_out:             Path to write the generated MIDI file.
        wav_out:              Path to write the rendered WAV file.
        max_length:           Maximum number of tokens to generate.
        temperature:          Sampling temperature.
        top_k:                Top-k cutoff (0 = disabled).
        top_p:                Nucleus sampling threshold (0.0 = disabled).
        pitch_penalty:        Repetition penalty for ON pitch tokens only.
                              1.0 = disabled. Try 1.1–1.5 first.
        pitch_penalty_window: How many recent tokens to check for repeats.
        seed:                 Optional (1, seq_len) long tensor to condition on.

    Returns:
        token_indices:  List of raw token index integers.
        notes:          List of (start, end, pitch, velocity_bin) tuples.
        decode_errors:  Count of malformed token sequences encountered.
    """
    from midi_gen.data_management.tokenizing import create_vocabulary, reconstruct_notes
    from midi_gen.data_management.midi_io import write_midi, midi_to_wav

    tokens = create_sample_tokens(
        model, max_length=max_length, seed=seed,
        temperature=temperature, top_k=top_k, top_p=top_p,
        pitch_penalty=pitch_penalty, pitch_penalty_window=pitch_penalty_window,
    )
    token_indices = tokens[0].tolist()

    _, inverse = create_vocabulary()
    token_strings = [inverse[t] for t in token_indices]
    notes, errors = reconstruct_notes(token_strings)

    write_midi(notes, midi_out)
    midi_to_wav(midi_out, wav_out)

    return token_indices, notes, len(errors)
