import torch
from torch.nn import functional as F
from pathlib import Path


def _sample_next_token(logits: torch.Tensor, temperature: float, top_k: int = 0, top_p: float = 0.0) -> torch.Tensor:
    """Sample a single token index from logits with temperature, optional top-k, and optional top-p (nucleus)."""
    if temperature == 0.0:
        return logits.argmax(dim=-1)

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
    # prepend <SOS> if not already present
    if seed[0, 0] != 1:
        sos = torch.ones(1, 1, dtype=torch.long, device=device)
        seed = torch.cat([sos, seed], dim=1)
    return seed


def create_sample_tokens(model, max_length: int, seed: torch.Tensor | None = None, temperature: float = 1.0, top_k: int = 0, top_p: float = 0.0) -> torch.Tensor:
    """Autoregressively sample a token sequence from the model.

    Args:
        model:       trained GPTMidiV1, already on the correct device
        max_length:  maximum number of tokens to generate
        seed:        optional (1, seq_len) long tensor to condition on
        temperature: sampling temperature. 0.0 = greedy, higher = more random
        top_k:       if > 0, restrict sampling to the k most likely tokens
        top_p:       if > 0.0, nucleus sampling over cumulative probability mass

    Returns:
        (1, seq_len) long tensor of token indices including the seed
    """
    device = next(model.parameters()).device
    tokens = _build_seed(seed, device)

    # unwrap DataParallel if needed
    m = model.module if hasattr(model, 'module') else model
    num_layers = len(m.transformer_blocks)
    num_heads  = m.transformer_blocks[0].num_heads
    d_head     = m.transformer_blocks[0].d_head
    dtype      = next(m.parameters()).dtype

    # pre-allocate fixed-size KV buffers — no allocations during the decode loop
    kv_caches = [
        (
            torch.zeros(1, num_heads, max_length, d_head, dtype=dtype, device=device),
            torch.zeros(1, num_heads, max_length, d_head, dtype=dtype, device=device),
            0,  # filled length
        )
        for _ in range(num_layers)
    ]

    model.eval()
    with torch.no_grad():
        # prefill: write seed K/V into buffers, get logits for the last seed position
        logits, kv_caches = model(tokens, use_cache=True, kv_caches=kv_caches)
        next_token = _sample_next_token(logits[0, -1, :], temperature, top_k, top_p)
        tokens = torch.cat([tokens, next_token.view(1, 1)], dim=1)

        # decode: single token per step, in-place buffer writes, no new allocations
        while tokens.shape[1] < max_length and tokens[0, -1] != 2:
            logits, kv_caches = model(tokens[:, -1:], use_cache=True, kv_caches=kv_caches)
            next_token = _sample_next_token(logits[0, -1, :], temperature, top_k, top_p)
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
    seed: torch.Tensor | None = None,
) -> tuple[list[int], list[tuple], int]:
    """Generate a music sample from a pre-loaded model.

    Args:
        model:       GPTMidiV1, already on the correct device and in eval mode.
        midi_out:    path to write the generated MIDI file.
        wav_out:     path to write the rendered WAV file.
        max_length:  maximum number of tokens to generate.
        temperature: sampling temperature.
        top_k:       top-k cutoff (0 = disabled).
        top_p:       nucleus sampling threshold (0.0 = disabled).
        seed:        optional (1, seq_len) long tensor to condition on.

    Returns:
        token_indices:  list of raw token index integers.
        notes:          list of (start, end, pitch, velocity_bin) tuples.
        decode_errors:  count of malformed token sequences encountered.
    """
    from midi_gen.data_management.tokenizing import create_vocabulary, reconstruct_notes
    from midi_gen.data_management.midi_io import write_midi, midi_to_wav

    tokens = create_sample_tokens(
        model, max_length=max_length, seed=seed,
        temperature=temperature, top_k=top_k, top_p=top_p,
    )
    token_indices = tokens[0].tolist()

    _, inverse = create_vocabulary()
    token_strings = [inverse[t] for t in token_indices]
    notes, errors = reconstruct_notes(token_strings)

    write_midi(notes, midi_out)
    midi_to_wav(midi_out, wav_out)

    return token_indices, notes, len(errors)
