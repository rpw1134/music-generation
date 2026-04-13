import torch
from torch.nn import functional as F


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

    model.eval()
    with torch.no_grad():
        while tokens.shape[1] < max_length and tokens[0, -1] != 2:
            logits = model(tokens)           # (1, seq_len, vocab_size)
            next_logits = logits[0, -1, :]  # logits for last position only
            next_token = _sample_next_token(next_logits, temperature, top_k, top_p)
            tokens = torch.cat([tokens, next_token.view(1, 1)], dim=1)

    return tokens
