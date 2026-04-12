import torch
from torch.nn import functional as F


def _sample_next_token(logits: torch.Tensor, temperature: float, top_k: int = 0) -> torch.Tensor:
    """Sample a single token index from logits with temperature and optional top-k."""
    if temperature == 0.0:
        # greedy
        return logits.argmax(dim=-1)
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        logits[logits < values[-1]] = float('-inf')  # mask everything outside top-k
    probs = F.softmax(logits / temperature, dim=-1)
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


def create_sample_tokens(model, max_length: int, seed: torch.Tensor | None = None, temperature: float = 1.0, top_k: int = 0) -> torch.Tensor:
    """Autoregressively sample a token sequence from the model.

    Args:
        model:       trained GPTMidiV1, already on the correct device
        max_length:  maximum number of tokens to generate
        seed:        optional (1, seq_len) long tensor to condition on
        temperature: sampling temperature. 0.0 = greedy, higher = more random

    Returns:
        (1, seq_len) long tensor of token indices including the seed
    """
    device = next(model.parameters()).device
    tokens = _build_seed(seed, device)

    model.eval()
    with torch.no_grad():
        # while under max length and last token isn't <EOS>, keep generating
        while tokens.shape[1] < max_length and tokens[0, -1] != 2:
            logits = model(tokens)           # (1, seq_len, vocab_size)
            next_logits = logits[0, -1, :]  # logits for last position only
            next_token = _sample_next_token(next_logits, temperature, top_k)
            tokens = torch.cat([tokens, next_token.view(1, 1)], dim=1)
            print(tokens.shape)

    return tokens
