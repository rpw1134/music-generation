import torch
from torch import nn
from torch.nn import functional as F
def create_sample_tokens(seed, model, max_length):
    if not seed:
        seed = torch.ones(1)

    while len(seed) < max_length and seed[-1] != 2:
        next_token_logits = model(seed)
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.argmax(probs, dim=-1)
        seed.append(next_token)

