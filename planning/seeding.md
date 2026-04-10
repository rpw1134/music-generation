# Seeding and Initial Conditions

## What Is a Seed?

In autoregressive generation, the model produces one token at a time by sampling from
`P(token_t | token_0, token_1, ..., token_{t-1})`. It cannot generate without at least one
starting token — the seed is the initial context the model conditions on.

---

## Would `<SOS>` Always Yield the Same Output?

No — and this is by design. Generation involves **sampling** from the predicted probability
distribution at each step, not taking the argmax. Two runs from the same `<SOS>` will diverge
immediately after the first sampled token, because even a heavily peaked distribution still has
randomness. This randomness is controlled by **temperature**:

```
P(token_i) = softmax(logits_i / T)

T < 1.0  →  distribution sharpens, model becomes more deterministic, outputs more "expected"
T = 1.0  →  raw model distribution
T > 1.0  →  distribution flattens, more diversity, higher risk of incoherence
```

Fixing the random seed of the sampling process (e.g., `torch.manual_seed(42)`) would make
generation reproducible from the same input. Without fixing it, `<SOS>` is non-deterministic.

---

## Types of Seeds

**1. Minimal seed (`<SOS>` only)**
The model generates entirely from its learned prior. Output reflects the statistical average
of the training data — competent but generic. At very low temperature, outputs may converge
toward common patterns in the training set.

**2. Musical prompt (continuation)**
Provide `<SOS>` + a sequence of real music tokens as context. The model continues the piece.
The longer and more specific the prompt, the more constrained the output. Useful for
style-matching or completing a fragment.

**3. Conditioning vector (future: image)**
The seed token sequence is still just `<SOS>`, but the image embedding provided via
cross-attention shifts the entire conditional distribution. Every token is sampled from
`P(token_t | token_0..t-1, image_embedding)` rather than the unconditional prior. The image
doesn't consume context window positions — it biases generation from outside the sequence.

---

## Stochasticity vs. Determinism

| Approach | Deterministic? | Notes |
|---|---|---|
| Argmax (greedy) | Yes | Often produces repetitive, degenerate output |
| Fixed seed + sampling | Yes (given same input) | Reproducible, still diverse across seeds |
| Top-k sampling | No | Restricts sampling to k most likely tokens |
| Top-p (nucleus) sampling | No | Restricts to smallest set covering p% of mass |
| Temperature only | No | Scales full distribution |

In practice, top-p or top-k sampling with a moderate temperature (0.8–1.0) produces the best
balance of coherence and variety. Pure argmax decoding tends to loop.
