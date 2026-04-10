# CLIP and Cross-Attention

## What is CLIP?

CLIP (Contrastive Language-Image Pretraining, OpenAI 2021) is a model trained to align images
and text in a shared embedding space. It consists of two encoders — one for images (Vision
Transformer), one for text — trained jointly on hundreds of millions of (image, caption) pairs.
The objective is contrastive: pull matching image-text pairs close together in the embedding
space, push non-matching pairs apart.

The result is an image encoder that produces a dense vector (e.g., 512 or 768 floats) capturing
high-level semantic content — not pixel statistics, but *meaning*. A photo of a stormy ocean
and the text "turbulent, dark, dramatic" end up near each other in this space.

This makes CLIP embeddings ideal as conditioning signals: they encode *vibe*, not structure.

---

## Self-Attention vs. Cross-Attention

**Self-attention** lets each token in a sequence attend to every other token in *the same*
sequence. Queries, keys, and values all come from the same source. This is how the GPT decoder
builds context — each music token attends to all preceding music tokens.

**Cross-attention** splits the source: queries come from the target sequence (music tokens being
generated), but keys and values come from an *external* sequence (the image embedding). Each
music token asks: "given what I need to generate next, which parts of the image representation
are most relevant?"

```
Self-attention:    Q, K, V  ← music tokens
Cross-attention:   Q        ← music tokens
                   K, V     ← image embedding
```

The mechanics are identical — scaled dot-product attention — only the source of K and V differs.

---

## How the Image Becomes a Seed

Rather than a seed in the token sense, the image embedding acts as a **persistent conditioning
signal** throughout generation. At every decoder layer that has a cross-attention sublayer, the
music tokens attend to the image embedding. The image doesn't occupy positions in the token
sequence — it lives in a parallel context the decoder can query at each step.

Concretely, in the image-to-music model:
1. A frozen CLIP encoder converts the image to a vector (or a small set of patch vectors).
2. The decoder begins with just `<SOS>`.
3. At each layer, after self-attention over the music sequence, cross-attention pulls information
   from the image embedding to bias what token comes next.
4. The image influences every single generation step — it never "runs out" the way a token
   prompt would.

This is why cross-attention conditioning is strictly more powerful than simply prepending an
image token to the sequence: the signal is injected at every layer, not just consumed once at
the input.
