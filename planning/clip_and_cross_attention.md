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

### You Don't Train CLIP

OpenAI released CLIP weights publicly. You load a frozen pretrained encoder and use it as a
feature extractor — you never retrain it. The only parameters you train are the parts of your
decoder that learn to interpret CLIP's output (cross-attention projection weights, and optionally
a small connector MLP).

A typical connector is just a single linear layer mapping CLIP's output dim (e.g., 512) to your
model's hidden dim. This is all you need to start. Training this alongside cross-attention while
CLIP stays frozen is low-cost even on a laptop GPU.

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

### They Stack — Self-Attention Is Not Replaced

Cross-attention does not replace self-attention. A decoder block contains both, applied in order:

```
x = x + self_attention(x)           # attend to previous music tokens (causal mask applied)
x = x + cross_attention(x, image)   # attend to image embedding
x = x + feed_forward(x)
```

Sequential context — everything that came before in the music sequence — flows through
self-attention exactly as in a standard GPT decoder. Cross-attention is an additional sublayer
that runs after, injecting the image signal into an already-context-aware representation.

At each generation step the current token has both:
- Full causal context from all prior music tokens (self-attention)
- Image conditioning injected at every layer (cross-attention)

Self-attention answers "what has happened so far in the music?" Cross-attention answers "given
the image, what should happen next?" Both signals combine before the final token prediction.

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
