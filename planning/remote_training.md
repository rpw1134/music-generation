# Remote Training and Local Inference

## The Basic Workflow

Training and inference are fully decoupled. PyTorch models are saved as weight files that can
be loaded on any machine, regardless of where training happened.

**On the remote (after training):**
```
torch.save(model.state_dict(), "checkpoint.pt")
```

**On your Mac (inference):**
```python
model = MusicTransformer(config)               # same architecture definition
state = torch.load("checkpoint.pt", map_location="cpu")
model.load_state_dict(state)
model.to("mps")                                # Apple Silicon GPU
model.eval()
```

That's it. The weight file is portable — it has no knowledge of where it was created.

### Save More Than Just Weights

Colab sessions die unexpectedly. Save full checkpoints periodically, not just at the end:

```python
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": loss,
    "config": config,          # save your ModelConfig too
}, f"checkpoint_epoch_{epoch}.pt")
```

This lets you resume training from any checkpoint rather than starting over. When using Colab,
mount Google Drive and write checkpoints there — they persist after the session ends.

---

## Apple Silicon for Inference

Yes, Apple Silicon is more than sufficient for inference on a model this size. The key
properties that make it well-suited:

- **Unified memory**: CPU and GPU share the same physical RAM pool. A 16GB M-series chip has
  16GB available to the GPU — no VRAM bottleneck. Most discrete laptop GPUs have 4-8GB VRAM.
- **MPS backend**: PyTorch supports Metal Performance Shaders (`device = "mps"`) for
  GPU-accelerated inference on M1/M2/M3/M4.
- **No gradient computation**: Inference uses roughly 1/3 the memory of training (no gradients,
  no optimizer states). A model that barely fits in training VRAM runs easily at inference.

For a 100M parameter model in float32: ~400MB. In float16 (standard for inference): ~200MB.
An M2 MacBook Air with 8GB handles this without breaking a sweat.

---

## What Size Should the Model Be?

100M parameters is a reasonable upper bound for this task, not a minimum. Consider:

- GPT-2 Small (117M params) generates coherent English text over a vocabulary of 50,000 tokens.
- Your vocabulary is 448 tokens. Music has more structure and less vocabulary diversity than
  language. A smaller model can capture it.
- The MAESTRO dataset (~200 files for your subset) is small relative to typical LLM training
  corpora. A very large model will overfit.

**Recommended progression:**

| Phase | Params | Purpose |
|---|---|---|
| Fast iteration | 5–15M | Verify training loop, loss curve, generation quality |
| Real training | 30–60M | Good quality output, trains in hours on a T4 |
| Scaled up | 80–120M | Diminishing returns given dataset size |

Start small. A 10M model that generates something musical after 2 hours of training gives you
faster feedback than a 100M model that takes a day and may overfit anyway.

---

## Cheap Training Services

### Free

**Google Colab (free tier)**
- GPU: NVIDIA T4 (16GB VRAM)
- Limits: ~12h sessions, GPU not always available, disconnects if idle
- Critical: mount Google Drive to persist checkpoints across sessions
- Best for: getting started, short experiments

**Kaggle Kernels**
- GPU: T4 or P100 (16GB VRAM)
- Limits: 30 GPU hours/week, but no idle disconnection
- No session management headaches — more reliable than free Colab
- Best for: longer runs without babysitting

### Paid (Pay-as-you-go)

**Google Colab Pro / Pro+**
- Pro (~$10/mo): better GPU access (A100 sometimes), longer sessions
- Pro+  (~$50/mo): background execution, more compute units
- Best for: if you're already using Colab and want fewer interruptions

**RunPod / Vast.ai**
- Community marketplace: rent idle GPUs from individuals
- Price: $0.10–$0.50/hr for an RTX 3090 or A100
- Vast.ai is cheaper but less reliable; RunPod is slightly more polished
- Best for: longer training runs where you want to leave it overnight

**Lambda Labs**
- $0.50–$1.50/hr for A100/H100
- More reliable than community marketplaces, SSH access, persistent storage
- Best for: if you want a proper cloud environment without AWS complexity

### Practical Recommendation

For this project: **start on Kaggle** (free, reliable, 30h/week). When you have a training loop
that works and you need longer runs, move to **RunPod** for overnight jobs at ~$0.20–0.40/hr.
Avoid Colab free tier for anything over 30 minutes — the session instability wastes time.

---

## Colab-Specific Tips

```python
# Mount Drive at the top of your notebook
from google.colab import drive
drive.mount("/content/drive")
CHECKPOINT_DIR = "/content/drive/MyDrive/music-gen/checkpoints"

# Install your package
!pip install -e /content/music-gen --quiet

# Save checkpoint to Drive (survives session death)
torch.save(checkpoint, f"{CHECKPOINT_DIR}/epoch_{epoch}.pt")
```

Clone your repo from GitHub at the start of each session. Keep weights on Drive, code on GitHub.
Never rely on the Colab instance's local disk persisting — it won't.
