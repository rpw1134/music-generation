import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader

from midi_gen.data_management.tokenizing import create_vocabulary
from midi_gen.model.models.GPTMidiV1 import GPTMidiV1


def _train_epoch(model, loader, optimizer, lr_sched, vocab_size, grad_clip, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        # device
        x, y = x.to(device), y.to(device)

        # logits then flatten for CE loss
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        # backwards and clip
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        total_loss += loss.item()

        # steps
        optimizer.step()
        optimizer.zero_grad()
        lr_sched.step()

    return total_loss / len(loader)


def _validate(model, loader, vocab_size, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            # prediction
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

            # tracking
            total_loss += loss.item()
            total_correct += (logits.argmax(dim=-1) == y).sum().item()
            total_tokens += y.numel()

    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return avg_loss, accuracy, perplexity


def _save_checkpoint(model, optimizer, epoch, val_loss, path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
    }, path)


def load_checkpoint(model, optimizer, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["val_loss"]


def training_loop(model: nn.Module,
                  train_loader: DataLoader,
                  val_loader: DataLoader,
                  num_epochs=10,
                  lr=3e-4,
                  warmup_steps=200,
                  weight_decay=0.1,
                  b1=0.9,
                  b2=0.95,
                  grad_clip=1.0,
                  checkpoint_path="models/midiv1.pt"):

    # vocab size
    vocab, _ = create_vocabulary()
    vocab_size = len(vocab)

    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    # number of weight updates
    total_steps = len(train_loader) * num_epochs

    # transformers better with decoupled weight decay and LR
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(b1, b2))

    # transformers better with a warmup period followed by cosine annealing
    warmup_sched = optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
    cosine_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-5)

    # switches to cosine after warmup steps
    lr_sched = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps])

    # debugging purposes
    best_val_loss = float('inf')
    history = {"train_loss": [], "val_loss": [], "accuracy": [], "perplexity": []}

    for epoch in range(num_epochs):
        # TRAIN
        train_loss = _train_epoch(model, train_loader, optimizer, lr_sched, vocab_size, grad_clip, device)

        # EVAL
        val_loss, accuracy, perplexity = _validate(model, val_loader, vocab_size, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["accuracy"].append(accuracy)
        history["perplexity"].append(perplexity)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Accuracy: {accuracy:.4f} - Perplexity: {perplexity:.4f}")

        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
            print(f"  Saved best model (val_loss={val_loss:.4f})")

    return history
