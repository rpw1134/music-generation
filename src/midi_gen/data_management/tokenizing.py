import numpy as np
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # music-gen/
DATA_DIR = PROJECT_ROOT / "data"

# Vocabulary layout: 157 TIME_SHIFT + 128 ON + 128 OFF + 32 VELOCITY + 3 special = 448 tokens
TEST_MAX_SEQ_LENGTH = 1024


def create_vocabulary(bins=157, pitches=128, velocities=32):
    """Build token vocabulary and inverse lookup list.

    Returns:
        vocabulary: dict mapping token name to integer index, e.g. {"<TIME_SHIFT_0>": 3}
        inverse:    list mapping integer index to token name, e.g. inverse[3] == "<TIME_SHIFT_0>"

    Decoding a list of model output indices:
        tokens = [inverse[i] for i in model_output]

    Interpreting a token:
        type, value = token.split("_")[0], int(token.split("_")[-1])
        - TIME_SHIFT: pass value to get_time_shift_by_bin([value]) to get seconds
        - ON / OFF:   value is the MIDI pitch (1-indexed)
        - VELOCITY:   value is the velocity bin (1-indexed)
    """
    vocabulary = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2}
    inverse = ["<PAD>", "<SOS>", "<EOS>"]
    curr = 3
    for i in range(bins):
        vocabulary[f"<TIME_SHIFT_{i}>"] = curr
        inverse.append(f"<TIME_SHIFT_{i}>")
        curr += 1
    for i in range(pitches):
        vocabulary[f"<ON_{i+1}>"] = curr
        inverse.append(f"<ON_{i+1}>")
        curr += 1
    for i in range(pitches):
        vocabulary[f"<OFF_{i+1}>"] = curr
        inverse.append(f"<OFF_{i+1}>")
        curr += 1
    for i in range(velocities):
        vocabulary[f"<VELOCITY_{i+1}>"] = curr
        inverse.append(f"<VELOCITY_{i+1}>")
        curr += 1
    return vocabulary, inverse


def get_time_shift_bin(t, begin=0.01, end=1.0, bins=157):
    """Map a time shift value (seconds) to a list of log-scale bin indices.
    Values exceeding end are split into multiple tokens."""
    if t > end:
        div = t / end
        wholes = int(div)
        frac = div - wholes
        li = [1] * wholes
        li.append(frac)
        return sum([get_time_shift_bin(ti) for ti in li], [])  # flatten

    t = max(begin, min(t, end))  # clamp
    log_ratio = np.log(end / begin) / bins
    bin_idx = round(np.log(t / begin) / log_ratio)
    return [min(bin_idx, bins - 1)]


def get_time_shift_by_bin(indices, begin=0.01, end=1.0, bins=157):
    """Inverse of get_time_shift_bin. Convert bin indices back to time shift values (seconds)."""
    return [begin * (end / begin) ** (i / bins) for i in indices]


def quantize_velocity(velocity, bins=32):
    """Map a MIDI velocity (0-127) to a bin index (1-indexed, matching VELOCITY tokens)."""
    return min(int(velocity * bins / 128) + 1, bins)


def dequantize_velocity(bin_idx, bins=32):
    """Map a velocity bin index (1-indexed) back to a MIDI velocity (0-127)."""
    return min(int((bin_idx - 1) * 128 / bins) + 4, 127)


def decode_token(token: str) -> tuple[str, int]:
    """Parse a token string into (kind, value).

    Examples:
        "<TIME_SHIFT_5>"  -> ("TIME_SHIFT", 5)
        "<ON_64>"         -> ("ON", 64)
        "<OFF_64>"        -> ("OFF", 64)
        "<VELOCITY_12>"   -> ("VELOCITY", 12)
        "<SOS>"           -> ("SOS", -1)
    """
    inner = token.strip("<>")
    if "_" not in inner:
        return inner, -1
    *parts, val = inner.split("_")
    return "_".join(parts), int(val)


def reconstruct_notes(token_strings: list[str]) -> tuple[list[tuple], list[dict]]:
    """Convert a decoded token sequence into a list of notes and a list of errors.

    Each note is a tuple of (start, end, pitch, velocity_bin).
    Errors are collected for malformed sequences (e.g. OFF before ON, ON with no
    preceding VELOCITY) rather than raising, so the caller can inspect bad tokens.
    """
    notes = []
    errors = []
    active = defaultdict(dict)  # pitch -> {velocity, on}
    current_velocity = None
    time = 0.0

    for i, token in enumerate(token_strings):
        kind, val = decode_token(token)

        if kind in ("SOS", "EOS", "PAD"):
            continue
        elif kind == "TIME_SHIFT":
            time += get_time_shift_by_bin([val])[0]
        elif kind == "VELOCITY":
            current_velocity = val
        elif kind == "ON":
            if current_velocity is None:
                errors.append({"error": "ON token with no preceding VELOCITY", "index": i, "token": token})
                continue
            active[val] = {"pitch": val, "velocity": current_velocity, "on": time}
            current_velocity = None
        elif kind == "OFF":
            if val not in active:
                errors.append({"error": "OFF token with no matching ON", "index": i, "token": token})
                continue
            entry = active.pop(val)
            notes.append((entry["on"], time, entry["pitch"], entry["velocity"]))

    return notes, errors


def notes_to_events(vec) -> list[tuple]:
    """Expand a raw note matrix (Nx4) into a sorted list of (time, event, velocity, pitch) tuples.

    Each note produces three events: a velocity event and a note-on at start time,
    and a note-off at end time. Events are sorted by time, then pitch.
    """
    events = []
    for note in vec:
        start, end, pitch, velocity = note
        pitch = int(pitch)
        velocity = quantize_velocity(int(velocity))
        events.append((start, "velocity", velocity, pitch))
        events.append((start, "on", pitch, pitch))
        events.append((end, "off", pitch, pitch))
    events.sort(key=lambda x: (x[0], x[3]))
    return events


def events_to_token_array(events: list[tuple], bins=157, pitches=128, velocities=32) -> np.ndarray:
    """Convert a sorted event list into a 1D array of integer token indices."""
    vocab, _ = create_vocabulary(bins=bins, pitches=pitches, velocities=velocities)
    curr_time = 0
    tokens = [vocab["<SOS>"]]
    for time, event, velocity, pitch in events:
        if time - curr_time > 0.005:
            for bin_idx in get_time_shift_bin(time - curr_time, bins=bins):
                tokens.append(vocab[f"<TIME_SHIFT_{bin_idx}>"])
            curr_time = time
        if event == "velocity":
            tokens.append(vocab[f"<VELOCITY_{velocity}>"])
        elif event == "on":
            tokens.append(vocab[f"<ON_{pitch}>"])
        elif event == "off":
            tokens.append(vocab[f"<OFF_{pitch}>"])
    tokens.append(vocab["<EOS>"])
    return np.array(tokens, dtype=np.int32)


def tokenize_file(file_path: str, bins=157, pitches=128, velocities=32) -> np.ndarray:
    """Tokenize a single MIDI file into a 1D array of integer token indices.

    Intended to be called on every file in the dataset. The resulting arrays
    can be concatenated and split into fixed-length windows for training.
    """
    from midi_gen.data_management.midi_io import file_path_to_vector
    vec = file_path_to_vector(file_path)
    events = notes_to_events(vec)
    return events_to_token_array(events, bins=bins, pitches=pitches, velocities=velocities)

def tokenize_dataset(glob_pattern: str, bins=157, pitches=128, velocities=32, seq_length=1024) -> np.ndarray:
    files = sorted(Path(DATA_DIR).glob(glob_pattern))
    arrays = np.concatenate([tokenize_file(str(f), bins, pitches, velocities) for f in files], axis=0)
    remainder = len(arrays) % seq_length
    if remainder != 0:
        arrays = np.concatenate([arrays, np.zeros(seq_length - remainder, dtype=np.int32)])
    return arrays.reshape(-1, seq_length)


if __name__ == "__main__":
    from .midi_io import save_vector_to_file
    arr = tokenize_dataset("maestro-v3.0.0/**/*.midi", seq_length=2048)
    print(arr.shape)
    save_vector_to_file(DATA_DIR / "tokenized_dataset_long.npy", arr)
