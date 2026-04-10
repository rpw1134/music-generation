import numpy as np
from midi_gen.data_management.parsing import file_path_to_vector

# will break into NOTE_ON PITCH, NOTE OFF PITCH, TIME_DELAY, VELOCITY tokens
# TIME_DELAY is 10ms - 1s, 157 bins with: N bins from t_min to t_max, the bin edges are t_min * (t_max/t_min)^(i/N) for i in 0..N.
# VELOCITY is 32 values. Bins
# PITCH is 128 vals, same reason
# 157 + 32 + 128 + 128 + 3 (PAD, EOS, BOS) = 448


TEST_MAX_SEQ_LENGTH = 1024
VOCAB = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2}

def create_vocabulary(bins=157, pitches=128, velocities=32):
    """Build token vocabulary and inverse lookup list.

    Returns:
        vocabulary: dict mapping token name to integer index, e.g. {"<TIME_SHIFT_0>": 3}
        inverse:    list mapping integer index to token name, e.g. inverse[3] == "TIME_SHIFT_0"

    Decoding a list of model output indices:
        tokens = [inverse[i] for i in model_output]

    Interpreting a token:
        type, value = token.split("_")[0], int(token.split("_")[-1])
        - TIME_SHIFT: pass value to get_time_shift_by_bin([value]) to get seconds
        - PITCH:      value is the MIDI pitch (1-indexed)
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
        curr+=1
    for i in range(pitches):
        vocabulary[f"<OFF_{i + 1}>"] = curr
        inverse.append(f"<OFF_{i + 1}>")
        curr += 1
    for i in range(velocities):
        vocabulary[f"<VELOCITY_{i+1}>"] = curr
        inverse.append(f"<VELOCITY_{i+1}>")
        curr+=1
    return vocabulary, inverse

def get_time_shift_bin(t, begin=0.01, end=1.0, bins=157):
    """Map a time shift value (seconds) to a list of log-scale bin indices.
    Values exceeding end are split into multiple tokens."""
    # recursively split if necessary
    if t > end:
        div = t/end
        wholes = int(div)
        frac = div - wholes
        li = [1] * wholes
        li.append(frac)
        return sum([get_time_shift_bin(ti) for ti in li], [])  # flatten

    t = max(begin, min(t, end))  # clamp
    log_ratio = np.log(end / begin) / bins
    bin_idx = round(np.log(t / begin) / log_ratio)
    return [min(bin_idx, bins - 1)]

def quantize_velocity(velocity, bins=32):
    """Map a MIDI velocity (0-127) to a bin index (1-indexed, matching VELOCITY tokens)."""
    return min(int(velocity * bins / 128) + 1, bins)

def dequantize_velocity(bin_idx, bins=32):
    """Map a velocity bin index (1-indexed) back to a MIDI velocity (0-127)."""
    return min(int((bin_idx - 1) * 128 / bins) + 4, 127)

def get_time_shift_by_bin(indices, begin=0.01, end=1.0, bins=157):
    """Inverse of get_time_shift_bin. Convert bin indices back to time shift values (seconds)."""
    # indices is a list of time shift indices to compute for
    return [begin * (end/begin)**(i/bins) for i in indices]

if __name__ == "__main__":
    pass