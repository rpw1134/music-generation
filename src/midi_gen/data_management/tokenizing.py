import numpy as np
from midi_gen.data_management.parsing import file_path_to_vector

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
    inverse = ["PAD", "SOS", "EOS"]
    curr = 3
    for i in range(bins):
        vocabulary[f"<TIME_SHIFT_{i}>"] = curr
        inverse.append(f"TIME_SHIFT_{i}")
        curr += 1
    for i in range(pitches):
        vocabulary[f"<PITCH_{i+1}>"] = curr
        inverse.append(f"PITCH_{i+1}")
        curr+=1
    for i in range(velocities):
        vocabulary[f"<VELOCITY_{i+1}>"] = curr
        inverse.append(f"VELOCITY_{i+1}")
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

def get_time_shift_by_bin(indices, begin=0.01, end=1.0, bins=157):
    """Inverse of get_time_shift_bin. Convert bin indices back to time shift values (seconds)."""
    # indices is a list of time shift indices to compute for
    return [begin * (end/begin)**(i/bins) for i in indices]

# tester function
def tokenize_sample(file_path: str) -> np.ndarray:
    vocab, _ = create_vocabulary(bins=157, pitches=128, velocities=32)
    # vector of notes Nx4
    vec = file_path_to_vector(file_path)
    # time in seconds since 0, type, velocity (pitch for on offs), pitch (associated pitch for velocities)
    notes = []
    for note in vec:
        start, end, pitch, velocity = note
        pitch = int(pitch)
        velocity = quantize_velocity(int(velocity))
        notes.append((start, "velocity", velocity, pitch))
        notes.append((start, "on", pitch, pitch))
        notes.append((end, "off", pitch, pitch))
    notes.sort(key=lambda x: (x[0], x[3]))
    curr_time = 0
    tokens = [vocab["<SOS>"]]
    for note in notes:
        time, event, velocity, pitch = note
        if time-curr_time > 0.005:
            time_shifts = get_time_shift_bin(time-curr_time)
            for time_shift in time_shifts:
                tokens.append(vocab[f"<TIME_SHIFT_{time_shift}>"])
            curr_time = time
        if event == "velocity":
            tokens.append(vocab[f"<VELOCITY_{int(velocity)}>"])
        elif event == "on":
            tokens.append(vocab[f"<PITCH_{pitch}>"])
        elif event == "off":
            tokens.append(vocab[f"<PITCH_{pitch}>"])
    tokens.append(vocab["<EOS>"])
    print(tokens)
    return np.array(tokens)





if __name__ == "__main__":
    tokenize_sample("data/maestro-v3.0.0/2018/MIDI-Unprocessed_Chamber2_MID--AUDIO_09_R3_2018_wav--1.midi")