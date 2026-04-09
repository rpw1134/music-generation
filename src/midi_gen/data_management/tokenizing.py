import numpy as np
from midi_gen.data_management.parsing import file_path_to_vector

TEST_MAX_SEQ_LENGTH = 1024
VOCAB = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2}

def get_time_shift_bin(t, begin=0.01, end=1, bins=100):
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

def get_time_shift_by_bin(indices, begin=0.01, end=1, bins=100):
    # indices is a list of time shift indices to compute for
    return [begin * (end/begin)**(i/bins) for i in indices]

# tester function
def tokenize_sample(file_path: str) -> np.ndarray:
    # vector of notes Nx4
    vec = file_path_to_vector(file_path)
    # time in seconds since 0, type, velocity (pitch for on offs), pitch (associated pitch for velocities)
    notes = []
    for note in vec:
        start, end, pitch, velocity = note
        notes.append((start, "velocity", velocity, pitch))
        notes.append((start, "on", pitch, pitch))
        notes.append((end, "off", pitch, pitch))
    notes.sort(key=lambda x: (x[0], x[3]))
    print(notes)



if __name__ == "__main__":
    tokenize_sample("data/maestro-v3.0.0/2018/MIDI-Unprocessed_Chamber2_MID--AUDIO_09_R3_2018_wav--1.midi")