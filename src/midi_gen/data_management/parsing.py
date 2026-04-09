import pretty_midi
import numpy as np

def midi_to_notes(filepath = "data/maestro-v3.0.0/2018/MIDI-Unprocessed_Chamber2_MID--AUDIO_09_R3_2018_wav--1.midi"):
    midi_client = pretty_midi.PrettyMIDI(filepath)
    notes = []
    for instr in midi_client.instruments:
        for note in instr.notes:
            notes.append([note.start, note.end, note.pitch, note.velocity])
    return notes

def notes_to_vector(notes):
    # row is a note, cols are the data points
    return np.array(notes)

def file_path_to_vector(file_path):
    notes = midi_to_notes(file_path)
    return notes_to_vector(notes)

def save_vector_to_file(filepath, vector):
    np.save(filepath, vector)

# will break into NOTE_ON PITCH, NOTE OFF PITCH, TIME_DELAY, VELOCITY tokens
# TIME_DELAY is 10ms - 1s, 157 bins with: N bins from t_min to t_max, the bin edges are t_min * (t_max/t_min)^(i/N) for i in 0..N.
# VELOCITY is 32 values. Bins
# PITCH is 128 vals, same reason
# 157 + 32 + 128 + 128 + 3 (PAD, EOS, BOS) = 448

if __name__ == "__main__":
    v = file_path_to_vector("data/maestro-v3.0.0/2018/MIDI-Unprocessed_Chamber2_MID--AUDIO_09_R3_2018_wav--1.midi")
    save_vector_to_file("data/np/first.npy", v)