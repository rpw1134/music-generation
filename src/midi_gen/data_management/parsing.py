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

if __name__ == "__main__":
    v = file_path_to_vector("data/maestro-v3.0.0/2018/MIDI-Unprocessed_Chamber2_MID--AUDIO_09_R3_2018_wav--1.midi")
    save_vector_to_file("data/np/first.npy", v)