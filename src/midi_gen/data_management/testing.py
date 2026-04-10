from midi_gen.data_management.midi_io import write_midi
from midi_gen.data_management.tokenizing import create_vocabulary, tokenize_file, reconstruct_notes
import numpy as np
from midi_gen.exploration.midi_test import create_and_play_audio

def tokenize_sample(file_path: str) -> np.ndarray:
    return tokenize_file(file_path)

def parse_tokens_to_midi(tokens, output_path: str):
    """Decode a token index sequence back to a MIDI file.

    Converts indices to token strings via the inverse vocabulary, reconstructs
    notes using reconstruct_notes from tokenizing.py, and writes the result.
    Any malformed token sequences (e.g. OFF before ON) are collected as errors
    and printed rather than raising.
    """
    _, inverse = create_vocabulary(bins=157, pitches=128, velocities=32)
    token_strings = [inverse[t] for t in tokens]
    notes, errors = reconstruct_notes(token_strings)
    if errors:
        print(f"{len(errors)} parse error(s):")
        for e in errors:
            print(" ", e)
    write_midi(notes, output_path)
    return notes, errors

def play_sequences(indices: list[int], dataset_path: str, midi_out: str = "data/midi/preview.midi", wav_out: str = "data/outputs/preview.wav"):
    """Grab rows by index from the dataset, concatenate, decode to MIDI, and play."""
    arr = np.load(dataset_path, mmap_mode="r")
    tokens = arr[indices].flatten()
    parse_tokens_to_midi(tokens, output_path=midi_out)
    create_and_play_audio(filepath=midi_out, output_file=wav_out)


if __name__ == "__main__":
    # tokens = tokenize_sample("data/maestro-v3.0.0/2018/MIDI-Unprocessed_Chamber2_MID--AUDIO_09_R3_2018_wav--1.midi")
    # print(len(tokens))
    # notes, errors = parse_tokens_to_midi(tokens, output_path="data/midi/first_test.midi")
    # print(len(notes))
    # # create_and_play_audio("data/maestro-v3.0.0/2018/MIDI-Unprocessed_Chamber2_MID--AUDIO_09_R3_2018_wav--1.midi", output_file="data/outputs/output.wav")
    # create_and_play_audio(filepath="data/midi/first_test.midi", output_file="data/outputs/first_test.wav")
    play_sequences([7777], dataset_path="data/tokenized_dataset.npy", midi_out="data/midi/preview.midi", wav_out="data/outputs/preview.wav")



