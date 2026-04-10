from midi_gen.data_management.midi_io import file_path_to_vector, write_midi
from midi_gen.data_management.tokenizing import create_vocabulary, quantize_velocity, get_time_shift_bin, reconstruct_notes
import numpy as np

from midi_gen.exploration.midi_test import create_and_play_audio


def tokenize_sample(file_path: str) -> np.ndarray:
    """Tokenize a single MIDI file into a 1D array of integer token indices.

    Each MIDI note is expanded into three events: a VELOCITY event followed by a
    NOTE_ON and a NOTE_OFF, all anchored to their absolute timestamp. Events are
    then sorted by time (ties broken by pitch) and converted to tokens.

    Token sequence design:
    - Begins with <SOS> and ends with <EOS>.
    - TIME_SHIFT tokens are emitted before each group of simultaneous events to
      encode the elapsed time since the last group. Gaps smaller than 5ms are
      collapsed (treated as simultaneous) to avoid cluttering the sequence with
      near-zero shifts. Time is encoded on a log scale across 157 bins (range
      0.01–1.0s); values exceeding 1s are split into multiple tokens.
    - VELOCITY is emitted before its corresponding NOTE_ON so the model sees
      dynamics before the pitch, matching the natural performance order.
    - NOTE_ON and NOTE_OFF are separate token types (ON_1–ON_128, OFF_1–OFF_128),
      giving the model an explicit signal for note boundaries.
    - Pitches are 1-indexed (PITCH_1–PITCH_128) and velocities are quantized
      into 32 bins (VELOCITY_1–VELOCITY_32).

    Returns a numpy array of integer indices into the vocabulary.
    """
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
            tokens.append(vocab[f"<ON_{pitch}>"])
        elif event == "off":
            tokens.append(vocab[f"<OFF_{pitch}>"])
    tokens.append(vocab["<EOS>"])
    return np.array(tokens)

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

if __name__ == "__main__":
    tokens = tokenize_sample("data/maestro-v3.0.0/2018/MIDI-Unprocessed_Chamber2_MID--AUDIO_09_R3_2018_wav--1.midi")
    print(len(tokens))
    notes, errors = parse_tokens_to_midi(tokens, output_path="data/midi/first_test.midi")
    print(len(notes))
    # create_and_play_audio("data/maestro-v3.0.0/2018/MIDI-Unprocessed_Chamber2_MID--AUDIO_09_R3_2018_wav--1.midi", output_file="data/outputs/output.wav")
    create_and_play_audio(filepath="data/midi/first_test.midi", output_file="data/outputs/first_test.wav")




