import pretty_midi
import numpy as np
from collections import defaultdict
from midi_gen.data_management.tokenizing import get_time_shift_by_bin, dequantize_velocity

def midi_to_notes(filepath = "data/maestro-v3.0.0/2018/MIDI-Unprocessed_Chamber2_MID--AUDIO_09_R3_2018_wav--1.midi"):
    midi_client = pretty_midi.PrettyMIDI(filepath)
    notes = []
    for instr in midi_client.instruments:
        for note in instr.notes:
            notes.append([note.start, note.end, note.pitch, note.velocity])
    return notes

def notes_to_vector(notes)->np.ndarray:
    # row is a note, cols are the data points
    return np.array(notes)

def file_path_to_vector(file_path):
    notes = midi_to_notes(file_path)
    vec = notes_to_vector(notes)
    print(vec.shape)
    return notes_to_vector(notes)

def save_vector_to_file(filepath, vector):
    np.save(filepath, vector)

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
    kind = "_".join(parts)
    return kind, int(val)


def reconstruct_notes(token_strings: list[str]) -> tuple[list[tuple], list[dict]]:
    """Convert a decoded token sequence into a list of notes and a list of errors.

    Each note is a tuple of (start, end, pitch, velocity).
    Errors are collected for malformed sequences (e.g. OFF before ON, VELOCITY not
    followed by ON) rather than raising, so the caller can inspect bad tokens.
    """
    notes = []
    errors = []
    active = defaultdict(dict)  # pitch -> {velocity, on, pitch}
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


def notes_to_pretty_midi(notes: list[tuple]) -> pretty_midi.PrettyMIDI:
    """Convert a list of (start, end, pitch, velocity_bin) tuples to a PrettyMIDI object.

    Velocity bins (1-indexed, 32 bins) are scaled back to MIDI velocity range 0-127.
    """
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)
    for start, end, pitch, velocity_bin in notes:
        midi_velocity = dequantize_velocity(velocity_bin)
        duration = max(end, start + 0.05)
        piano.notes.append(pretty_midi.Note(
            velocity=midi_velocity, pitch=pitch, start=start, end=duration
        ))
    pm.instruments.append(piano)
    return pm


def write_midi(notes: list[tuple], output_path: str) -> None:
    """Write a list of (start, end, pitch, velocity_bin) tuples to a MIDI file."""
    pm = notes_to_pretty_midi(notes)
    pm.write(output_path)

if __name__ == "__main__":
    v = file_path_to_vector("data/maestro-v3.0.0/2018/MIDI-Unprocessed_Chamber2_MID--AUDIO_09_R3_2018_wav--1.midi")
