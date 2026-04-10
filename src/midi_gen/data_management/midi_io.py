import pretty_midi
import numpy as np
from midi_gen.data_management.tokenizing import dequantize_velocity


def midi_to_notes(filepath):
    midi_client = pretty_midi.PrettyMIDI(filepath)
    notes = []
    for instr in midi_client.instruments:
        for note in instr.notes:
            notes.append([note.start, note.end, note.pitch, note.velocity])
    return notes


def notes_to_vector(notes) -> np.ndarray:
    return np.array(notes)


def file_path_to_vector(file_path) -> np.ndarray:
    return notes_to_vector(midi_to_notes(file_path))


def save_vector_to_file(filepath, vector):
    np.save(filepath, vector)


def notes_to_pretty_midi(notes: list[tuple]) -> pretty_midi.PrettyMIDI:
    """Convert a list of (start, end, pitch, velocity_bin) tuples to a PrettyMIDI object.

    Velocity bins (1-indexed, 32 bins) are scaled back to MIDI velocity range 0-127.
    """
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)
    for start, end, pitch, velocity_bin in notes:
        midi_velocity = dequantize_velocity(velocity_bin)
        duration = max(end, start + 0.05)  # guard against zero-length notes
        piano.notes.append(pretty_midi.Note(
            velocity=midi_velocity, pitch=pitch, start=start, end=duration
        ))
    pm.instruments.append(piano)
    return pm


def write_midi(notes: list[tuple], output_path: str) -> None:
    """Write a list of (start, end, pitch, velocity_bin) tuples to a MIDI file."""
    notes_to_pretty_midi(notes).write(output_path)
