import math
from dataclasses import dataclass


@dataclass
class GenerationStats:
    total_tokens: int
    tokens_per_second: float
    note_count: int
    decode_errors: int
    pitch_mean: float
    pitch_min: int
    pitch_max: int
    pitch_std: float
    pitch_histogram: dict[str, int]   # "octave_4" → note count


def compute_generation_stats(
    token_indices: list[int],
    notes: list[tuple],
    elapsed_seconds: float,
    decode_errors: int = 0,
) -> GenerationStats:
    """Compute statistics over a completed generation.

    Args:
        token_indices:   raw list of integer token indices from the model.
        notes:           list of (start, end, pitch, velocity_bin) tuples.
        elapsed_seconds: wall-clock seconds for the generate_sample() call.
        decode_errors:   count of malformed token sequences from reconstruct_notes().

    Returns:
        GenerationStats dataclass — call dataclasses.asdict() to serialize.
    """
    total_tokens = len(token_indices)
    tokens_per_second = round(total_tokens / elapsed_seconds, 2) if elapsed_seconds > 0 else 0.0

    if notes:
        pitches = [n[2] for n in notes]
        pitch_mean = sum(pitches) / len(pitches)
        pitch_min  = min(pitches)
        pitch_max  = max(pitches)
        variance   = sum((p - pitch_mean) ** 2 for p in pitches) / len(pitches)
        pitch_std  = math.sqrt(variance)

        histogram: dict[str, int] = {}
        for p in pitches:
            label = f"octave_{(p // 12) - 1}"
            histogram[label] = histogram.get(label, 0) + 1
    else:
        pitch_mean = pitch_std = 0.0
        pitch_min  = pitch_max = 0
        histogram  = {}

    return GenerationStats(
        total_tokens=total_tokens,
        tokens_per_second=tokens_per_second,
        note_count=len(notes),
        decode_errors=decode_errors,
        pitch_mean=round(pitch_mean, 2),
        pitch_min=pitch_min,
        pitch_max=pitch_max,
        pitch_std=round(pitch_std, 2),
        pitch_histogram=histogram,
    )
