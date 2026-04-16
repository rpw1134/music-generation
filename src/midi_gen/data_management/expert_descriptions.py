"""
Per-bar expert description extraction for REMI+ conditioning.

Each bar in a tokenized MIDI sequence is preceded by a short description header —
a fixed-length sequence of tokens encoding note density, pitch range, velocity,
polyphony, and time signature. The model learns to generate notes that match the
description, forcing bar-level structural reasoning.

Typical usage
-------------
    pm = pretty_midi.PrettyMIDI(filepath)
    bar_features = compute_bar_features(pm)

    vocab, inverse = create_description_vocabulary()
    boundaries = DEFAULT_BOUNDARIES  # replace with fit_boundaries() output on ADL corpus

    per_bar_tokens = bar_features_to_tokens(bar_features, vocab, boundaries)
    # per_bar_tokens[i] is a list of token strings for bar i, e.g.:
    # ["<TIME_SIG_4_4>", "<DENSITY_4>", "<PITCH_LOW_3>", "<PITCH_HIGH_5>", "<VEL_MEAN_6>", "<POLY_1>"]
"""

import glob as _glob
import numpy as np
import pretty_midi
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POSITION_RESOLUTION = 16  # sub-divisions per bar (shared with tokenizing_remi.py)

# Time signatures given explicit tokens. Anything not in this list → TIME_SIG_OTHER.
# Update after surveying the ADL corpus (see remi_bar_exploration.ipynb §8).
KNOWN_TIME_SIGS = ["4_4", "3_4", "6_8", "2_4", "2_2"]

# Placeholder bin boundaries for each feature.
# Replace with output of fit_boundaries() computed over the full ADL corpus.
# Each array has (n_bins - 1) split points; np.searchsorted maps a value to bin 0..n_bins-1.
DEFAULT_BOUNDARIES: dict[str, np.ndarray] = {
    "density":    np.array([2.2, 3.3, 4.2, 5.3, 6.6, 8.5, 11.9]),
    "pitch_low":    np.array([36.0, 41.0, 44.0, 48.0, 52.0, 55.0, 60.0]),
    "pitch_high":    np.array([64.0, 66.0, 69.0, 71.0, 74.0, 77.0, 82.0]),
    "vel_mean":    np.array([58.7, 70.0, 78.7, 86.5, 95.0, 100.0, 114.0]),
    "poly":    np.array([2.3, 3.0, 4.0]),
}


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def get_time_sig_at(pm: pretty_midi.PrettyMIDI, t: float) -> pretty_midi.TimeSignature:
    """Return the active TimeSignature at time t.

    pretty_midi always inserts a 4/4 default at t=0, so this never returns None.
    """
    active = pm.time_signature_changes[0]
    for ts in pm.time_signature_changes:
        if ts.time <= t:
            active = ts
    return active


def get_tempo_at(pm: pretty_midi.PrettyMIDI, t: float) -> float:
    """Return the active tempo (BPM) at time t."""
    times, tempos = pm.get_tempo_changes()
    if len(tempos) == 0:
        return 120.0
    idx = int(np.searchsorted(times, t, side="right")) - 1
    return float(tempos[max(0, idx)])


def collect_notes(pm: pretty_midi.PrettyMIDI) -> list[tuple]:
    """Flatten all instruments into a sorted list of (start, end, pitch, velocity) tuples."""
    notes = []
    for instr in pm.instruments:
        for note in instr.notes:
            notes.append((note.start, note.end, note.pitch, note.velocity))
    notes.sort()
    return notes


def get_position_in_bar(t: float, bar_start: float, bar_end: float,
                        resolution: int = POSITION_RESOLUTION) -> int:
    """Quantize a note onset to a position index 0..resolution-1 within its bar.

    Uses a fixed grid regardless of time signature — a 3/4 and 4/4 bar both map
    to positions 0–15. The model infers rhythmic density from the note pattern.
    """
    bar_dur = bar_end - bar_start
    if bar_dur <= 0:
        return 0
    offset = (t - bar_start) / bar_dur
    offset = max(0.0, min(offset, 1.0 - 1e-9))
    return int(offset * resolution)


def quantize_to_bin(value: float, boundaries: np.ndarray) -> int:
    """Map a scalar value to a bin index using pre-computed split points.

    Args:
        value:      The feature value to quantize.
        boundaries: Sorted array of (n_bins - 1) split points.

    Returns:
        Integer bin index in range [0, len(boundaries)].
    """
    return int(np.searchsorted(boundaries, value, side="right"))


# ---------------------------------------------------------------------------
# Bar feature extraction
# ---------------------------------------------------------------------------

def _compute_mean_polyphony(bar_notes: list[tuple]) -> float:
    """Estimate mean polyphony within a bar.

    For each note onset, counts how many other notes are sounding at that moment.
    Returns the mean across all onsets.
    """
    if not bar_notes:
        return 0.0
    starts = np.array([n[0] for n in bar_notes])
    ends   = np.array([n[1] for n in bar_notes])
    counts = [(starts <= s) & (ends > s) for s in starts]
    return float(np.mean([c.sum() for c in counts]))


def compute_bar_features(pm: pretty_midi.PrettyMIDI) -> list[dict]:
    """Extract per-bar musical features from a PrettyMIDI object.

    Returns a list of dicts (one per bar) with the following keys:
        bar_idx        int    — 0-indexed bar number
        bar_start      float  — bar start time in seconds
        bar_dur        float  — bar duration in seconds
        time_sig       str    — e.g. "4_4", "3_4" (underscored for use in token names)
        note_count     int    — total notes in bar
        note_density   float  — notes per second
        pitch_min      int    — lowest MIDI pitch (0 if empty)
        pitch_max      int    — highest MIDI pitch (0 if empty)
        pitch_mean     float  — mean MIDI pitch (0.0 if empty)
        vel_mean       float  — mean MIDI velocity 0–127 (0.0 if empty)
        poly_mean      float  — mean simultaneous notes at each onset (0.0 if empty)

    Bars with duration < 0.05s are skipped (artefacts from tempo changes near the end).
    """
    notes     = collect_notes(pm)
    downbeats = pm.get_downbeats()

    if len(downbeats) == 0 or len(notes) == 0:
        return []

    note_starts = np.array([n[0] for n in notes])
    bar_indices = np.searchsorted(downbeats, note_starts, side="right") - 1
    bar_indices = np.clip(bar_indices, 0, len(downbeats) - 1)

    end_time = pm.get_end_time()
    bar_features = []

    for bar_idx in range(len(downbeats)):
        bar_start = float(downbeats[bar_idx])
        bar_end   = float(downbeats[bar_idx + 1]) if bar_idx + 1 < len(downbeats) else end_time
        bar_dur   = bar_end - bar_start

        if bar_dur < 0.05:
            continue

        ts       = get_time_sig_at(pm, bar_start)
        time_sig = f"{ts.numerator}_{ts.denominator}"

        mask      = bar_indices == bar_idx
        bar_notes = [notes[i] for i in np.where(mask)[0]]

        if bar_notes:
            pitches = [n[2] for n in bar_notes]
            vels    = [n[3] for n in bar_notes]
            features = {
                "bar_idx":      bar_idx,
                "bar_start":    round(bar_start, 4),
                "bar_dur":      round(bar_dur, 4),
                "time_sig":     time_sig,
                "note_count":   len(bar_notes),
                "note_density": round(len(bar_notes) / bar_dur, 3),
                "pitch_min":    int(min(pitches)),
                "pitch_max":    int(max(pitches)),
                "pitch_mean":   round(float(np.mean(pitches)), 2),
                "vel_mean":     round(float(np.mean(vels)), 2),
                "poly_mean":    round(_compute_mean_polyphony(bar_notes), 2),
            }
        else:
            features = {
                "bar_idx":      bar_idx,
                "bar_start":    round(bar_start, 4),
                "bar_dur":      round(bar_dur, 4),
                "time_sig":     time_sig,
                "note_count":   0,
                "note_density": 0.0,
                "pitch_min":    0,
                "pitch_max":    0,
                "pitch_mean":   0.0,
                "vel_mean":     0.0,
                "poly_mean":    0.0,
            }

        bar_features.append(features)

    return bar_features


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

def create_description_vocabulary(
    n_bins: int = 8,
    n_poly_bins: int = 4,
    time_sigs: list[str] = KNOWN_TIME_SIGS,
) -> tuple[dict[str, int], list[str]]:
    """Build the description token vocabulary.

    Returns (vocab, inverse) using the same convention as tokenizing.create_vocabulary():
        vocab:   dict mapping token string → integer index (0-based within this vocab)
        inverse: list mapping integer index → token string

    The indices here are local to the description vocab. The REMI+ tokenizer is
    responsible for offsetting them when merging into the full vocabulary.

    Token groups:
        TIME_SIG_*      — one per entry in time_sigs, plus TIME_SIG_OTHER
        DENSITY_*       — 0..n_bins-1
        PITCH_LOW_*     — 0..n_bins-1
        PITCH_HIGH_*    — 0..n_bins-1
        VEL_MEAN_*      — 0..n_bins-1
        POLY_*          — 0..n_poly_bins-1
    """
    vocab: dict[str, int] = {}
    inverse: list[str] = []

    def _add(name: str) -> None:
        vocab[name] = len(inverse)
        inverse.append(name)

    for sig in time_sigs:
        _add(f"<TIME_SIG_{sig}>")
    _add("<TIME_SIG_OTHER>")

    for group, count in [
        ("DENSITY",    n_bins),
        ("PITCH_LOW",  n_bins),
        ("PITCH_HIGH", n_bins),
        ("VEL_MEAN",   n_bins),
        ("POLY",       n_poly_bins),
    ]:
        for i in range(count):
            _add(f"<{group}_{i}>")

    return vocab, inverse


# ---------------------------------------------------------------------------
# Token conversion
# ---------------------------------------------------------------------------

def bar_features_to_tokens(
    bar_features: list[dict],
    vocab: dict[str, int],
    boundaries: dict[str, np.ndarray] = DEFAULT_BOUNDARIES,
) -> list[list[str]]:
    """Convert per-bar feature dicts into description token sequences.

    Args:
        bar_features: output of compute_bar_features().
        vocab:        output of create_description_vocabulary().
        boundaries:   dict mapping feature name → bin split points array.
                      Keys: "density", "pitch_low", "pitch_high", "vel_mean", "poly".

    Returns:
        List of length len(bar_features). Each element is a list of token strings
        for that bar's description header, e.g.:
            ["<TIME_SIG_4_4>", "<DENSITY_4>", "<PITCH_LOW_3>",
             "<PITCH_HIGH_5>", "<VEL_MEAN_6>", "<POLY_1>"]

        Empty bars (note_count == 0) still receive a description header so the
        model learns to generate silence when conditioned on zero density.
    """
    result = []
    for bar in bar_features:
        sig_token = f"<TIME_SIG_{bar['time_sig']}>"
        if sig_token not in vocab:
            sig_token = "<TIME_SIG_OTHER>"

        tokens = [
            sig_token,
            f"<DENSITY_{quantize_to_bin(bar['note_density'], boundaries['density'])}>",
            f"<PITCH_LOW_{quantize_to_bin(bar['pitch_min'], boundaries['pitch_low'])}>",
            f"<PITCH_HIGH_{quantize_to_bin(bar['pitch_max'], boundaries['pitch_high'])}>",
            f"<VEL_MEAN_{quantize_to_bin(bar['vel_mean'], boundaries['vel_mean'])}>",
            f"<POLY_{quantize_to_bin(bar['poly_mean'], boundaries['poly'])}>",
        ]
        result.append(tokens)
    return result


# ---------------------------------------------------------------------------
# Corpus boundary fitting
# ---------------------------------------------------------------------------

def fit_boundaries(
    source: str | list[str | Path],
    n_bins: int = 8,
    n_poly_bins: int = 4,
) -> dict[str, np.ndarray]:
    """Compute data-driven bin boundaries from a corpus of MIDI files.

    Uses percentile-based bins so each bin covers an equal share of the data
    distribution. Run this once on the full ADL corpus to replace DEFAULT_BOUNDARIES.

    Args:
        source:      Either a glob pattern string (e.g. "data/adl_clean/**/*.mid")
                     or an explicit list of file paths.
        n_bins:      Number of bins for density, pitch, and velocity features.
        n_poly_bins: Number of bins for polyphony.

    Returns:
        Dict matching the shape of DEFAULT_BOUNDARIES, with empirical split points.
        Prints the result as copy-pasteable Python so you can hardcode it.
    """
    if isinstance(source, str):
        file_list = _glob.glob(source, recursive=True)
        print(f"fit_boundaries: found {len(file_list)} files matching '{source}'")
    else:
        file_list = list(source)

    densities, pitch_lows, pitch_highs, vel_means, poly_means = [], [], [], [], []
    errors = 0

    for path in file_list:
        try:
            pm = pretty_midi.PrettyMIDI(str(path))
            for bar in compute_bar_features(pm):
                if bar["note_count"] == 0:
                    continue
                densities.append(bar["note_density"])
                pitch_lows.append(bar["pitch_min"])
                pitch_highs.append(bar["pitch_max"])
                vel_means.append(bar["vel_mean"])
                poly_means.append(bar["poly_mean"])
        except Exception:
            errors += 1

    if errors:
        print(f"fit_boundaries: skipped {errors} files that failed to parse")

    def _edges(values: list[float], n: int) -> np.ndarray:
        """Return (n-1) percentile-based split points from a value list."""
        percentiles = np.linspace(0, 100, n + 1)[1:-1]  # exclude 0th and 100th
        return np.percentile(values, percentiles)

    boundaries = {
        "density":    _edges(densities,   n_bins),
        "pitch_low":  _edges(pitch_lows,  n_bins),
        "pitch_high": _edges(pitch_highs, n_bins),
        "vel_mean":   _edges(vel_means,   n_bins),
        "poly":       _edges(poly_means,  n_poly_bins),
    }

    print("\n# --- paste into DEFAULT_BOUNDARIES in expert_descriptions.py ---")
    print("DEFAULT_BOUNDARIES: dict[str, np.ndarray] = {")
    for key, arr in boundaries.items():
        vals = ", ".join(f"{v:.1f}" for v in arr)
        print(f'    "{key}":    np.array([{vals}]),')
    print("}")

    return boundaries


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def extract_expert_descriptions(
    source: str | Path | pretty_midi.PrettyMIDI,
    vocab: dict[str, int],
    boundaries: dict[str, np.ndarray] = DEFAULT_BOUNDARIES,
) -> tuple[list[dict], list[list[str]]]:
    """Extract bar features and description tokens from a MIDI file or PrettyMIDI object.

    Args:
        source:     File path or already-loaded PrettyMIDI instance.
        vocab:      Output of create_description_vocabulary().
        boundaries: Bin split points; defaults to DEFAULT_BOUNDARIES.

    Returns:
        (bar_features, per_bar_tokens)
            bar_features:    list of feature dicts from compute_bar_features().
            per_bar_tokens:  list of token string lists from bar_features_to_tokens().
    """
    pm = source if isinstance(source, pretty_midi.PrettyMIDI) else pretty_midi.PrettyMIDI(str(source))
    bar_features = compute_bar_features(pm)
    per_bar_tokens = bar_features_to_tokens(bar_features, vocab, boundaries)
    return bar_features, per_bar_tokens
