"""Filter the Lakh Clean dataset to piano-only files.

A file is considered piano-only if:
  - Every non-drum instrument has a GM program in the piano family (0–7)
  - At least MIN_NOTES notes are present across all instruments
  - No drum tracks are present (channel 9 / is_drum=True)

Run as a script to scan a directory and report the qualifying files:
    uv run python -m midi_gen.data_management.lakh_filter --src data/lakh_clean --out data/lakh_piano_files.txt
"""

import argparse
from pathlib import Path

import pretty_midi

# General MIDI programs 0–7 are the piano family:
# 0 Acoustic Grand Piano, 1 Bright Acoustic Piano, 2 Electric Grand Piano,
# 3 Honky-tonk Piano, 4 Electric Piano 1, 5 Electric Piano 2,
# 6 Harpsichord, 7 Clavinet
PIANO_PROGRAMS = set(range(8))
MIN_NOTES = 50


def is_piano_only(path: str | Path) -> tuple[bool, int]:
    """Return (qualifies, note_count) for a MIDI file.

    Qualifies means: no drums, all instruments are piano-family, >= MIN_NOTES notes.
    Returns (False, 0) on any parse error.
    """
    try:
        pm = pretty_midi.PrettyMIDI(str(path))
    except Exception:
        return False, 0

    if not pm.instruments:
        return False, 0

    note_count = 0
    for instr in pm.instruments:
        if instr.is_drum:
            return False, 0
        if instr.program not in PIANO_PROGRAMS:
            return False, 0
        note_count += len(instr.notes)

    return note_count >= MIN_NOTES, note_count


def scan(src: Path) -> list[Path]:
    """Walk src recursively and return all piano-only MIDI files."""
    candidates = list(src.rglob("*.mid")) + list(src.rglob("*.midi"))
    qualifying = []
    total = len(candidates)

    for i, path in enumerate(candidates):
        if i % 500 == 0:
            print(f"  {i}/{total} scanned, {len(qualifying)} qualifying so far...")
        ok, _ = is_piano_only(path)
        if ok:
            qualifying.append(path)

    return qualifying


def main():
    parser = argparse.ArgumentParser(description="Filter Lakh Clean to piano-only MIDI files.")
    parser.add_argument("--src", required=True, help="Root directory of Lakh Clean dataset")
    parser.add_argument("--out", default="data/lakh_piano_files.txt", help="Output file list path")
    args = parser.parse_args()

    src = Path(args.src)
    if not src.exists():
        raise FileNotFoundError(f"Source directory not found: {src}")

    print(f"Scanning {src} ...")
    files = scan(src)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(str(f) for f in files) + "\n")

    print(f"\nDone. {len(files)} piano-only files found.")
    print(f"File list written to {out}")


if __name__ == "__main__":
    main()
