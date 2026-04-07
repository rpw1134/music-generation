from matplotlib import pyplot as plt
import pretty_midi

midi = pretty_midi.PrettyMIDI("data/maestro-v3.0.0/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.midi")

for instrument in midi.instruments:
    print(f"Instrument: {instrument.name}, Notes: {len(instrument.notes)}")
    for note in instrument.notes[:10]:
        print(f"  pitch={note.pitch}, start={note.start:.2f}, end={note.end:.2f}, velocity={note.velocity}")
print(midi.get_end_time())

# piano_roll = midi.get_piano_roll(fs=1)
# plt.figure(figsize=(14, 4))
# plt.imshow(piano_roll, aspect='auto', origin='lower')
# plt.xlabel('Time')
# plt.ylabel('Pitch')
# plt.show()

import subprocess


subprocess.run([
    'fluidsynth',
    '-ni',                          # -n: no MIDI input, -i: no interactive shell
    '-g', '1.0',                    # gain
    '-F', 'output.wav',  # write to file
    'data/GeneralUser-GS/GeneralUser-GS.sf2',
    'data/maestro-v3.0.0/2018/MIDI-Unprocessed_Chamber2_MID--AUDIO_09_R3_2018_wav--1.midi'
])

subprocess.run(['afplay', 'output.wav'])  # macOS