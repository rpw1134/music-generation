from matplotlib import pyplot as plt
import pretty_midi
import numpy as np


# piano_roll = midi.get_piano_roll(fs=1)
# plt.figure(figsize=(14, 4))
# plt.imshow(piano_roll, aspect='auto', origin='lower')
# plt.xlabel('Time')
# plt.ylabel('Pitch')
# plt.show()

# for testing purposes
def create_and_play_audio(filepath = "data/maestro-v3.0.0/2018/MIDI-Unprocessed_Chamber2_MID--AUDIO_09_R3_2018_wav--1.midi", output_file = "output.wav"):
    import subprocess
    subprocess.run([
        'fluidsynth',
        '-ni',                          # -n: no MIDI input, -i: no interactive shell
        '-g', '1.0',                    # gain
        '-F', output_file,  # write to file
        'data/GeneralUser-GS/GeneralUser-GS.sf2',
        filepath
    ])

    subprocess.run(['afplay', output_file])


if __name__ == "__main__":
    pass