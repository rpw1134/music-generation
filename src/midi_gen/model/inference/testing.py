import torch

from midi_gen.model.models.GPTMidiV1 import GPTMidiV1
from midi_gen.model.inference.base_inference import create_sample_tokens
from midi_gen.data_management.tokenizing import create_vocabulary, reconstruct_notes
from midi_gen.data_management.midi_io import write_midi
from midi_gen.exploration.midi_test import create_and_play_audio


def generate_random_sample(
    model_path: str,
    midi_out: str = "data/midi/generated.midi",
    wav_out: str = "data/outputs/generated.wav",
    max_length: int = 1024,
    temperature: float = 1.0,
    top_k: int = 0,
):
    # device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # load model and weights from checkpoint
    model = GPTMidiV1()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # generate token sequence
    tokens = create_sample_tokens(model, max_length=max_length, temperature=temperature, top_k=top_k)
    token_indices = tokens[0].tolist()

    # decode token indices to note events
    _, inverse = create_vocabulary()
    token_strings = [inverse[t] for t in token_indices]
    notes, errors = reconstruct_notes(token_strings)
    if errors:
        print(f"{len(errors)} decode error(s):")
        for e in errors:
            print(" ", e)

    # write midi and play
    write_midi(notes, midi_out)
    create_and_play_audio(filepath=midi_out, output_file=wav_out)

    return token_indices, notes


if __name__ == "__main__":
    generate_random_sample("src/midi_gen/model/models/midiv1_best.pt", temperature=0.7, top_k=40)
