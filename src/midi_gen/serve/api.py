import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from midi_gen.model.models.GPTMidiV1 import GPTMidiV1
from midi_gen.serve.routes.generate import router as generate_router

from dotenv import load_dotenv

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = os.getenv("MODEL_PATH")
    if not model_path:
        raise RuntimeError("MODEL_PATH environment variable is not set.")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}

    max_seq_len = state_dict["transformer_blocks.0.rope_cos"].shape[2]
    d_model     = state_dict["embedding.weight"].shape[1]
    num_layers  = len({k.split(".")[1] for k in state_dict if k.startswith("transformer_blocks.")})
    model = GPTMidiV1(max_seq_len=max_seq_len, d_model=d_model, num_layers=num_layers)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    app.state.model  = model
    app.state.device = device

    print(f"Model loaded: {model_path}  device={device}  d_model={d_model}  num_layers={num_layers}  max_seq_len={max_seq_len}")
    yield

    app.state.model  = None
    app.state.device = None


app = FastAPI(title="MusicGen API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

app.include_router(generate_router)
