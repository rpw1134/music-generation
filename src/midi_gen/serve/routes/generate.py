import os
import tempfile

import torch
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from midi_gen.model.inference.base_inference import generate_sample
from midi_gen.serve.schemas.generate import GenerateRequest

router = APIRouter()


@router.post("/generate")
async def generate(req: GenerateRequest, request: Request):
    model  = request.app.state.model
    device = request.app.state.device

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    seed_tensor: torch.Tensor | None = None
    if req.seed:
        seed_tensor = torch.tensor([req.seed], dtype=torch.long, device=device)

    try:
        with tempfile.TemporaryDirectory() as tmp:
            midi_out = os.path.join(tmp, "generated.midi")
            wav_out  = os.path.join(tmp, "generated.wav")
            generate_sample(
                model=model,
                midi_out=midi_out,
                wav_out=wav_out,
                max_length=req.max_length,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
                pitch_penalty=req.pitch_penalty,
                pitch_penalty_window=req.pitch_penalty_window,
                seed=seed_tensor,
            )
            with open(wav_out, "rb") as f:
                wav_bytes = f.read()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return Response(content=wav_bytes, media_type="audio/wav")
