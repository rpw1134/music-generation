import base64
import dataclasses
import os
import tempfile
import time

import torch
from fastapi import APIRouter, HTTPException, Request

from midi_gen.model.inference.base_inference import generate_sample
from midi_gen.model.inference.stats import compute_generation_stats
from midi_gen.serve.schemas.generate import GenerateRequest, GenerateResponse

router = APIRouter()


@router.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest, request: Request):
    model  = request.app.state.model
    device = request.app.state.device

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    seed_tensor: torch.Tensor | None = None
    if req.seed:
        seed_tensor = torch.tensor([req.seed], dtype=torch.long, device=device)

    t_start = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            midi_out = os.path.join(tmp, "generated.midi")
            wav_out  = os.path.join(tmp, "generated.wav")
            token_indices, notes, decode_errors = generate_sample(
                model=model,
                midi_out=midi_out,
                wav_out=wav_out,
                max_length=req.max_length,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
                seed=seed_tensor,
            )
            with open(wav_out, "rb") as f:
                wav_b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    elapsed = time.perf_counter() - t_start
    stats = compute_generation_stats(token_indices, notes, elapsed, decode_errors)

    return GenerateResponse(wav_b64=wav_b64, stats=dataclasses.asdict(stats))
