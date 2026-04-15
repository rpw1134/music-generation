from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    seed: list[int] | None      = Field(default=None, description="Token indices to condition on. None = random start.")
    temperature: float          = Field(default=1.0,  ge=0.0,  le=5.0)
    top_k: int                  = Field(default=0,    ge=0)
    top_p: float                = Field(default=0.0,  ge=0.0,  le=1.0)
    max_length: int             = Field(default=1024, ge=1,    le=4096)
    pitch_penalty: float        = Field(default=1.0,  ge=1.0,  le=5.0,  description="Repetition penalty applied to ON pitch tokens only. 1.0 = disabled. Try 1.1–1.5 to reduce note looping.")
    pitch_penalty_window: int   = Field(default=64,   ge=1,    le=2048, description="Number of recent tokens to scan for repeated pitches. Smaller = more local; larger = longer memory.")


