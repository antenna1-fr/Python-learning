from __future__ import annotations
from pathlib import Path
from typing import Literal, ClassVar
from pydantic import BaseModel, ConfigDict, Field
from garage import Car, Racetrack


class Config(BaseModel):
    weather: Literal["Sunny", "Damp", "Wet", "Soaked"]
    laps: int = Field(gt=0, lt=16)
    seed: int = 42

    car : Car
    racetrack: Racetrack

    model_config: ClassVar[ConfigDict] = {
        "strict" : True,
        "extra": "forbid"         
    }

def load_config(path: Path) -> Config:
    return Config.model_validate_json(path.read_text())
