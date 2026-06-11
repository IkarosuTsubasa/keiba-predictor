from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class RecentRun(BaseModel):
    race_id: str | None = None
    date: str | None = None
    race_name: str | None = None
    course: str | None = None
    surface: str | None = None
    distance: int | None = None
    track_condition: str | None = None
    finish: int | None = None
    field_size: int | None = None
    jockey: str | None = None
    odds: float | None = None
    popularity: int | None = None
    passing_order: str | None = None
    corner_positions: list[int] | None = None
    final_3f: float | None = None
    margin: str | None = None


class HorseEntry(BaseModel):
    horse_no: int
    frame_no: int | None = None
    horse_id: str | None = None
    horse_name: str
    jockey: str | None = None
    carried_weight: float | None = None
    odds: float | None = None
    popularity: int | None = None
    recent_runs: list[RecentRun] = Field(default_factory=list)


class RaceInfo(BaseModel):
    race_id: str
    race_name: str | None = None
    race_date: str | None = None
    course: str | None = None
    surface: str | None = None
    distance: int | None = None
    track_condition: str | None = None
    weather: str | None = None
    source: str | None = None
    scope_key: str | None = None


class RaceData(BaseModel):
    race_info: RaceInfo
    horses: list[HorseEntry] = Field(default_factory=list)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "RaceData":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(payload)
