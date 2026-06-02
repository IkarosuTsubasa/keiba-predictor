from __future__ import annotations

from pydantic import AliasChoices, BaseModel, Field


class ResultTop3(BaseModel):
    first: int = Field(alias="1st")
    second: int = Field(alias="2nd")
    third: int = Field(alias="3rd")

    def model_dump(self, *args, **kwargs):
        kwargs.setdefault("by_alias", True)
        return super().model_dump(*args, **kwargs)


class PayoutItem(BaseModel):
    bet_type: str = Field(validation_alias=AliasChoices("bet_type", "type"))
    combination: str
    payout: int
    popularity: int | None = None


class FinishOrderItem(BaseModel):
    finish: int
    horse_no: int
    horse_name: str
    jockey: str | None = None
    time: str | None = None
    margin: str | None = None
    popularity: int | None = None
    odds: float | None = None


class ResultData(BaseModel):
    race_id: str
    result: ResultTop3
    payouts: list[PayoutItem] = Field(default_factory=list)
    finish_order: list[FinishOrderItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
