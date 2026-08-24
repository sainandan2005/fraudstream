from __future__ import annotations

import time
import uuid

from pydantic import BaseModel, Field


def now_ms() -> int:
    return int(time.time() * 1000)


class Transaction(BaseModel):
    txn_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    ts_ms: int = Field(default_factory=now_ms)
    card_id: str
    user_id: str
    merchant_id: str
    amount: float
    currency: str = "INR"
    lat: float
    lon: float
    city: str = ""
    country: str = "IN"


class Alert(BaseModel):
    alert_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    created_ms: int = Field(default_factory=now_ms)
    txn: Transaction
    risk_score: int
    ml_score: float | None = None
    triggered_rules: list[dict] = []


class ScoreRecord(BaseModel):
    model_config = {"protected_namespaces": ()}

    txn_id: str
    card_id: str
    ml_score: float
    model_version: str = "unknown"
    ts_ms: int = Field(default_factory=now_ms)
