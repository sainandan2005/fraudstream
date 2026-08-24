from __future__ import annotations

import json
import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from prometheus_client import Counter, make_asgi_app
from pydantic import BaseModel
from redis import Redis

from common.config import settings
from common.db import STATUSES, get_pool
from detector.features import PROFILE_KEY, read_window_count, read_window_sum

RECENT_ALERTS_KEY = "alerts:recent"
RULE_HITS_KEY = "stats:rule_hits"
STATS_KEY = "stats"
LABELS_KEY = "stats:labels"
TXNS_RECENT_KEY = "txns:recent"
CARD_TXNS_KEY = "card:{card}:txns"
STATIC_DIR = Path(__file__).parent / "static"


class LabelRequest(BaseModel):
    status: str


LABELS_PROM = Counter(
    "fraudstream_labels_total", "analyst labels applied", ["status"]
)

app = FastAPI(title="FraudStream Alerts API", version="1.0.0")
app.mount("/metrics", make_asgi_app())
redis = Redis.from_url(
    os.environ.get("REDIS_URL", settings.redis_url), decode_responses=True
)


@app.get("/", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/healthz")
def healthz() -> dict:
    health = {"status": "ok", "redis": bool(redis.ping()), "postgres": False}
    try:
        with get_pool().connection() as conn:
            health["postgres"] = conn.execute("SELECT 1").fetchone() is not None
    except Exception:  # noqa: BLE001, S110 - any failure means "postgres down" for the probe
        pass
    return health


def _row_to_alert(row) -> dict:
    (
        alert_id, txn_id, created_ms, card_id, user_id, merchant_id,
        city, amount, currency, lat, lon, risk_score, ml_score, rules, status,
    ) = row
    return {
        "alert_id": alert_id,
        "created_ms": created_ms,
        "txn": {
            "txn_id": txn_id,
            "card_id": card_id,
            "user_id": user_id,
            "merchant_id": merchant_id,
            "city": city,
            "amount": float(amount),
            "currency": currency,
            "lat": lat,
            "lon": lon,
        },
        "risk_score": risk_score,
        "ml_score": ml_score,
        "triggered_rules": rules if isinstance(rules, list) else json.loads(rules),
        "status": status,
    }


def alerts_from_postgres(limit: int) -> list[dict] | None:
    try:
        with get_pool().connection() as conn:
            rows = conn.execute(
                "SELECT alert_id, txn_id, created_ms, card_id, user_id, merchant_id,"
                " city, amount, currency, lat, lon, risk_score, ml_score, rules, status"
                " FROM alerts ORDER BY created_ms DESC LIMIT %s",
                (limit,),
            ).fetchall()
        return [_row_to_alert(r) for r in rows]
    except Exception:  # noqa: BLE001 - any PG failure triggers the Redis fallback path
        return None


@app.get("/alerts")
def recent_alerts(limit: int = Query(default=50, ge=1, le=1000)) -> list[dict]:
    pg = alerts_from_postgres(limit)
    if pg is not None:
        return pg
    raw = redis.zrevrange(RECENT_ALERTS_KEY, 0, limit - 1)
    return [json.loads(item) for item in raw]


@app.post("/alerts/{alert_id}/label")
def label_alert(alert_id: str, body: LabelRequest) -> dict:
    if body.status not in STATUSES:
        raise HTTPException(status_code=422, detail=f"status must be one of {STATUSES}")
    try:
        with get_pool().connection() as conn:
            row = conn.execute(
                "UPDATE alerts SET status = %s, labeled_ms = %s WHERE alert_id = %s RETURNING txn_id",
                (body.status, int(time.time() * 1000), alert_id),
            ).fetchone()
            conn.commit()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="postgres unavailable") from exc
    if row is None:
        raise HTTPException(status_code=404, detail="alert not found")
    if body.status in ("confirmed_fraud", "false_positive"):
        redis.hincrby(LABELS_KEY, body.status, 1)
    LABELS_PROM.labels(status=body.status).inc()
    return {"alert_id": alert_id, "status": body.status, "txn_id": row[0]}


@app.get("/alerts/{card_id}")
def alerts_for_card(card_id: str, limit: int = Query(default=50, ge=1, le=1000)) -> list[dict]:
    out = []
    for item in redis.zrevrange(RECENT_ALERTS_KEY, 0, -1):
        alert = json.loads(item)
        if alert["txn"]["card_id"] == card_id:
            out.append(alert)
            if len(out) >= limit:
                break
    return out


@app.get("/stats")
def stats() -> dict:
    totals = redis.hgetall(STATS_KEY)
    return {
        "txns_processed": int(totals.get("txns_processed", 0)),
        "alerts_raised": int(totals.get("alerts_raised", 0)),
        "last_processed_ms": int(totals.get("last_processed_ms", 0)),
        "rule_hits": {rule: int(n) for rule, n in redis.hgetall(RULE_HITS_KEY).items()},
        "labels": {s: int(n) for s, n in redis.hgetall(LABELS_KEY).items()},
    }


@app.get("/transactions")
def transactions(limit: int = Query(default=100, ge=1, le=1000)) -> list[dict]:
    raw = redis.lrange(TXNS_RECENT_KEY, 0, limit - 1)
    return [json.loads(item) for item in raw]


@app.get("/trace/{txn_id}")
def trace_txn(txn_id: str) -> dict:
    for item in redis.lrange(TXNS_RECENT_KEY, 0, -1):
        record = json.loads(item)
        if record["txn"]["txn_id"] == txn_id:
            return _with_card_state(record["txn"]["card_id"], record)
    raise HTTPException(status_code=404, detail="txn not found in recent buffer")


def _with_card_state(card_id: str, payload: dict) -> dict:
    profile = redis.hgetall(PROFILE_KEY.format(card=card_id))
    return {
        **payload,
        "card_state": {
            "blacklisted": bool(redis.sismember("blacklist:cards", card_id)),
            "profile": profile,
            "velocity": {
                "count_window_sec": settings.velocity_count_window_sec,
                "count": read_window_count(redis, card_id),
                "sum_window_sec": settings.velocity_sum_window_sec,
                "sum": read_window_sum(redis, card_id),
            },
        },
    }


@app.get("/cards/{card_id}")
def card_detail(card_id: str) -> dict:
    raw = redis.lrange(CARD_TXNS_KEY.format(card=card_id), 0, -1)
    txns = [json.loads(item) for item in raw]
    if not txns:
        recent = [json.loads(item) for item in redis.lrange(TXNS_RECENT_KEY, 0, -1)]
        txns = [r for r in recent if r["txn"]["card_id"] == card_id]
    if not txns and not redis.exists(PROFILE_KEY.format(card=card_id)):
        raise HTTPException(status_code=404, detail=f"no history for {card_id}")
    return {
        "card_id": card_id,
        "txn_count": len(txns),
        "flagged_count": sum(1 for t in txns if t["score"] >= settings.alert_threshold),
        "transactions": [_with_card_state(card_id, t) for t in txns],
    }
