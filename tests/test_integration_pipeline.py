"""End-to-end checks against the running docker compose stack.

These tests hit real Kafka/Redis/Postgres/API surfaces and are skipped
automatically whenever the stack is not reachable, so `pytest -q` stays green
on machines without Docker running. Run them while the stack is up with:

    pytest -q                (includes these)
    pytest -q -m "not integration"   (unit tests only)
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request

import pytest
from redis import Redis

from common.config import settings

API = "http://localhost:8000"


def _get_json(url: str, timeout: float = 4.0):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())


def _stack_reachable() -> bool:
    try:
        Redis.from_url(settings.redis_url, socket_connect_timeout=2).ping()
        _get_json(f"{API}/healthz", timeout=3)
        return True
    except Exception:  # noqa: BLE001 - any connectivity failure just skips the suite
        return False


pytestmark = pytest.mark.skipif(
    not _stack_reachable(), reason="docker compose stack not reachable"
)


def _stats() -> dict:
    return _get_json(f"{API}/stats")


def test_healthz_reports_all_backends():
    health = _get_json(f"{API}/healthz")
    assert health["status"] == "ok"
    assert health["redis"] is True
    assert health["postgres"] is True


def test_detector_heartbeat_is_fresh():
    last = _stats()["last_processed_ms"]
    assert last > 0
    age_sec = (time.time() * 1000 - last) / 1000
    assert age_sec < 15, f"detector heartbeat stale: {age_sec:.1f}s"


def test_pipeline_throughput_is_alive():
    before = _stats()["txns_processed"]
    time.sleep(6)
    after = _stats()["txns_processed"]
    assert after > before, "detector stopped consuming transactions"


def test_stream_records_carry_verdicts_and_low_lag():
    records = _get_json(f"{API}/transactions?limit=25")
    assert len(records) >= 10
    now = time.time() * 1000
    for rec in records:
        assert set(rec) >= {"txn", "score", "rules", "ml_score", "lag_ms"}
        assert rec["score"] >= 0
        assert rec["lag_ms"] < 5000, f"pipeline lag spiked: {rec['lag_ms']}ms"
        assert abs(rec["txn"]["ts_ms"] - now) < 10 * 60 * 1000


def test_alerts_are_scored_by_rules_and_model():
    alerts = _get_json(f"{API}/alerts?limit=30")
    if len(alerts) < 5:
        pytest.skip("not enough alerts generated yet")
    enriched = [a for a in alerts if a.get("ml_score") is not None]
    assert len(enriched) >= max(1, len(alerts) // 3), (
        "too few alerts carried an ML score, retro-patch may be broken"
    )
    for alert in alerts:
        assert alert["risk_score"] >= 70
        assert alert["triggered_rules"], "alert without triggered rules"


def test_label_round_trip_survives_refetch():
    alerts = _get_json(f"{API}/alerts?limit=1")
    assert alerts, "no alerts available to label"
    alert_id = alerts[0]["alert_id"]

    req = urllib.request.Request(
        f"{API}/alerts/{alert_id}/label",
        data=json.dumps({"status": "confirmed_fraud"}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200

    refetched = _get_json(f"{API}/alerts?limit=100")
    match = [a for a in refetched if a["alert_id"] == alert_id]
    assert match and match[0]["status"] == "confirmed_fraud"


def test_training_examples_accumulate():
    import psycopg

    # the containerized postgres is published on host port 5433
    dsn = os.environ.get(
        "TEST_DATABASE_URL", "postgresql://fraud:fraud@localhost:5433/fraudstream"
    )
    with psycopg.connect(dsn, connect_timeout=3) as conn:
        (count,) = conn.execute("SELECT count(*) FROM training_examples").fetchone()
        (fresh,) = conn.execute(
            "SELECT count(*) FROM training_examples WHERE ts_ms > %s",
            ((time.time() * 1000) - 60_000,),
        ).fetchone()
    assert count > 200, "training sink stalled"
    assert fresh > 0, "no new training rows in the last minute"


def test_prometheus_scrapes_the_pipeline():
    query = urllib.parse.quote('fraudstream_txns_processed_total')
    data = _get_json(f"http://localhost:9090/api/v1/query?query={query}")
    assert data["status"] == "success"
    results = [r for r in data["data"]["result"] if float(r["value"][1]) > 0]
    assert results, "prometheus has no live txn counter from any service"
