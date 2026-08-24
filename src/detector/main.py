from __future__ import annotations

import json
import logging

from redis import Redis

from common.config import settings
from common.db import get_pool
from common.kafka_io import ensure_topics, make_consumer, make_producer, send_json
from common.metrics import (
    ALERTS_RAISED,
    PROCESS_LATENCY,
    RULE_HITS,
    SCORES_RECEIVED,
    TXNS_PROCESSED,
    start_metrics_server,
)
from common.models import Alert, ScoreRecord, Transaction, now_ms
from detector.features import extract_features, mark_processed, seed_blacklists
from detector.rules import evaluate, risk_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("detector")

STATS_KEY = "stats"
RULE_HITS_KEY = "stats:rule_hits"
RECENT_ALERTS_KEY = "alerts:recent"
RECENT_ALERTS_MAX = 1000
ALERT_BY_TXN_KEY = "alert:{txn_id}"
TXNS_RECENT_KEY = "txns:recent"
TXNS_RECENT_MAX = 1000
CARD_TXNS_KEY = "card:{card}:txns"
CARD_TXNS_MAX = 40
CARD_TXNS_TTL_SEC = 6 * 3600
ML_TXN_KEY = "ml:{txn_id}"
ML_CARD_KEY = "ml:card:{card_id}"
ML_TTL_SEC = 900


def persist_training_example(txn: Transaction, features: dict, score: int) -> None:
    try:
        with get_pool().connection() as conn:
            conn.execute(
                """
                INSERT INTO training_examples (txn_id, ts_ms, card_id, amount,
                    window_count, window_sum, user_window_count, new_merchant,
                    blacklisted, implied_speed_kmh, geo_distance_km, amount_z,
                    amount_samples, risk_score, flagged)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (txn_id) DO NOTHING
                """,
                (
                    txn.txn_id,
                    txn.ts_ms,
                    txn.card_id,
                    features["amount"],
                    features["window_count"],
                    features["window_sum"],
                    features["user_window_count"],
                    features["new_merchant"],
                    features["blacklisted"],
                    features["implied_speed_kmh"],
                    features["geo_distance_km"],
                    features["amount_z"],
                    features["amount_samples"],
                    score,
                    score >= settings.alert_threshold,
                ),
            )
            conn.commit()
    except Exception:
        log.exception("postgres training write failed for txn %s", txn.txn_id)


def persist_alert(alert: Alert) -> None:
    try:
        with get_pool().connection() as conn:
            conn.execute(
                """
                INSERT INTO alerts (alert_id, txn_id, created_ms, card_id, user_id,
                                    merchant_id, city, amount, currency, lat, lon,
                                    risk_score, ml_score, rules, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'new')
                ON CONFLICT (alert_id) DO NOTHING
                """,
                (
                    alert.alert_id,
                    alert.txn.txn_id,
                    alert.created_ms,
                    alert.txn.card_id,
                    alert.txn.user_id,
                    alert.txn.merchant_id,
                    alert.txn.city,
                    alert.txn.amount,
                    alert.txn.currency,
                    alert.txn.lat,
                    alert.txn.lon,
                    alert.risk_score,
                    alert.ml_score,
                    json.dumps(alert.triggered_rules),
                ),
            )
            conn.commit()
    except Exception:
        log.exception("postgres write failed for alert %s", alert.alert_id)


def _parse_ml_score(raw: str | None) -> float | None:
    try:
        return float(raw) if raw is not None else None
    except ValueError:
        return None


def _lookup_ml_score(redis: Redis, txn: Transaction) -> float | None:
    return _parse_ml_score(
        redis.get(ML_TXN_KEY.format(txn_id=txn.txn_id))
        or redis.get(ML_CARD_KEY.format(card_id=txn.card_id))
    )


def process(redis: Redis, producer, txn: Transaction) -> Alert | None:
    with PROCESS_LATENCY.time():
        return _process_inner(redis, producer, txn)


def _process_inner(redis: Redis, producer, txn: Transaction) -> Alert | None:
    features = extract_features(redis, txn)
    hits = evaluate(**features)
    score = risk_score(hits)
    TXNS_PROCESSED.inc()
    for hit in hits:
        RULE_HITS.labels(rule=hit.rule).inc()

    record = {
        "txn": txn.model_dump(),
        "score": score,
        "rules": [{"rule": h.rule, "weight": h.weight, "detail": h.detail} for h in hits],
        "ml_score": None,
        "lag_ms": max(0, now_ms() - txn.ts_ms),
    }
    card_key = CARD_TXNS_KEY.format(card=txn.card_id)
    pipe = redis.pipeline()
    pipe.lpush(TXNS_RECENT_KEY, json.dumps(record))
    pipe.ltrim(TXNS_RECENT_KEY, 0, TXNS_RECENT_MAX - 1)
    pipe.lpush(card_key, json.dumps(record))
    pipe.ltrim(card_key, 0, CARD_TXNS_MAX - 1)
    pipe.expire(card_key, CARD_TXNS_TTL_SEC)
    pipe.hincrby(STATS_KEY, "txns_processed", 1)
    pipe.hset(STATS_KEY, "last_processed_ms", now_ms())
    pipe.execute()
    persist_training_example(txn, features, score)

    if not hits or score < settings.alert_threshold:
        return None

    alert = Alert(
        txn=txn,
        risk_score=score,
        ml_score=_lookup_ml_score(redis, txn),
        triggered_rules=[{"rule": h.rule, "weight": h.weight, "detail": h.detail} for h in hits],
    )
    send_json(producer, settings.alerts_topic, txn.card_id, alert.model_dump())

    member = alert.model_dump_json()
    pipe = redis.pipeline()
    pipe.zadd(RECENT_ALERTS_KEY, {member: alert.created_ms})
    pipe.zremrangebyrank(RECENT_ALERTS_KEY, 0, -(RECENT_ALERTS_MAX + 1))
    pipe.setex(ALERT_BY_TXN_KEY.format(txn_id=txn.txn_id), ML_TTL_SEC, member)
    pipe.hincrby(STATS_KEY, "alerts_raised", 1)
    for hit in hits:
        pipe.hincrby(RULE_HITS_KEY, hit.rule, 1)
    pipe.execute()
    persist_alert(alert)
    ALERTS_RAISED.inc()

    rules_str = ", ".join(f"{h.rule}(+{h.weight})" for h in hits)
    log.warning(
        "ALERT score=%d card=%s txn=%s amount=₹%.2f rules=[%s]",
        score,
        txn.card_id,
        txn.txn_id,
        txn.amount,
        rules_str,
    )
    return alert


def handle_score(redis: Redis, payload: bytes) -> None:
    record = ScoreRecord.model_validate_json(payload)
    SCORES_RECEIVED.inc()
    alert_key = ALERT_BY_TXN_KEY.format(txn_id=record.txn_id)
    raw_alert = redis.get(alert_key)
    pipe = redis.pipeline()
    pipe.set(ML_TXN_KEY.format(txn_id=record.txn_id), record.ml_score, ex=ML_TTL_SEC)
    pipe.set(ML_CARD_KEY.format(card_id=record.card_id), record.ml_score, ex=ML_TTL_SEC)
    if raw_alert:
        patched = json.loads(raw_alert)
        patched["ml_score"] = record.ml_score
        member = json.dumps(patched)
        pipe.zrem(RECENT_ALERTS_KEY, raw_alert)
        pipe.zadd(RECENT_ALERTS_KEY, {member: patched["created_ms"]})
        pipe.setex(alert_key, ML_TTL_SEC, member)
    pipe.execute()
    try:
        with get_pool().connection() as conn:
            conn.execute(
                "UPDATE alerts SET ml_score = %s WHERE txn_id = %s AND ml_score IS NULL",
                (record.ml_score, record.txn_id),
            )
            conn.commit()
    except Exception:
        log.exception("postgres ml patch failed for txn %s", record.txn_id)


def main() -> None:
    start_metrics_server(9100)
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    seed_blacklists(redis)
    ensure_topics([settings.transactions_topic, settings.scores_topic])
    producer = make_producer()
    consumer = make_consumer(
        "detector",
        [settings.transactions_topic, settings.scores_topic],
        auto_commit=False,
    )
    log.info(
        "detector online: topic=%s scores=%s threshold=%d",
        settings.transactions_topic,
        settings.scores_topic,
        settings.alert_threshold,
    )
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                log.error("consumer error: %s", msg.error())
                continue
            try:
                if msg.topic() == settings.scores_topic:
                    handle_score(redis, msg.value())
                    consumer.commit(message=msg)
                    continue
                txn = Transaction.model_validate_json(msg.value())
                if mark_processed(redis, txn.txn_id):
                    process(redis, producer, txn)
                consumer.commit(message=msg)
            except Exception:
                log.exception("failed processing message topic=%s offset=%d", msg.topic(), msg.offset())
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()
        producer.flush(10)


if __name__ == "__main__":
    main()
