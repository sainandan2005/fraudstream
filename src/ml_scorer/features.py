"""Self-contained online features for the scorer.

Mirrors the detector's semantics but under `ml*`-prefixed keys, so the scorer
maintains its own replicas of the stateful features (windows, geo profile,
spend stats). This removes the ordering race with the detector and lets the
model consume the exact fields it was trained on.
"""

from __future__ import annotations

from redis import Redis

from common.config import settings
from common.models import Transaction
from detector.features import amount_zscore, welford_update

ML_COUNT_KEY = "mlvel:count:{card}"
ML_SUM_KEY = "mlvel:sum:{card}"
ML_USER_KEY = "mlvel:user:{user}"
ML_PROFILE_KEY = "mlprofile:{card}"
ML_SEEN_KEY = "mlseen:{card}"


def _record_count(redis: Redis, key: str, member: str, ts_ms: int, window_sec: int) -> int:
    cutoff = ts_ms - window_sec * 1000
    pipe = redis.pipeline(transaction=False)
    pipe.zremrangebyscore(key, "-inf", cutoff)
    pipe.zadd(key, {member: ts_ms})
    pipe.expire(key, window_sec * 2)
    pipe.zcard(key)
    return int(pipe.execute()[-1])


def _window_sum(redis: Redis, key: str, cutoff_ts_ms: int, window_sec: int, add_member: tuple[str, int] | None = None) -> float:
    cutoff = cutoff_ts_ms - window_sec * 1000
    pipe = redis.pipeline(transaction=False)
    pipe.zremrangebyscore(key, "-inf", cutoff)
    if add_member is not None:
        pipe.zadd(key, {add_member[0]: cutoff_ts_ms})
        pipe.expire(key, window_sec * 2)
    pipe.zrange(key, 0, -1)
    members = pipe.execute()[-1]
    total = 0.0
    for m in members:
        try:
            total += float(m.rsplit("|", 1)[1])
        except (ValueError, IndexError):
            continue
    return round(total, 2)


def _geo_and_update(redis: Redis, key: str, txn: Transaction) -> tuple[float | None, float | None]:
    speed = distance = None
    prev = redis.hgetall(key)
    if prev:
        try:
            plat, plon, pts = float(prev["lat"]), float(prev["lon"]), int(prev["ts_ms"])
            distance = _haversine(plat, plon, txn.lat, txn.lon)
            dt = max((txn.ts_ms - pts) / 1000.0, 1.0)
            speed = distance / (dt / 3600.0)
        except (KeyError, ValueError):
            speed = distance = None
    redis.hset(
        key,
        mapping={"lat": txn.lat, "lon": txn.lon, "ts_ms": txn.ts_ms},
    )
    redis.expire(key, 86400)
    return speed, distance


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    from math import asin, cos, radians, sin, sqrt

    p1, p2 = radians(lat1), radians(lat2)
    dp, dl = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dp / 2) ** 2 + cos(p1) * cos(p2) * sin(dl / 2) ** 2
    return 2 * 6371.0 * asin(sqrt(a))


def _zscore_and_update(redis: Redis, key: str, txn: Transaction) -> tuple[float | None, int]:
    prev = redis.hmget(key, ["n", "mean", "m2"])
    try:
        n = int(prev[0] or 0)
        mean = float(prev[1] or 0.0)
        m2 = float(prev[2] or 0.0)
    except (TypeError, ValueError):
        n, mean, m2 = 0, 0.0, 0.0
    z = amount_zscore(n, mean, m2, txn.amount)
    n2, mean2, m22 = welford_update(n, mean, m2, txn.amount)
    redis.hset(key, mapping={"n": n2, "mean": round(mean2, 4), "m2": round(m22, 4)})
    return z, n2


def observe_and_extract(redis: Redis, txn: Transaction, blacklisted_hits: list[str]) -> dict:
    """Update the scorer's own feature replicas with this transaction, then
    return the raw feature dict (pre-model `build_row` input)."""
    count_key = ML_COUNT_KEY.format(card=txn.card_id)
    sum_key = ML_SUM_KEY.format(card=txn.card_id)
    user_key = ML_USER_KEY.format(user=txn.user_id)
    profile_key = ML_PROFILE_KEY.format(card=txn.card_id)

    window_count = _record_count(redis, count_key, txn.txn_id, txn.ts_ms, settings.velocity_count_window_sec)

    member = f"{txn.txn_id}|{txn.amount:.2f}"
    prior_sum = _window_sum(redis, sum_key, txn.ts_ms, settings.velocity_sum_window_sec)
    window_sum = round(prior_sum + txn.amount, 2)
    _window_sum(redis, sum_key, txn.ts_ms, settings.velocity_sum_window_sec, add_member=(member, txn.ts_ms))

    user_count = _record_count(redis, user_key, txn.txn_id, txn.ts_ms, settings.user_velocity_window_sec)

    speed, distance = _geo_and_update(redis, f"{profile_key}:geo", txn)
    z, samples = _zscore_and_update(redis, f"{profile_key}:stats", txn)

    added = redis.sadd(ML_SEEN_KEY.format(card=txn.card_id), txn.merchant_id)
    redis.expire(ML_SEEN_KEY.format(card=txn.card_id), 30 * 86400)

    return {
        "amount": txn.amount,
        "ts_ms": txn.ts_ms,
        "window_count": window_count,
        "window_sum": window_sum,
        "user_window_count": user_count,
        "new_merchant": bool(added),
        "blacklisted": bool(blacklisted_hits),
        "implied_speed_kmh": speed,
        "geo_distance_km": distance,
        "amount_z": z,
        "amount_samples": samples,
    }
