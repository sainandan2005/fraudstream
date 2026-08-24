from __future__ import annotations

import math
import time

from redis import Redis

from common.config import settings
from common.models import Transaction

COUNT_KEY = "vel:count:{card}"
SUM_KEY = "vel:sum:{card}"
USER_COUNT_KEY = "vel:user:{user}"
PROFILE_KEY = "profile:{card}"
SEEN_MERCHANTS_KEY = "seen_m:{card}"
BLACKLIST_CARDS_KEY = "blacklist:cards"
BLACKLIST_MERCHANTS_KEY = "blacklist:merchants"

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def implied_speed_kmh(
    prev_lat: float, prev_lon: float, prev_ts_ms: int, lat: float, lon: float, ts_ms: int
) -> float:
    dt_sec = max((ts_ms - prev_ts_ms) / 1000.0, 1.0)
    distance = haversine_km(prev_lat, prev_lon, lat, lon)
    return distance / (dt_sec / 3600.0)


def record_and_count(redis: Redis, txn: Transaction) -> int:
    key = COUNT_KEY.format(card=txn.card_id)
    cutoff = txn.ts_ms - settings.velocity_count_window_sec * 1000
    pipe = redis.pipeline(transaction=False)
    pipe.zremrangebyscore(key, "-inf", cutoff)
    pipe.zadd(key, {txn.txn_id: txn.ts_ms})
    pipe.expire(key, settings.velocity_count_window_sec * 2)
    pipe.zcard(key)
    return int(pipe.execute()[-1])


def record_and_sum(redis: Redis, txn: Transaction) -> float:
    key = SUM_KEY.format(card=txn.card_id)
    cutoff = txn.ts_ms - settings.velocity_sum_window_sec * 1000
    member = f"{txn.txn_id}|{txn.amount:.2f}"
    pipe = redis.pipeline(transaction=False)
    pipe.zremrangebyscore(key, "-inf", cutoff)
    pipe.zadd(key, {member: txn.ts_ms})
    pipe.expire(key, settings.velocity_sum_window_sec * 2)
    pipe.zrange(key, 0, -1)
    members = pipe.execute()[-1]
    total = 0.0
    for m in members:
        try:
            total += float(m.rsplit("|", 1)[1])
        except (ValueError, IndexError):
            continue
    return round(total, 2)


def update_profile_and_speed(redis: Redis, txn: Transaction) -> tuple[float | None, float | None]:
    key = PROFILE_KEY.format(card=txn.card_id)
    speed: float | None = None
    distance: float | None = None
    if prev := redis.hgetall(key):
        try:
            prev_lat, prev_lon = float(prev["lat"]), float(prev["lon"])
            prev_ts = int(prev["ts_ms"])
            distance = haversine_km(prev_lat, prev_lon, txn.lat, txn.lon)
            speed = implied_speed_kmh(prev_lat, prev_lon, prev_ts, txn.lat, txn.lon, txn.ts_ms)
        except (KeyError, ValueError):
            speed = None
            distance = None
    redis.hset(
        key,
        mapping={"lat": txn.lat, "lon": txn.lon, "ts_ms": txn.ts_ms, "city": txn.city or ""},
    )
    redis.expire(key, 86400)
    return speed, distance


def blacklist_hits(redis: Redis, txn: Transaction) -> list[str]:
    hits: list[str] = []
    if redis.sismember(BLACKLIST_CARDS_KEY, txn.card_id):
        hits.append(f"card:{txn.card_id}")
    if redis.sismember(BLACKLIST_MERCHANTS_KEY, txn.merchant_id):
        hits.append(f"merchant:{txn.merchant_id}")
    return hits


def seed_blacklists(redis: Redis) -> None:
    if settings.blacklist_cards:
        redis.sadd(BLACKLIST_CARDS_KEY, *settings.blacklist_cards)
    if settings.blacklist_merchants:
        redis.sadd(BLACKLIST_MERCHANTS_KEY, *settings.blacklist_merchants)


def extract_features(redis: Redis, txn: Transaction) -> dict:
    hits = blacklist_hits(redis, txn)
    speed, distance = update_profile_and_speed(redis, txn)
    amount_z, samples = update_amount_stats(redis, txn)
    return {
        "amount": txn.amount,
        "window_count": record_and_count(redis, txn),
        "window_sum": record_and_sum(redis, txn),
        "implied_speed_kmh": speed,
        "geo_distance_km": distance,
        "user_window_count": record_user_count(redis, txn),
        "amount_z": amount_z,
        "amount_samples": samples,
        "new_merchant": is_new_merchant(redis, txn),
        "blacklisted": bool(hits),
        "blacklist_hits": hits,
    }


def mark_processed(redis: Redis, txn_id: str, ttl_sec: int = 6 * 3600) -> bool:
    return bool(redis.set(f"dedup:{txn_id}", "1", nx=True, ex=ttl_sec))


def welford_update(n: int, mean: float, m2: float, x: float) -> tuple[int, float, float]:
    n += 1
    delta = x - mean
    mean += delta / n
    m2 += delta * (x - mean)
    return n, mean, m2


def amount_zscore(n: int, mean: float, m2: float, x: float) -> float | None:
    if n < 2 or m2 <= 0:
        return None
    return (x - mean) / math.sqrt(m2 / n)


def record_user_count(redis: Redis, txn: Transaction) -> int:
    key = USER_COUNT_KEY.format(user=txn.user_id)
    cutoff = txn.ts_ms - settings.user_velocity_window_sec * 1000
    pipe = redis.pipeline(transaction=False)
    pipe.zremrangebyscore(key, "-inf", cutoff)
    pipe.zadd(key, {txn.txn_id: txn.ts_ms})
    pipe.expire(key, settings.user_velocity_window_sec * 2)
    pipe.zcard(key)
    return int(pipe.execute()[-1])


def update_amount_stats(redis: Redis, txn: Transaction) -> tuple[float | None, int]:
    key = PROFILE_KEY.format(card=txn.card_id)
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


def is_new_merchant(redis: Redis, txn: Transaction) -> bool:
    key = SEEN_MERCHANTS_KEY.format(card=txn.card_id)
    added = redis.sadd(key, txn.merchant_id)
    redis.expire(key, 30 * 86400)
    return bool(added)


def read_window_count(redis: Redis, card_id: str) -> int:
    cutoff = int(time.time() * 1000) - settings.velocity_count_window_sec * 1000
    return int(redis.zcount(COUNT_KEY.format(card=card_id), f"({cutoff}", "+inf"))


def read_window_sum(redis: Redis, card_id: str) -> float:
    cutoff = int(time.time() * 1000) - settings.velocity_sum_window_sec * 1000
    members = redis.zrangebyscore(SUM_KEY.format(card=card_id), f"({cutoff}", "+inf")
    total = 0.0
    for m in members:
        try:
            total += float(m.rsplit("|", 1)[1])
        except (ValueError, IndexError):
            continue
    return round(total, 2)


def read_user_count(redis: Redis, user_id: str) -> int:
    cutoff = int(time.time() * 1000) - settings.user_velocity_window_sec * 1000
    return int(
        redis.zcount(USER_COUNT_KEY.format(user=user_id), f"({cutoff}", "+inf")
    )


def knows_merchant(redis: Redis, card_id: str, merchant_id: str) -> bool:
    return bool(
        redis.sismember(SEEN_MERCHANTS_KEY.format(card=card_id), merchant_id)
    )
