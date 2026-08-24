from __future__ import annotations

from dataclasses import dataclass

from common.config import settings

WEIGHT_AMOUNT = 75
WEIGHT_VELOCITY_COUNT = 75
WEIGHT_VELOCITY_SUM = 40
WEIGHT_GEO = 75
WEIGHT_BLACKLIST = 100
WEIGHT_USER_VELOCITY = 40
WEIGHT_AMOUNT_ANOMALY = 55
WEIGHT_NEW_MERCHANT = 15

RULE_AMOUNT = "amount_threshold"
RULE_VELOCITY_COUNT = "velocity_count"
RULE_VELOCITY_SUM = "velocity_sum"
RULE_GEO_SPEED = "geo_velocity"
RULE_BLACKLIST = "blacklist"
RULE_USER_VELOCITY = "user_velocity"
RULE_AMOUNT_ANOMALY = "amount_anomaly"
RULE_NEW_MERCHANT = "new_merchant"


@dataclass(frozen=True)
class RuleResult:
    rule: str
    weight: int
    detail: str


def check_amount(amount: float, limit: float) -> RuleResult | None:
    if amount > limit:
        return RuleResult(RULE_AMOUNT, WEIGHT_AMOUNT, f"amount ${amount:.2f} > ${limit:.2f}")
    return None


def check_velocity_count(count: int, max_count: int, window_sec: int) -> RuleResult | None:
    if count > max_count:
        return RuleResult(
            RULE_VELOCITY_COUNT,
            WEIGHT_VELOCITY_COUNT,
            f"{count} txns in {window_sec}s (max {max_count})",
        )
    return None


def check_velocity_sum(total: float, cap: float, window_sec: int) -> RuleResult | None:
    if total > cap:
        return RuleResult(
            RULE_VELOCITY_SUM,
            WEIGHT_VELOCITY_SUM,
            f"${total:.2f} spent in {window_sec}s (cap ${cap:.2f})",
        )
    return None


MIN_GEO_DISTANCE_KM = 50.0


def check_geo_speed(
    speed_kmh: float | None, limit_kmh: float, distance_km: float | None = None
) -> RuleResult | None:
    if speed_kmh is None or distance_km is None:
        return None
    if distance_km >= MIN_GEO_DISTANCE_KM and speed_kmh > limit_kmh:
        return RuleResult(
            RULE_GEO_SPEED,
            WEIGHT_GEO,
            f"{speed_kmh:,.0f} km/h over {distance_km:,.0f} km (limit {limit_kmh:.0f})",
        )
    return None


def check_blacklisted(blacklisted: bool, hits: list[str] | None = None) -> RuleResult | None:
    if blacklisted:
        detail = f"deny-list hit: {', '.join(hits)}" if hits else "card/merchant on deny-list"
        return RuleResult(RULE_BLACKLIST, WEIGHT_BLACKLIST, detail)
    return None


def check_user_velocity(count: int, max_count: int, window_sec: int) -> RuleResult | None:
    if count > max_count:
        return RuleResult(
            RULE_USER_VELOCITY,
            WEIGHT_USER_VELOCITY,
            f"{count} txns across user's cards in {window_sec}s (max {max_count})",
        )
    return None


def check_amount_anomaly(
    z: float | None, samples: int, min_samples: int, sigma_mult: float
) -> RuleResult | None:
    if z is None or samples < min_samples:
        return None
    if z > sigma_mult:
        return RuleResult(
            RULE_AMOUNT_ANOMALY,
            WEIGHT_AMOUNT_ANOMALY,
            f"amount {z:.1f}σ above personal mean ({samples} txns seen)",
        )
    return None


def check_new_merchant(new_merchant: bool, weight: int = WEIGHT_NEW_MERCHANT) -> RuleResult | None:
    if new_merchant:
        return RuleResult(RULE_NEW_MERCHANT, weight, "first-ever merchant for this card")
    return None


def evaluate(
    *,
    amount: float,
    window_count: int,
    window_sum: float,
    implied_speed_kmh: float | None,
    geo_distance_km: float | None = None,
    user_window_count: int = 0,
    amount_z: float | None = None,
    amount_samples: int = 0,
    new_merchant: bool = False,
    blacklisted: bool,
    blacklist_hits: list[str] | None = None,
) -> list[RuleResult]:
    results = [
        check_amount(amount, settings.amount_limit),
        check_velocity_count(window_count, settings.velocity_count_max, settings.velocity_count_window_sec),
        check_velocity_sum(window_sum, settings.velocity_sum_cap, settings.velocity_sum_window_sec),
        check_geo_speed(implied_speed_kmh, settings.geo_speed_kmh_max, geo_distance_km),
        check_user_velocity(
            user_window_count, settings.user_velocity_max, settings.user_velocity_window_sec
        ),
        check_amount_anomaly(
            amount_z, amount_samples, settings.anomaly_min_samples, settings.anomaly_sigma_mult
        ),
        check_new_merchant(new_merchant),
        check_blacklisted(blacklisted, blacklist_hits),
    ]
    return [r for r in results if r is not None]


def risk_score(hits: list[RuleResult]) -> int:
    return min(sum(h.weight for h in hits), 100)
