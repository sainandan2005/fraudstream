from __future__ import annotations

import pytest

from common.config import settings
from detector.features import (
    amount_zscore,
    haversine_km,
    implied_speed_kmh,
    welford_update,
)
from detector.rules import (
    RULE_AMOUNT,
    WEIGHT_AMOUNT,
    WEIGHT_BLACKLIST,
    WEIGHT_GEO,
    WEIGHT_VELOCITY_SUM,
    evaluate,
    risk_score,
)

MUMBAI = (19.0760, 72.8777)
DELHI = (28.6139, 77.2090)


def benign_features() -> dict:
    return {
        "amount": 850.0,
        "window_count": 2,
        "window_sum": 2400.0,
        "implied_speed_kmh": 30.0, "geo_distance_km": 5.0,
        "user_window_count": 3,
        "amount_z": None, "amount_samples": 25,
        "new_merchant": False,
        "blacklisted": False,
        "blacklist_hits": [],
    }


class TestGeometry:
    def test_haversine_mumbai_to_delhi(self):
        distance = haversine_km(*MUMBAI, *DELHI)
        assert 1100 < distance < 1250

    def test_haversine_zero(self):
        assert haversine_km(*MUMBAI, *MUMBAI) == pytest.approx(0.0, abs=1e-6)

    def test_haversine_bengaluru_to_delhi_under_50km_guard_is_irrelevant(self):
        assert haversine_km(12.9716, 77.5946, *DELHI) > 50

    def test_implied_speed_supersonic(self):
        speed = implied_speed_kmh(*MUMBAI, 1_000_000_000_000, *DELHI, 1_000_000_000_000 + 120_000)
        assert speed > settings.geo_speed_kmh_max

    def test_implied_speed_normal(self):
        speed = implied_speed_kmh(19.07, 72.87, 0, 19.17, 72.97, 3_600_000)
        assert speed < 100


class TestWelford:
    def test_welford_matches_direct_variance(self):
        xs = [700.0, 800.0, 650.0, 900.0, 750.0]
        n, mean, m2 = 0, 0.0, 0.0
        for x in xs:
            n, mean, m2 = welford_update(n, mean, m2, x)
        direct_mean = sum(xs) / len(xs)
        direct_m2 = sum((x - direct_mean) ** 2 for x in xs)
        assert mean == pytest.approx(direct_mean)
        assert m2 == pytest.approx(direct_m2)

    def test_zscore_of_outlier_is_high(self):
        xs = [700.0 + (i % 5) * 30 for i in range(30)]
        n, mean, m2 = 0, 0.0, 0.0
        for x in xs:
            n, mean, m2 = welford_update(n, mean, m2, x)
        z = amount_zscore(n, mean, m2, 40000.0)
        assert z is not None and z > settings.anomaly_sigma_mult

    def test_zscore_none_when_cold_start(self):
        assert amount_zscore(1, 500.0, 0.0, 90000.0) is None


class TestNewRules:
    def test_user_velocity_hit(self):
        features = {**benign_features(), "user_window_count": settings.user_velocity_max + 1}
        assert any(h.rule == "user_velocity" for h in evaluate(**features))

    def test_user_velocity_benign(self):
        hits = evaluate(**{**benign_features(), "user_window_count": settings.user_velocity_max})
        assert not any(h.rule == "user_velocity" for h in hits)

    def test_amount_anomaly_hit(self):
        features = {**benign_features(), "amount_z": 6.5, "amount_samples": 40}
        hits = [h for h in evaluate(**features) if h.rule == "amount_anomaly"]
        assert len(hits) == 1 and "σ" in hits[0].detail

    def test_amount_anomaly_needs_samples(self):
        features = {**benign_features(), "amount_z": 9.9, "amount_samples": 5}
        assert not any(h.rule == "amount_anomaly" for h in evaluate(**features))

    def test_anomaly_plus_new_merchant_reaches_threshold(self):
        features = {
            **benign_features(),
            "amount_z": 5.0,
            "new_merchant": True,
        }
        score = risk_score(evaluate(**features))
        assert score >= settings.alert_threshold

    def test_new_merchant_alone_below_threshold(self):
        hits = evaluate(**{**benign_features(), "new_merchant": True})
        assert [h.rule for h in hits] == ["new_merchant"]
        assert risk_score(hits) < settings.alert_threshold


class TestRules:
    def test_amount_hit(self):
        hits = evaluate(**{**benign_features(), "amount": 250000.0})
        rules = [h.rule for h in hits]
        assert RULE_AMOUNT in rules

    def test_amount_benign(self):
        assert evaluate(**benign_features()) == []

    def test_velocity_count_hit(self):
        hits = evaluate(**{**benign_features(), "window_count": settings.velocity_count_max + 1})
        assert any(h.rule == "velocity_count" for h in hits)

    def test_velocity_count_edge_not_triggered(self):
        hits = evaluate(**{**benign_features(), "window_count": settings.velocity_count_max})
        assert not any(h.rule == "velocity_count" for h in hits)

    def test_velocity_sum_hit(self):
        hits = evaluate(**{**benign_features(), "window_sum": 300000.0})
        assert any(h.rule == "velocity_sum" for h in hits)

    def test_geo_speed_hit(self):
        features = {**benign_features(), "implied_speed_kmh": 1200.0, "geo_distance_km": 1150.0}
        geo_hits = [h for h in evaluate(**features) if h.rule == "geo_velocity"]
        assert len(geo_hits) == 1
        assert geo_hits[0].weight == WEIGHT_GEO

    def test_near_jump_supersonic_not_flagged(self):
        features = {**benign_features(), "implied_speed_kmh": 50000.0, "geo_distance_km": 8.0}
        assert not any(h.rule == "geo_velocity" for h in evaluate(**features))

    def test_geo_speed_none_is_benign(self):
        assert not any(
            h.rule == "geo_velocity" for h in evaluate(**{**benign_features(), "implied_speed_kmh": None})
        )

    def test_blacklisted_instant_score(self):
        hits = evaluate(**{**benign_features(), "blacklisted": True, "blacklist_hits": ["card:x"]})
        assert risk_score(hits) >= settings.alert_threshold
        assert hits[-1].weight == WEIGHT_BLACKLIST

    def test_whale_alone_reaches_threshold(self):
        hits = evaluate(**{**benign_features(), "amount": 120000.0})
        assert [h.rule for h in hits] == [RULE_AMOUNT]
        assert risk_score(hits) >= settings.alert_threshold

    def test_combined_rules_reach_threshold(self):
        features = {
            **benign_features(),
            "amount": 150000.0,
            "window_count": 9,
            "window_sum": 600000.0,
        }
        hits = evaluate(**features)
        expected = min(
            WEIGHT_AMOUNT + 75 + WEIGHT_VELOCITY_SUM, 100
        )
        assert risk_score(hits) == expected
        assert risk_score(hits) >= settings.alert_threshold

    def test_single_moderate_rule_below_threshold(self):
        features = {**benign_features(), "window_sum": 300000.0}
        hits = evaluate(**features)
        assert risk_score(hits) == WEIGHT_VELOCITY_SUM
        assert risk_score(hits) < settings.alert_threshold

    def test_risk_score_capped_at_100(self):
        assert risk_score([type("R", (), {"weight": 60})()] * 3) == 100
