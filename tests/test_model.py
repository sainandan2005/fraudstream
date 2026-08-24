from __future__ import annotations

import math

import numpy as np
import pytest

from ml_scorer.model import (
    FEATURE_COLUMNS,
    HeuristicModel,
    SklearnModel,
    build_row,
    load_model,
    to_vector,
)


def raw(**overrides) -> dict:
    base = {
        "txn_id": "t1",
        "ts_ms": 1_787_564_400_000,  # some fixed instant
        "amount": 850.0,
        "window_count": 2,
        "window_sum": 2400.0,
        "user_window_count": 3,
        "new_merchant": False,
        "blacklisted": False,
        "implied_speed_kmh": 30.0,
        "geo_distance_km": 5.0,
        "amount_z": 0.4,
        "amount_samples": 25,
    }
    return {**base, **overrides}


class TestBuildRow:
    def test_contains_expected_columns(self):
        row = build_row(raw())
        assert list(row.keys()) == FEATURE_COLUMNS
        assert row["log_amount"] == pytest.approx(math.log1p(850.0))

    def test_cyclical_hour_encoding(self):
        noon = build_row({"ts_ms": (12 * 3_600_000)})
        midnight = build_row({"ts_ms": 0})
        assert noon["hour_cos"] == pytest.approx(-1.0, abs=1e-6)
        assert midnight["hour_cos"] == pytest.approx(1.0, abs=1e-6)
        assert midnight["hour_sin"] == pytest.approx(0.0, abs=1e-6)

    def test_missing_ts_gets_neutral_hour(self):
        row = build_row({k: v for k, v in raw().items() if k != "ts_ms"})
        assert row["hour_sin"] == 0.0 and row["hour_cos"] == 0.0

    def test_bool_coercion(self):
        row = build_row(raw(new_merchant=True, blacklisted=True))
        assert row["new_merchant"] == 1 and row["blacklisted"] == 1


class TestVector:
    def test_nan_passthrough_for_geo(self):
        vec = to_vector(
            raw(implied_speed_kmh=None, geo_distance_km=None, amount_z=None)
        )
        for idx, col in enumerate(FEATURE_COLUMNS):
            if col in ("implied_speed_kmh", "geo_distance_km", "amount_z"):
                assert math.isnan(vec[idx]), col
            else:
                assert not math.isnan(vec[idx])

    def test_column_count_matches_vector(self):
        assert len(to_vector(raw())) == len(FEATURE_COLUMNS)


class TestHeuristic:
    def test_benign_low(self):
        assert HeuristicModel().predict(raw()) < 30

    def test_burst_high(self):
        score = HeuristicModel().predict(raw(amount=800, window_count=11, window_sum=40000))
        assert score > 80

    def test_blacklist_raises(self):
        model = HeuristicModel()
        assert model.predict(raw(blacklisted=True)) > model.predict(raw()) + 50


def _synthetic_rows(n_benign: int = 150, n_fraud: int = 60) -> list[dict]:
    rng = np.random.default_rng(11)
    rows = []
    ts = 1_700_000_000_000
    slots = rng.permutation(n_benign + n_fraud)
    for k in range(n_benign):
        rows.append(
            raw(
                txn_id=f"b{k}",
                ts_ms=ts + int(slots[k]) * 1000,
                amount=float(rng.uniform(10, 30000)),
                window_count=int(rng.integers(1, 5)),
                window_sum=float(rng.uniform(50, 50000)),
                user_window_count=int(rng.integers(1, 12)),
                implied_speed_kmh=float(rng.uniform(0, 120)),
                geo_distance_km=float(rng.uniform(0, 15)),
                amount_z=float(rng.uniform(-2, 2)),
                flagged=False,
            )
        )
    for k in range(n_fraud):
        rows.append(
            raw(
                txn_id=f"f{k}",
                ts_ms=ts + int(slots[n_benign + k]) * 1000,
                amount=float(rng.uniform(120000, 350000)),
                window_count=int(rng.integers(6, 15)),
                window_sum=float(rng.uniform(250000, 600000)),
                user_window_count=int(rng.integers(13, 30)),
                new_merchant=True,
                blacklisted=bool(rng.integers(0, 2)),
                implied_speed_kmh=float(rng.uniform(900, 50000)),
                geo_distance_km=float(rng.uniform(50, 4000)),
                amount_z=float(rng.uniform(4, 12)),
                flagged=True,
            )
        )
    return rows


class TestTrainer:
    def test_train_model_returns_best_bundle_with_metrics(self):
        from ml_scorer.train import train_model

        bundle = train_model(_synthetic_rows())
        assert set(bundle) >= {"version", "features", "model", "metrics"}
        assert bundle["version"].startswith(("hgb-", "logreg-"))
        m = bundle["metrics"]
        assert m["roc_auc"] > 0.9 and m["pr_auc"] > 0.9
        assert 0 < m["precision_at_2pct"] <= 1

    def test_trained_model_serves_through_sklearn_wrapper(self):
        from sklearn.linear_model import LogisticRegression

        from ml_scorer.model import to_vector

        rows = _synthetic_rows()
        X = np.array([to_vector(build_row(r)) for r in rows])
        y = np.array([1 if r["amount"] > 100000 else 0 for r in rows])
        clf = LogisticRegression(max_iter=2000).fit(X, y)
        model = SklearnModel(
            {"version": "test", "features": FEATURE_COLUMNS, "model": clf}
        )
        score = model.predict(rows[0])
        assert 0 <= score <= 100


class TestLoadFallback:
    def test_fallback_when_no_artifact(self, monkeypatch):
        monkeypatch.setenv("MODEL_PATH", "/nonexistent/model.pkl")
        assert isinstance(load_model(), HeuristicModel)
