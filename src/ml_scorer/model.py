from __future__ import annotations

import math
import os

MODEL_DIR = os.environ.get("MODELS_DIR", "/app/models")
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(MODEL_DIR, "model.pkl"))


def build_row(x: dict) -> dict:
    """Derive the model feature row from raw per-transaction fields.

    Both the trainer (Postgres rows) and the scorer (live Redis reads) call
    this with the same keys, guaranteeing train/serve parity.
    """
    amount = float(x.get("amount") or 0)
    window_sum = float(x.get("window_sum") or 0)

    def _num(v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return math.nan
        return v

    ts = x.get("ts_ms")
    if ts is not None:
        hour = (int(ts) / 3_600_000.0) % 24
        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)
    else:
        hour_sin = hour_cos = 0.0

    return {
        "amount": amount,
        "log_amount": math.log1p(max(amount, 0)),
        "window_count": int(x.get("window_count") or 0),
        "log_window_sum": math.log1p(max(window_sum, 0)),
        "user_window_count": int(x.get("user_window_count") or 0),
        "new_merchant": int(bool(x.get("new_merchant"))),
        "blacklisted": int(bool(x.get("blacklisted"))),
        "implied_speed_kmh": _num(x.get("implied_speed_kmh")),
        "geo_distance_km": _num(x.get("geo_distance_km")),
        "amount_z": _num(x.get("amount_z")),
        "card_samples": int(x.get("amount_samples") or 0),
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
    }


FEATURE_COLUMNS = list(build_row({}).keys())


def to_vector(x: dict) -> list[float]:
    row = build_row(x)
    return [row[c] for c in FEATURE_COLUMNS]


class HeuristicModel:
    version = "heuristic-v0"

    INTERCEPT = -2.0
    W_BLACKLIST = 3.0
    W_COUNT_EXCESS = 0.8
    W_SUM_SPIKE = 2.2
    W_AMOUNT_HIGH = 2.4
    W_AMOUNT_EXTREME = 1.2

    def predict(self, x: dict) -> float:
        amount = float(x.get("amount") or 0)
        count = int(x.get("window_count") or 0)
        window_sum = float(x.get("window_sum") or 0)
        z = self.INTERCEPT
        if x.get("blacklisted"):
            z += self.W_BLACKLIST
        z += self.W_COUNT_EXCESS * max(0, count - 5)
        if window_sum > 200000:
            z += self.W_SUM_SPIKE
        if amount > 100000:
            z += self.W_AMOUNT_HIGH
        if amount > 250000:
            z += self.W_AMOUNT_EXTREME
        return round(100.0 / (1.0 + math.exp(-z)), 2)


class SklearnModel:
    def __init__(self, bundle: dict) -> None:
        self._clf = bundle["model"]
        self.version = bundle["version"]
        if bundle.get("features") != FEATURE_COLUMNS:
            raise ValueError(
                f"model feature mismatch: {bundle.get('features')} != {FEATURE_COLUMNS}"
            )

    def predict(self, x: dict) -> float:
        proba_fraud = self._clf.predict_proba([to_vector(x)])[0][1]
        return round(float(proba_fraud) * 100.0, 2)


def _model_path() -> str:
    return os.environ.get("MODEL_PATH") or os.path.join(MODEL_DIR, "model.pkl")


def load_model() -> HeuristicModel | SklearnModel:
    path = _model_path()
    if path and os.path.exists(path):
        try:
            import joblib

            bundle = joblib.load(path)
            model = SklearnModel(bundle)
            print(f"loaded trained model {model.version} from {path}")
            return model
        except Exception as exc:  # noqa: BLE001 - corrupted artifact must never take scoring down
            print(f"failed to load {path}: {exc} — falling back to heuristic")
    return HeuristicModel()
