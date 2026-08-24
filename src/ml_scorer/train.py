from __future__ import annotations

import os
from datetime import datetime, timezone

import joblib
import numpy as np

from common.db import get_pool
from ml_scorer.model import FEATURE_COLUMNS, build_row, to_vector

MIN_ROWS = 200
TEST_FRACTION = 0.2


TRAINING_COLUMNS = [
    "txn_id", "ts_ms", "amount", "window_count", "window_sum",
    "user_window_count", "new_merchant", "blacklisted",
    "implied_speed_kmh", "geo_distance_km", "amount_z",
    "amount_samples", "risk_score", "flagged",
]


def fetch_rows() -> list[dict]:
    cols = f"{', '.join(TRAINING_COLUMNS)}"
    with get_pool().connection() as conn:
        rows = conn.execute(
            f"SELECT {cols} FROM training_examples ORDER BY ts_ms LIMIT 2000000"
        ).fetchall()
    keys = cols.split(", ")
    return [dict(zip(keys, r)) for r in rows]


def _matrix(rows: list[dict]) -> np.ndarray:
    return np.array([to_vector(build_row(r)) for r in rows], dtype=float)


def _candidate_models():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    return {
        "logreg": make_pipeline(
            SimpleImputer(strategy="median"),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        ),
        "hgb": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.08,
            max_leaf_nodes=31,
            early_stopping=False,
            random_state=42,
        ),
    }


def _evaluate(clf, X_test, y_test) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    proba = clf.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    fraud = y_test == 1
    flagged = pred == 1
    top2pct = max(1, int(len(proba) * 0.02))
    top_idx = np.argsort(proba)[::-1][:top2pct]
    metrics = {
        "precision": round(float((pred & fraud).sum() / max(flagged.sum(), 1)), 4),
        "recall": round(float(fraud[flagged].sum() / max(fraud.sum(), 1)), 4),
        "precision_at_2pct": round(float(fraud[top_idx].mean()), 4),
        "test_rows": len(y_test),
    }
    if len(np.unique(y_test)) == 2:
        metrics["roc_auc"] = round(float(roc_auc_score(y_test, proba)), 4)
        metrics["pr_auc"] = round(float(average_precision_score(y_test, proba)), 4)
    else:
        metrics["roc_auc"] = metrics["pr_auc"] = None
    return metrics


def train_model(rows: list[dict]) -> dict:
    rows = sorted(rows, key=lambda r: r["ts_ms"])
    split = int(len(rows) * (1 - TEST_FRACTION))
    train_rows, test_rows = rows[:split], rows[split:]

    X_train, y_train = _matrix(train_rows), np.array([1 if r["flagged"] else 0 for r in train_rows])
    X_test, y_test = _matrix(test_rows), np.array([1 if r["flagged"] else 0 for r in test_rows])

    if len(np.unique(y_train)) < 2:
        raise RuntimeError(
            "training window contains a single class — collect more data before training"
        )

    results = {}
    for name, clf in _candidate_models().items():
        clf.fit(X_train, y_train)
        results[name] = (_evaluate(clf, X_test, y_test), clf)
        print(f"[{name}] {_fmt(results[name][0])}")

    def _pr(entry):
        return entry[0]["pr_auc"] if entry[0]["pr_auc"] is not None else -1.0

    winner = max(results, key=lambda n: _pr(results[n]))
    metrics, best_clf = results[winner]
    fraud_rate = sum(1 for r in rows if r["flagged"]) / len(rows)
    version = f"{winner}-{datetime.now(tz=timezone.utc):%Y%m%d}-{len(rows)}"
    return {
        "version": version,
        "features": FEATURE_COLUMNS,
        "model": best_clf,
        "metrics": {**metrics, "rows": len(rows), "fraud_rate": round(fraud_rate, 4)},
    }


def _fmt(m: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in m.items())


def main(out_dir: str | None = None) -> str | None:
    out_dir = out_dir or os.environ.get("MODELS_OUT", "/app/models")
    rows = fetch_rows()
    if len(rows) < MIN_ROWS:
        print(f"only {len(rows)} training rows collected (need {MIN_ROWS}) — keep the pipeline running")
        return None
    bundle = train_model(rows)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "model.pkl")
    joblib.dump(bundle, path)
    print(f"saved {bundle['version']} -> {path}")
    print(f"  metrics: {_fmt(bundle['metrics'])}")
    return path


if __name__ == "__main__":
    main()
