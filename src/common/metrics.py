from __future__ import annotations

from prometheus_client import Counter, Histogram, start_http_server

TXNS_PROCESSED = Counter(
    "fraudstream_txns_processed_total", "transactions evaluated by detector"
)
ALERTS_RAISED = Counter("fraudstream_alerts_raised_total", "alerts emitted")
RULE_HITS = Counter(
    "fraudstream_rule_hits_total", "rule triggers", ["rule"]
)
PROCESS_LATENCY = Histogram(
    "fraudstream_txn_process_seconds", "per-transaction processing latency"
)
SCORES_RECEIVED = Counter(
    "fraudstream_scores_received_total", "ml scores consumed from scores topic"
)


def start_metrics_server(port: int) -> None:
    start_http_server(port)
