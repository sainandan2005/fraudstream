from __future__ import annotations

import logging

from prometheus_client import Counter
from redis import Redis

from common.config import settings
from common.kafka_io import ensure_topics, make_consumer, make_producer, send_json
from common.metrics import start_metrics_server
from common.models import ScoreRecord, Transaction
from detector.features import blacklist_hits
from ml_scorer.features import observe_and_extract
from ml_scorer.model import load_model

SCORED = Counter("fraudstream_scored_total", "transactions scored by ml-scorer")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("ml-scorer")


def main() -> None:
    start_metrics_server(9101)
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    ensure_topics([settings.transactions_topic, settings.scores_topic])
    model = load_model()
    producer = make_producer()
    consumer = make_consumer("ml-scorer", [settings.transactions_topic])
    log.info(
        "ml-scorer online: model=%s topic=%s -> %s",
        model.version,
        settings.transactions_topic,
        settings.scores_topic,
    )
    scored = 0
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                log.error("consumer error: %s", msg.error())
                continue
            try:
                txn = Transaction.model_validate_json(msg.value())
                x = observe_and_extract(redis, txn, blacklist_hits(redis, txn))
                score = model.predict(x)
                send_json(
                    producer,
                    settings.scores_topic,
                    txn.card_id,
                    ScoreRecord(
                        txn_id=txn.txn_id,
                        card_id=txn.card_id,
                        ml_score=score,
                        model_version=model.version,
                    ).model_dump(),
                )
                scored += 1
                SCORED.inc()
                if scored % 500 == 0:
                    log.info("scored %d transactions (last=%.1f)", scored, score)
            except Exception:
                log.exception("failed scoring message offset=%d", msg.offset())
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()
        producer.flush(10)


if __name__ == "__main__":
    main()
