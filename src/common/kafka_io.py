from __future__ import annotations

import json
import logging
import time

from confluent_kafka import Consumer, Producer
from confluent_kafka.admin import AdminClient, NewTopic

from common.config import settings

log = logging.getLogger("kafka")


def ensure_topics(topics: list[str], retries: int = 30, delay_sec: float = 2.0) -> None:
    for attempt in range(1, retries + 1):
        try:
            admin = AdminClient({"bootstrap.servers": settings.kafka_bootstrap_servers})
            existing = set(admin.list_topics(timeout=10).topics)
            missing = [
                NewTopic(t, num_partitions=settings.topic_partitions, replication_factor=1)
                for t in topics
                if t not in existing
            ]
            if missing:
                for topic, fut in admin.create_topics(missing).items():
                    try:
                        fut.result(timeout=15)
                        log.info("created topic %s", topic)
                    except Exception as exc:  # noqa: BLE001 - log and continue on per-topic races
                        log.warning("topic %s: %s", topic, exc)
            return
        except Exception as exc:  # noqa: BLE001 - broker may not be up yet; retry any error
            log.warning("waiting for kafka (attempt %d/%d): %s", attempt, retries, exc)
            time.sleep(delay_sec)
    raise RuntimeError(f"kafka not reachable at {settings.kafka_bootstrap_servers}")


def make_producer() -> Producer:
    return Producer(
        {
            "bootstrap.servers": settings.kafka_bootstrap_servers,
            "linger.ms": 20,
            "compression.type": "lz4",
        }
    )


def make_consumer(group_id: str, topics: list[str], auto_commit: bool = True) -> Consumer:
    consumer = Consumer(
        {
            "bootstrap.servers": settings.kafka_bootstrap_servers,
            "group.id": group_id,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": auto_commit,
        }
    )
    consumer.subscribe(topics)
    return consumer


def _delivery_report(err, msg) -> None:
    if err is not None:
        log.error("delivery failed for %s: %s", msg.key(), err)


def send_json(producer: Producer, topic: str, key: str, payload: dict) -> None:
    producer.produce(
        topic,
        key=key.encode(),
        value=json.dumps(payload).encode(),
        on_delivery=_delivery_report,
    )
    producer.poll(0)
