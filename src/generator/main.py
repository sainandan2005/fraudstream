from __future__ import annotations

import logging
import random
import time

from common.config import settings
from common.kafka_io import ensure_topics, make_producer, send_json
from common.models import Transaction

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("generator")

CITIES = [
    ("mumbai", 19.0760, 72.8777),
    ("delhi", 28.6139, 77.2090),
    ("bengaluru", 12.9716, 77.5946),
    ("hyderabad", 17.3850, 78.4867),
    ("chennai", 13.0827, 80.2707),
    ("kolkata", 22.5726, 88.3639),
    ("pune", 18.5204, 73.8567),
]

NUM_CARDS = 5000
NUM_MERCHANTS = 140


def jitter(rng: random.Random, lat: float, lon: float, deg: float = 0.02) -> tuple[float, float]:
    return round(lat + rng.uniform(-deg, deg), 6), round(lon + rng.uniform(-deg, deg), 6)


def build_world(rng: random.Random) -> tuple[list[dict], dict[str, list[dict]]]:
    cards = []
    for i in range(NUM_CARDS):
        city, lat, lon = rng.choice(CITIES)
        home_lat, home_lon = jitter(rng, lat, lon)
        cards.append(
            {
                "card_id": f"card_{i:04d}",
                "user_id": f"user_{i // 2:04d}",
                "city": city,
                "home": (home_lat, home_lon),
            }
        )
    merchants_by_city: dict[str, list[dict]] = {name: [] for name, _, _ in CITIES}
    for i in range(NUM_MERCHANTS):
        city, lat, lon = rng.choice(CITIES)
        m_lat, m_lon = jitter(rng, lat, lon)
        merchants_by_city[city].append(
            {"merchant_id": f"m_{i:04d}", "loc": (m_lat, m_lon), "city": city}
        )
    return cards, merchants_by_city


class Generator:
    def __init__(self) -> None:
        self.rng = random.Random(42)
        self.cards, self.merchants_by_city = build_world(self.rng)
        self.producer = make_producer()
        self.sent = 0

    def emit(
        self, card: dict, merchant_id: str, amount: float, loc: tuple[float, float], city: str
    ) -> None:
        txn = Transaction(
            card_id=card["card_id"],
            user_id=card["user_id"],
            merchant_id=merchant_id,
            amount=round(amount, 2),
            lat=loc[0],
            lon=loc[1],
            city=city,
        )
        send_json(self.producer, settings.transactions_topic, txn.card_id, txn.model_dump())
        self.sent += 1

    def local_merchant(self, card: dict) -> dict:
        return self.rng.choice(self.merchants_by_city[card["city"]])

    def normal_txn(self) -> None:
        card = self.rng.choice(self.cards)
        merchant = self.local_merchant(card)
        amount = min(max(self.rng.lognormvariate(6.5, 1.0), 10.0), 30000.0)
        self.emit(card, merchant["merchant_id"], amount, merchant["loc"], merchant["city"])

    def fraud_whale(self) -> None:
        card = self.rng.choice(self.cards)
        merchant = self.local_merchant(card)
        self.emit(
            card, merchant["merchant_id"], self.rng.uniform(120000, 350000),
            merchant["loc"], merchant["city"],
        )

    def fraud_burst(self) -> None:
        card = self.rng.choice(self.cards)
        merchant = self.local_merchant(card)
        for _ in range(6):
            self.emit(
                card, merchant["merchant_id"], self.rng.uniform(200, 5000),
                merchant["loc"], merchant["city"],
            )

    def fraud_travel(self) -> None:
        card = self.rng.choice(self.cards)
        merchant = self.local_merchant(card)
        far_city_name, far_lat, far_lon = self.rng.choice(
            [c for c in CITIES if c[0] != card["city"]]
        )
        near_loc = jitter(self.rng, *card["home"])
        far_loc = jitter(self.rng, far_lat, far_lon)
        self.emit(card, merchant["merchant_id"], self.rng.uniform(300, 3000), near_loc, card["city"])
        self.emit(card, merchant["merchant_id"], self.rng.uniform(300, 3000), far_loc, far_city_name)

    def fraud_blacklisted_merchant(self) -> None:
        card = self.rng.choice(self.cards)
        merchant = self.local_merchant(card)
        bad_merchant = self.rng.choice(settings.blacklist_merchants)
        self.emit(
            card, bad_merchant, self.rng.uniform(1000, 18000),
            merchant["loc"], merchant["city"],
        )

    def tick(self) -> int:
        if self.rng.random() >= settings.fraud_ratio:
            self.normal_txn()
            return 1
        pattern = self.rng.random()
        if pattern < 0.25:
            self.fraud_whale()
        elif pattern < 0.50:
            self.fraud_burst()
            return 6
        elif pattern < 0.75:
            self.fraud_travel()
            return 2
        else:
            self.fraud_blacklisted_merchant()
        return 1


def main() -> None:
    gen = Generator()
    ensure_topics([settings.transactions_topic])
    log.info(
        "generating ~%.0f txns/sec on topic %s (%d cards, %d merchants)",
        settings.txn_rate_per_sec,
        settings.transactions_topic,
        NUM_CARDS,
        NUM_MERCHANTS,
    )
    try:
        while True:
            emitted = gen.tick()
            time.sleep(emitted / settings.txn_rate_per_sec)
            if gen.sent % 200 < emitted:
                log.info("produced %d transactions", gen.sent)
    except KeyboardInterrupt:
        pass
    finally:
        gen.producer.flush(10)


if __name__ == "__main__":
    main()
