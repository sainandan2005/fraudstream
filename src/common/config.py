from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    kafka_bootstrap_servers: str = "localhost:9092"
    redis_url: str = "redis://localhost:6379/0"
    database_url: str = "postgresql://fraud:fraud@localhost:5432/fraudstream"

    transactions_topic: str = "transactions.v1"
    alerts_topic: str = "fraud-alerts.v1"
    scores_topic: str = "fraud-scores.v1"
    topic_partitions: int = 6

    txn_rate_per_sec: float = 20.0
    fraud_ratio: float = 0.05

    alert_threshold: int = 70

    amount_limit: float = 100000.0
    velocity_count_max: int = 5
    velocity_count_window_sec: int = 300
    velocity_sum_cap: float = 250000.0
    velocity_sum_window_sec: int = 600
    geo_speed_kmh_max: float = 900.0

    user_velocity_max: int = 12
    user_velocity_window_sec: int = 600
    anomaly_min_samples: int = 20
    anomaly_sigma_mult: float = 4.0

    blacklist_cards: list[str] = ["card_dead_01", "card_dead_02"]
    blacklist_merchants: list[str] = ["m_bad_9001", "m_bad_9002"]

    api_host: str = "0.0.0.0"
    api_port: int = 8000


settings = Settings()
