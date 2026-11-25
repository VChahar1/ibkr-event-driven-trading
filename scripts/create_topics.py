#!/usr/bin/env python3
"""Create the Kafka topics from config/eventbot.yaml. Safe to run repeatedly; existing topics are
left as they are and reported. Exits non-zero if a topic could not be created."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aiokafka.admin import AIOKafkaAdminClient, NewTopic
from aiokafka.errors import TopicAlreadyExistsError, for_code

from eventbot.config import TopicConfig, load_config, load_settings


def topic_configs(t: TopicConfig) -> dict[str, str]:
    configs = {"retention.ms": str(t.retention_ms), "min.insync.replicas": "1"}
    if t.durable:
        configs["flush.messages"] = "1"
    return configs


async def create_topics(bootstrap: str, topics: list[TopicConfig]) -> int:
    admin = AIOKafkaAdminClient(bootstrap_servers=bootstrap, client_id="eventbot-create-topics")
    await admin.start()
    failures = 0
    try:
        for t in topics:
            new = NewTopic(
                name=t.name, num_partitions=t.partitions, replication_factor=1, topic_configs=topic_configs(t)
            )
            response = await admin.create_topics([new])
            for error in response.topic_errors:
                code = error[1]
                if code == 0:
                    print(
                        f"created  {t.name} (partitions={t.partitions}, retention_ms={t.retention_ms}, "
                        f"durable={t.durable})"
                    )
                elif for_code(code) is TopicAlreadyExistsError:
                    print(f"exists   {t.name}")
                else:
                    print(f"FAILED   {t.name}: {for_code(code).__name__}", file=sys.stderr)
                    failures += 1
    finally:
        await admin.close()
    return failures


def main() -> int:
    cfg, settings = load_config(), load_settings()
    return 1 if asyncio.run(create_topics(settings.kafka_bootstrap, cfg.topics.all())) else 0


if __name__ == "__main__":
    sys.exit(main())
