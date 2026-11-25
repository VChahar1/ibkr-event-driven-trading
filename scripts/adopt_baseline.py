#!/usr/bin/env python3
"Adopt the broker's current positions as the reconciliation baseline."

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eventbot.bus import TopicMap, next_message, publish
from eventbot.bus.kafka import KafkaBus
from eventbot.config import load_config, load_settings
from eventbot.ledger import FillLedger
from eventbot.reconciler import build_baseline
from eventbot.schemas import Fill, PositionSnapshot, utc_now


async def run(assume_yes: bool) -> int:
    cfg, settings = load_config(), load_settings()
    topics = TopicMap.from_config(cfg.topics)
    bus = KafkaBus(settings.kafka_bootstrap, "eventbot-baseline")
    await bus.start()
    consumer = bus.consumer([topics.fills, topics.positions_snapshot], None)
    await consumer.start()
    ledger = FillLedger()
    latest: PositionSnapshot | None = None
    try:
        while (
            delivery := await next_message(consumer, bus, topics, service="baseline", timeout_s=3.0)
        ) is not None:
            message = delivery.message
            if isinstance(message, Fill):
                ledger.apply(message)
            elif isinstance(message, PositionSnapshot) and message.kind != "baseline":
                latest = message
        if latest is None:
            print("No broker snapshot found. Is the adapter running?", file=sys.stderr)
            return 1
        age_s = (utc_now() - latest.ts_utc).total_seconds()
        if age_s > cfg.reconciler.max_snapshot_age_s:
            print(f"The latest snapshot is {age_s:.0f} s old. Start the adapter first.", file=sys.stderr)
            return 1
        if latest.open_order_refs:
            print(
                f"Refusing: {len(latest.open_order_refs)} orders are open, so positions are moving.",
                file=sys.stderr,
            )
            return 1

        print(f"Account {latest.account}, snapshot {age_s:.0f} s old")
        print("Broker positions to adopt:")
        for symbol, qty in sorted(latest.positions.items()):
            print(f"  {symbol:<10} {qty}")
        if not latest.positions:
            print("  (none)")
        print(f"Positions from fills, for comparison: {ledger.positions or '(none)'}")
        if (
            not assume_yes
            and input("Adopt these as the baseline? Type 'adopt' to confirm: ").strip() != "adopt"
        ):
            print("Not adopted.")
            return 1
        baseline = build_baseline(latest, ledger.exec_ids)
        await publish(bus, topics.positions_snapshot, latest.account, baseline)
        print("Baseline published. Restart the reconciler, and clear the kill switch if it was set.")
        return 0
    finally:
        await consumer.stop()
        await bus.stop()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    return asyncio.run(run(parser.parse_args().yes))


if __name__ == "__main__":
    sys.exit(main())
