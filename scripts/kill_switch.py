#!/usr/bin/env python3
"""Set, clear or show the kill switch. Clearing it is a human decision: nothing does it automatically."""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import redis.asyncio as aioredis

from eventbot.config import load_settings
from eventbot.state import StateStore


async def run(action: str, reason: str) -> int:
    client = aioredis.Redis.from_url(
        load_settings().redis_url, decode_responses=True, socket_connect_timeout=2
    )
    state = StateStore(client)
    try:
        if action == "on":
            await state.set_kill_switch(reason)
        elif action == "off":
            await state.clear_kill_switch()
        value = await state.kill_switch()
        print(f"kill_switch: {'ON - ' + value if value is not None else 'off'}")
        return 0
    finally:
        await client.aclose()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["on", "off", "status"])
    parser.add_argument("--reason", default="set by hand")
    args = parser.parse_args()
    return asyncio.run(run(args.action, args.reason))


if __name__ == "__main__":
    sys.exit(main())
