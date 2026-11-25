"""Entry point: python -m eventbot.adapter.main (make run-adapter). Exit 5 if the paper guard refuses."""

import asyncio
import logging
import sys

from eventbot.adapter.ib_client import IbAsyncClient
from eventbot.adapter.service import GatewayAdapter
from eventbot.config import ROOT
from eventbot.logs import log_event
from eventbot.paper_guard import PaperGuardError
from eventbot.runtime import EXIT_ERROR, EXIT_NOT_PAPER, EXIT_OK, build_runtime, install_stop_handlers
from stockbot import symbol_maps
from universe import load_universe

LOG = logging.getLogger("eventbot.adapter.main")


async def amain() -> int:
    """Build and run the paper Gateway adapter with configured bus and state."""
    rt = build_runtime("adapter")
    uni = load_universe(ROOT / rt.trading["stocks"]["universe_file"])
    to_ib, from_ib = symbol_maps(uni)
    primary = dict(zip(uni["symbol"], uni["primary_exchange"], strict=True))
    broker = IbAsyncClient(rt.settings, rt.cfg.adapter, to_ib, from_ib, primary)
    adapter = GatewayAdapter(
        rt.cfg,
        rt.settings,
        rt.bus,
        rt.state,
        broker,
        rt.topics,
        fill_timeout_s=float(rt.trading["risk"]["fill_timeout_s"]),
    )
    install_stop_handlers(adapter.request_stop)
    try:
        await rt.bus.start()
        await adapter.run()
        return EXIT_OK
    except PaperGuardError as exc:
        log_event(LOG, "paper_guard_refused", level=logging.CRITICAL, msg=str(exc))
        return EXIT_NOT_PAPER
    except Exception:
        LOG.exception("Adapter stopped on an error")
        return EXIT_ERROR
    finally:
        await rt.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(amain()))
