#!/usr/bin/env python3
"""fx_convert.py - show cash balances / convert currency on the PAPER account through the same IB
Gateway session (no second login needed)."""

import argparse

from ib_async import IB, Forex, MarketOrder

from config import is_live_port, load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair")
    ap.add_argument("--side", choices=["BUY", "SELL"])
    ap.add_argument("--qty", type=float)
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()

    conf = load_config()[0]["ib"]
    if is_live_port(conf["port"]):
        raise SystemExit(f"Port {conf['port']} is a LIVE port - refusing to connect.")
    ib = IB()
    ib.connect(conf["host"], conf["port"], clientId=conf["client_id"] + 1, timeout=20)
    try:
        acct = conf["account"] or ib.managedAccounts()[0]
        if not acct.startswith("DU"):
            raise SystemExit(f"{acct} is not a paper account - refusing.")
        print(f"Account {acct} cash balances:")
        for v in ib.accountValues(acct):
            if "CashBalance" in v.tag and v.currency:
                print(f"  {v.tag:<22} {v.currency:<5} {float(v.value):>16,.2f}")
        if not a.pair:
            return
        c = Forex(a.pair)
        ib.qualifyContracts(c)
        print(f"\nPlan: {a.side} {a.qty:,.0f} {a.pair[:3]} ({a.pair[:3]}.{a.pair[3:]} on IDEALPRO)")
        if not a.execute:
            print("DRY RUN - add --execute to place it.")
            return
        t = ib.placeOrder(c, MarketOrder(a.side, a.qty, account=acct, orderRef="fx-setup"))
        for _ in range(60):
            if t.isDone():
                break
            ib.sleep(1)
        print(
            f"Status: {t.orderStatus.status}, filled {t.orderStatus.filled:,.0f} "
            f"@ {t.orderStatus.avgFillPrice}"
        )
        ib.sleep(2)
        for v in ib.accountValues(acct):
            if (
                v.tag.endswith("CashBalance")
                and "Fx" not in v.tag
                and "Total" not in v.tag
                and v.currency in (a.pair[:3], a.pair[3:])
            ):
                print(f"  now {v.currency}: {float(v.value):,.2f}")
    finally:
        ib.disconnect()


if __name__ == "__main__":
    main()
