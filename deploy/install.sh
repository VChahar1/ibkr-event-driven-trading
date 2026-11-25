#!/usr/bin/env bash
set -euo pipefail
BOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
UNIT_DIR="$HOME/.config/systemd/user"
SRC="$BOT_DIR/deploy/systemd"
GW_VER=""

render() {
    sed -e "s|@BOT_DIR@|$BOT_DIR|g" -e "s|@GATEWAY_VERSION@|$GW_VER|g" "$SRC/$1" > "$UNIT_DIR/$1"
    echo "installed $UNIT_DIR/$1"
}

mkdir -p "$UNIT_DIR"
if [ "${1:-}" = "--gateway" ]; then
    command -v xvfb-run >/dev/null || { echo "xvfb-run missing: sudo apt install xvfb"; exit 1; }
    [ -x "$HOME/ibc/scripts/ibcstart.sh" ] || { echo "IBC not found: unpack it into ~/ibc"; exit 1; }
    [ -f "$HOME/ibc/config.ini" ] || { echo "~/ibc/config.ini missing (see deploy/README.md)"; exit 1; }
    [ "$(stat -c %a "$HOME/ibc/config.ini")" = "600" ] || { echo "run: chmod 600 ~/ibc/config.ini"; exit 1; }
    ini() { sed -n "s/^$1=//p" "$HOME/ibc/config.ini" | tail -1 | tr -d '[:space:]'; }
    bad=0
    [ "$(ini TradingMode)" = "paper" ] || { echo "config.ini: set TradingMode=paper"; bad=1; }
    [ "$(ini ExistingSessionDetectedAction)" = "primary" ] \
        || { echo "config.ini: set ExistingSessionDetectedAction=primary"; bad=1; }
    [ "$(ini AcceptIncomingConnectionAction)" = "reject" ] \
        || { echo "config.ini: set AcceptIncomingConnectionAction=reject (trusted 127.0.0.1 still connects)"; bad=1; }
    [ "$(ini ReadOnlyApi)" = "no" ] || { echo "config.ini: set ReadOnlyApi=no"; bad=1; }
    [ $bad -eq 0 ] || exit 1
    GW_VER="$(ls "$HOME/Jts/ibgateway" | grep -E '^[0-9]+$' | sort -n | tail -1)"
    [ -n "$GW_VER" ] || { echo "no Gateway install found in ~/Jts/ibgateway"; exit 1; }
    echo "IB Gateway version: $GW_VER"
    render ibgateway.service
fi
render ibkr-bot.service
render ibkr-bot.timer
systemctl --user daemon-reload
[ -n "$GW_VER" ] && systemctl --user enable --now ibgateway.service

if crontab -l 2>/dev/null | grep -v '^[[:space:]]*#' | grep -q run_daily.sh; then
    echo "NOT enabling ibkr-bot.timer: crontab still runs run_daily.sh."
    echo "Comment those lines out (crontab -e), then re-run this script."
    exit 1
fi
systemctl --user enable --now ibkr-bot.timer
systemctl --user list-timers ibkr-bot.timer --no-pager
