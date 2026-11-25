#!/usr/bin/env bash
set -u
BOT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$BOT_DIR/logs"; mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/daily_$(date +%Y-%m-%d).log"
exec >> "$LOG" 2>&1
echo "===== $(date '+%F %T') run_daily start ====="
cd "$BOT_DIR" || exit 1

source "$BOT_DIR/.venv/bin/activate"
if [ -f "$BOT_DIR/DRYRUN" ]; then
    python bot.py
else
    python bot.py --execute
fi
status=$?
echo "===== $(date '+%F %T') bot exit code $status ====="
if [ $status -ne 0 ] && [ $status -ne 2 ] && [ $status -ne 3 ]; then
    tail -n 60 "$LOG" | { "$BOT_DIR/.venv/bin/python" alerts.py "run_daily: bot failed with exit $status" \
        || python3 alerts.py "run_daily: bot failed with exit $status"; } || true
fi
if [ $status -ne 0 ] && command -v notify-send >/dev/null; then
    DISPLAY=:0 notify-send "IBKR bot" "Run ended with code $status - see $LOG"
fi
exit $status
