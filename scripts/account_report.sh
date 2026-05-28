#!/bin/bash
# Daily real-account QMT report (replaces paper_trade.sh).
# Run via launchd (com.moneyprinter.papertrade) at 18:00 Mon-Fri.

cd /Users/laighno/laighno/money-printer
source .venv/bin/activate

# Bypass Sparkle/mihomo proxy for akshare (ZZ500 fetch)
unset http_proxy https_proxy

echo "$(date): Starting account_report run"
mkdir -p data/logs
python scripts/account_report.py 2>&1 | tee -a data/logs/account_report.log
echo "$(date): account_report complete (exit=$?)"
