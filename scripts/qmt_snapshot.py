"""Print QMT account + positions as JSON on stdout (one line, marker-prefixed).

Designed to be invoked on ECS Windows via SSH from Mac. The Mac side
parses the marker line and ignores any xtquant connect banner output.
See scripts/sync_portfolio_from_qmt.py for the consumer side.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mp.execution.qmt_broker import QMTBroker

broker = QMTBroker(account_id="8886933837", qmt_userdata_path=r"C:\guojin\userdata_mini")
broker.connect()
try:
    info = broker.get_account_info()
    positions = broker.get_positions()
finally:
    broker.disconnect()

out = {
    "account": {
        "total_assets": float(info.total_assets),
        "cash_available": float(info.cash_available),
        "market_value": float(info.market_value),
        "updated_at": info.updated_at,
    },
    "positions": [
        {
            "code": p.code,
            "name": p.name or "",
            "shares": int(p.shares_total),
            "avg_cost": float(p.avg_cost),
            "market_price": float(getattr(p, "market_price", 0.0) or 0.0),
            "market_value": float(getattr(p, "market_value", 0.0) or 0.0),
        }
        for p in positions if int(p.shares_total) > 0
    ],
}
print("__QMT_JSON__ " + json.dumps(out, ensure_ascii=False))
