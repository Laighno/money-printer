"""One-shot Feishu delivery test for scheduled-task contexts (S4U debug)."""
import sys
from pathlib import Path
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))
from scripts.daily_report import send_to_feishu
ok = send_to_feishu("🔔 监控告警链路测试 ping — 收到请告知 (S4U scheduled-task context)")
print("send_to_feishu returned:", ok)
sys.exit(0 if ok else 7)
