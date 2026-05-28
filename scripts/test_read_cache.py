"""One-off: time get_market_data (NO download) for the FULL universe's today
1m from local cache. If fast (<30s), the 14:30 path replaces the 6min
download_history_data2 with a plain cache read. Run on ECS."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from xtquant import xtdata
from mp.data.fetcher import get_recommendation_universe

codes = [str(c).zfill(6) for c in get_recommendation_universe()]
codes = [c for c in codes if not c.startswith(("300", "301", "302", "688", "689"))]
xt = [c + (".SH" if c.startswith("6") else ".SZ") for c in codes]
print(f"universe: {len(xt)} codes")

t0 = time.time()
mkt = xtdata.get_market_data(
    field_list=["open", "high", "low", "close", "volume"], stock_list=xt, period="1m",
    start_time="20260528093000", end_time="20260528143000",
    dividend_type="none", fill_data=False,
)
dt = time.time() - t0
close = mkt.get("close")
n_codes = close.shape[0] if close is not None else 0
n_bars = close.shape[1] if close is not None else 0
print(f"get_market_data (NO download) {len(xt)} codes took {dt:.1f}s | shape={n_codes}x{n_bars}")
