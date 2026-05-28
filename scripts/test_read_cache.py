"""One-off: does xtdata.get_market_data return today's 1m from local cache
WITHOUT an explicit download_history_data2? If yes, the 14:30 path can just
read (fast) instead of downloading (6min hang). Run on ECS."""
import time
from xtquant import xtdata

codes = ["000001.SZ", "600000.SH", "002958.SZ"]

t0 = time.time()
mkt = xtdata.get_market_data(
    field_list=["close", "volume"], stock_list=codes, period="1m",
    start_time="20260528093000", end_time="20260528143000",
    dividend_type="none", fill_data=False,
)
dt = time.time() - t0
print(f"get_market_data (NO download) took {dt:.2f}s")
close = mkt.get("close")
if close is not None:
    print(f"close shape={close.shape}")
    if close.shape[1] > 0:
        print(f"times: {close.shape[1]} bars | first={close.columns[0]} last={close.columns[-1]}")
    else:
        print("EMPTY — cache does not have today's 1m without download")
