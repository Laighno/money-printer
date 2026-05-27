from xtquant import xtdata
import time

print("xtdata module loaded")

# Test 1: download single
code = "000001.SZ"
print(f"Calling download_history_data({code!r}, '1m', '20240102093000', '20240105153000')...")
t0 = time.time()
ret = xtdata.download_history_data(code, "1m", "20240102093000", "20240105153000")
print(f"  returned: {ret!r}, took {time.time()-t0:.2f}s")

# Test 2: read back
print(f"Calling get_local_data...")
t0 = time.time()
raw = xtdata.get_local_data(
    field_list=["open", "high", "low", "close", "volume"],
    stock_list=[code],
    period="1m",
    start_time="20240102093000",
    end_time="20240105153000",
    count=-1,
    dividend_type="none",
    fill_data=False,
)
print(f"  returned type: {type(raw).__name__}, took {time.time()-t0:.2f}s")
if isinstance(raw, dict):
    for k, v in raw.items():
        print(f"  {k}: type={type(v).__name__}, shape={getattr(v, 'shape', 'N/A')}")
        if hasattr(v, 'head'):
            print(v.head(3))
            print(f"  index: {v.index[:5].tolist() if len(v.index) >= 5 else v.index.tolist()}")
