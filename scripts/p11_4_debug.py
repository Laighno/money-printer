from xtquant import xtdata
import inspect

# get_local_data returns dict[stock_code → DataFrame[time × fields]] not dict[field → ...]
# Try get_market_data instead which IS dict[field → DataFrame[time × codes]]
print("Available getter functions:")
for name in sorted(dir(xtdata)):
    if name.startswith('get_'):
        print(f"  {name}")

print()
print("get_market_data signature:")
print(f"  {inspect.signature(xtdata.get_market_data)}")
print(f"  doc: {(xtdata.get_market_data.__doc__ or '')[:600]}")

# Also re-download and try get_market_data
print("\nDownload + get_market_data:")
results = []
def cb(d): results.append(d)
xtdata.download_history_data2(
    stock_list=["000001.SZ"],
    period="1d",  # try daily first — quicker to verify connection works
    start_time="20240102",
    end_time="20240131",
    callback=cb,
)
print(f"  download cb: {results}")

# get_market_data for daily
print("\nget_market_data daily:")
mkt = xtdata.get_market_data(
    field_list=["open", "high", "low", "close", "volume"],
    stock_list=["000001.SZ"],
    period="1d",
    start_time="20240102",
    end_time="20240131",
    dividend_type="none",
    fill_data=False,
)
print(f"  type: {type(mkt).__name__}")
if isinstance(mkt, dict):
    for k, v in mkt.items():
        print(f"  {k}: shape={getattr(v, 'shape', 'N/A')}")
        if hasattr(v, 'head'):
            print(v.head(3))

# Now 1m
print("\nDownload 1m + get_market_data:")
results2 = []
def cb2(d): results2.append(d)
xtdata.download_history_data2(
    stock_list=["000001.SZ"],
    period="1m",
    start_time="20240102093000",
    end_time="20240102153000",  # just one day
    callback=cb2,
)
print(f"  download cb: {results2}")

mkt = xtdata.get_market_data(
    field_list=["open", "high", "low", "close", "volume"],
    stock_list=["000001.SZ"],
    period="1m",
    start_time="20240102093000",
    end_time="20240102153000",
    dividend_type="none",
    fill_data=False,
)
print(f"  market_data 1m type: {type(mkt).__name__}")
if isinstance(mkt, dict):
    for k, v in mkt.items():
        print(f"  {k}: shape={getattr(v, 'shape', 'N/A')}")
        if hasattr(v, 'head'):
            print(v.head(3))
