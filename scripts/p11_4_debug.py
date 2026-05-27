from xtquant import xtdata
import time

# Now that download_history_data2 fired finished=1, check get_local_data
print("After download_history_data2, get_local_data for 000001.SZ 2024-01-02 ~ 2024-01-05:")
raw = xtdata.get_local_data(
    field_list=["open", "high", "low", "close", "volume"],
    stock_list=["000001.SZ"],
    period="1m",
    start_time="20240102000000",
    end_time="20240106000000",
    count=-1,
    dividend_type="none",
    fill_data=False,
)
print(f"  raw type: {type(raw).__name__}")
if isinstance(raw, dict):
    for k, v in raw.items():
        print(f"  field {k}: shape={getattr(v, 'shape', 'N/A')}")
        if hasattr(v, 'head'):
            print(v.head(5))
            print(f"  index[:5]: {v.index[:5].tolist() if len(v.index) >= 5 else v.index.tolist()}")

# Re-trigger download via download_history_data2 (batch)
print("\nRe-trigger download_history_data2 for 3 codes:")
results = []
def cb(data):
    results.append(data)
    print(f"  cb: {data}")

ret = xtdata.download_history_data2(
    stock_list=["000001.SZ", "000002.SZ", "600000.SH"],
    period="1m",
    start_time="20240102093000",
    end_time="20240131153000",
    callback=cb,
)
print(f"  returned: {ret!r}, callbacks: {len(results)}")

# Now read back all 3
print("\nget_local_data for all 3 codes:")
raw = xtdata.get_local_data(
    field_list=["open", "high", "low", "close", "volume"],
    stock_list=["000001.SZ", "000002.SZ", "600000.SH"],
    period="1m",
    start_time="20240102093000",
    end_time="20240131153000",
    count=-1,
    dividend_type="none",
    fill_data=False,
)
for k, v in raw.items():
    print(f"  field {k}: shape={getattr(v, 'shape', 'N/A')}")
