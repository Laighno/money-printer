from xtquant import xtdata

# Test 1m with a RECENT date (yesterday)
print("=== Test 1m with recent date 2026-05-26 ===")
xtdata.download_history_data("000001.SZ", "1m", "20260526000000", "20260526235959")

mkt = xtdata.get_market_data(
    field_list=["open", "close", "volume"],
    stock_list=["000001.SZ"],
    period="1m",
    start_time="20260526000000",
    end_time="20260526235959",
    dividend_type="none",
    fill_data=False,
)
for k, v in mkt.items():
    print(f"  {k}: shape={v.shape}")
    if v.shape[1] > 0:
        print(v.iloc[:, :5])  # first 5 cols

# Try get_local_data signature different
print("\n=== get_local_data with recent date ===")
local = xtdata.get_local_data(
    field_list=["open", "close", "volume"],
    stock_list=["000001.SZ"],
    period="1m",
    start_time="20260526000000",
    end_time="20260526235959",
    count=-1,
    dividend_type="none",
    fill_data=False,
)
for k, v in local.items():
    print(f"  {k}: shape={v.shape}")
    if v.shape[0] > 0:
        print(v.head(5))

# Try without download first — assume cache exists
print("\n=== get_market_data for 2024-01 1m WITHOUT explicit download ===")
mkt = xtdata.get_market_data(
    field_list=["open", "close", "volume"],
    stock_list=["000001.SZ"],
    period="1m",
    start_time="",
    end_time="",
    count=-1,
    dividend_type="none",
    fill_data=False,
)
for k, v in mkt.items():
    print(f"  {k}: shape={v.shape}")
    if v.shape[1] > 0:
        print(f"  date range cols: {v.columns[0]} ~ {v.columns[-1]}")
