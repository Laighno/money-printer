from xtquant import xtdata
import time

# Check disk size of cache before/after download attempt
import os, subprocess

cache_path = r"C:\guojin\bin.x64\..\userdata_mini\datadir"
# walking that path may be heavy. Just check broad downloaded periods via get_local_data count

print("=== Check 1m availability per code: how many trading days in local cache? ===")
for code in ["000001.SZ", "600000.SH"]:
    # Just use count -1, no time filter
    local = xtdata.get_local_data(
        field_list=["close"],
        stock_list=[code],
        period="1m",
        start_time="",
        end_time="",
        count=-1,
        dividend_type="none",
        fill_data=False,
    )
    df = local.get(code)
    if df is not None and not df.empty:
        idx = df.index
        # Extract YYYYMMDD from each timestamp (string format YYYYMMDDHHMMSS)
        days = set(str(t)[:8] for t in idx)
        print(f"  {code}: {df.shape[0]} bars covering {len(days)} unique days, range {min(idx)} ~ {max(idx)}")
    else:
        print(f"  {code}: empty")

print("\n=== Try aggressive download for 2024-01 (full month) ===")
t0 = time.time()
results = []
def cb(d): results.append(d)
ret = xtdata.download_history_data2(
    stock_list=["000001.SZ"],
    period="1m",
    start_time="20240101000000",
    end_time="20240131235959",
    callback=cb,
)
print(f"  download returned {ret!r} in {time.time()-t0:.1f}s, callbacks: {len(results)}")
for r in results:
    print(f"  cb: {r}")

# Now check local cache
local = xtdata.get_local_data(
    field_list=["close"],
    stock_list=["000001.SZ"],
    period="1m",
    start_time="20240101000000",
    end_time="20240131235959",
    count=-1,
    dividend_type="none",
    fill_data=False,
)
df = local.get("000001.SZ")
print(f"  after download 2024-01: shape={df.shape if df is not None else None}")

# Try a date with permission - 2025 maybe?
for ym in ["202401", "202407", "202501", "202509", "202601", "202603"]:
    y = ym[:4]
    m = ym[4:]
    start = f"{y}{m}01000000"
    end = f"{y}{m}28235959"

    rs = []
    xtdata.download_history_data2(["000001.SZ"], "1m", start, end, callback=lambda d: rs.append(d))
    local = xtdata.get_local_data(
        field_list=["close"], stock_list=["000001.SZ"], period="1m",
        start_time=start, end_time=end, count=-1, dividend_type="none", fill_data=False,
    )
    df = local.get("000001.SZ")
    days = set(str(t)[:8] for t in df.index) if df is not None else set()
    print(f"  {ym}: rows={df.shape[0] if df is not None else 0}, days={len(days)}, dl_cb={rs[-1] if rs else None}")
