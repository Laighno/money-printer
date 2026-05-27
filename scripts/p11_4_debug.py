from xtquant import xtdata
import inspect

print("Available download-related functions in xtdata:")
for name in sorted(dir(xtdata)):
    if 'download' in name.lower():
        if not name.startswith('_'):
            print(f"  {name}")

print()
print("xtdata.download_history_data:")
try:
    sig = inspect.signature(xtdata.download_history_data)
    print(f"  sig: {sig}")
except (ValueError, TypeError) as e:
    print(f"  inspect failed: {e}")
print(f"  doc: {(xtdata.download_history_data.__doc__ or '(none)')[:500]}")

for fn_name in ['download_history_data2', 'download_history_data_callback']:
    if hasattr(xtdata, fn_name):
        fn = getattr(xtdata, fn_name)
        print(f"\n{fn_name}:")
        try:
            print(f"  sig: {inspect.signature(fn)}")
        except (ValueError, TypeError) as e:
            print(f"  sig fail: {e}")
        print(f"  doc: {(fn.__doc__ or '(none)')[:500]}")

# Try the actual download with callback to see what's happening
print()
print("Trying download_history_data2 with callback:")
results = []
def cb(data):
    results.append(data)
    print(f"  callback got: {data!r}")

if hasattr(xtdata, 'download_history_data2'):
    ret = xtdata.download_history_data2(
        stock_list=["000001.SZ"],
        period="1m",
        start_time="20240102",
        end_time="20240105",
        callback=cb,
    )
    print(f"  download_history_data2 returned: {ret!r}")
    print(f"  callback fired {len(results)} times")
