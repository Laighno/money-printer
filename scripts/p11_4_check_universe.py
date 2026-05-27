from mp.data.fetcher import get_recommendation_universe

codes = list(get_recommendation_universe())
xtc = [
    str(c).zfill(6) + (".SH" if str(c).zfill(6).startswith("6") else ".SZ")
    for c in codes
]
print("size:", len(xtc))
print("chunk 300-400:")
for i in range(300, min(400, len(xtc)), 10):
    print(f"  {i}: {xtc[i:i+10]}")
