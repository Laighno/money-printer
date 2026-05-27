from mp.data.fetcher import get_recommendation_universe

codes = list(get_recommendation_universe())
xtc = [
    str(c).zfill(6) + (".SH" if str(c).zfill(6).startswith("6") else ".SZ")
    for c in codes
]
print("size:", len(xtc))
print("290-310:", xtc[290:310])
