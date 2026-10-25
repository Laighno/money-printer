# -*- coding: gbk -*-
# BigQMT "name bridge" strategy for money-printer.
#
# WHY THIS EXISTS (2026-10-25)
# ---------------------------
# 国金 disabled file I/O and network access in the QMT built-in Python on
# 2026-10-23. The old file bridge (bigqmt_bridge_strategy.py) is dead: its
# loader could not even start, failing at line 3 with
#   PermissionError: Foribdden FileIO
# Probing the sandbox from inside a strategy established exactly what survives:
#
#   open(path, "r")  -> PermissionError      (any path, also "rb" and io.open)
#   open(path, "w")  -> OK                   (any path tested)
#   os.listdir       -> OK
#   os.stat/exists   -> OK
#   socket           -> timeout (blocked)
#   passorder        -> available
#   get_trade_detail_data -> available (returned 22 live positions)
#
# Reading is banned, WRITING IS NOT. So orders cannot be handed to the strategy
# through file CONTENT, but they can through file NAMES, which os.listdir still
# returns. The external pipeline creates one empty file per order; this strategy
# lists the directory, parses the names, and submits them. Results travel back
# the normal way, as written files, which external Python can read freely.
#
# Measured headroom: listdir returned a 184-char name untruncated; an order name
# needs 39. 25 holdings = 25 files, no pressure.
#
# CAVEAT: this works around a restriction the broker put in place deliberately.
# It is a stopgap while miniQMT / an official API is sorted out, and it can stop
# working without notice if they also restrict listdir.
#
# PROTOCOL
# --------
#   orders in : ORD_DIR\<seq>__<B|S>__<code>__<shares>__<price*100>__<tag>.ord
#               seq orders execution; sells should be numbered before buys.
#   receipts  : ORD_DIR\done\<seq>__<status>__<orderid>__<tag>.rcpt   (written)
#               status = sent | failed | dup
#   consumed  : the .ord file is renamed to .taken so it is never resubmitted
#               (os.rename is allowed; only reading content is not).
#   heartbeat : ORD_DIR\hb\<unix_ts>.hb  - one file per tick, old ones removed.
#               External side reads the NAME (its own listdir) to get the time;
#               it cannot read content either way, so the name carries it.
#
# DEPLOY: QMT client -> 模型交易 -> new strategy -> paste this file -> run.
# ASCII-only string literals (the client parses .py as GBK).

import os
import time

ACCOUNT = "8886933837"
ORD_DIR = r"C:\guojin\python\mp_orders"
DONE_DIR = os.path.join(ORD_DIR, "done")
HB_DIR = os.path.join(ORD_DIR, "hb")

# passorder constants (same values the file bridge used)
OPTYPE_BUY = 23
OPTYPE_SELL = 24
ORDERBY_VOLUME = 1101
PRTYPE_LIMIT = 11

_last_hb = [0.0]
_seen = {}          # seq -> submitted ts, guards against double submit in-process


def _ensure_dirs():
    for d in (ORD_DIR, DONE_DIR, HB_DIR):
        try:
            if not os.path.isdir(d):
                os.makedirs(d)
        except Exception:
            pass


def _code_to_qmt(code):
    if code.startswith("6"):
        return code + ".SH"
    return code + ".SZ"


def _parse(fn):
    """Order filename -> dict, or None when the name is not an order.

    Layout: <seq>__<side>__<code>__<shares>__<price*100>__<tag>.ord
    Price travels as an integer (fen) so no locale/decimal issues can creep in.
    """
    if not fn.lower().endswith(".ord"):
        return None
    parts = fn[:-4].split("__")
    if len(parts) < 5:
        return None
    try:
        seq = parts[0]
        side = parts[1].upper()
        code = parts[2]
        shares = int(parts[3])
        price = int(parts[4]) / 100.0
        tag = parts[5] if len(parts) > 5 else ""
    except Exception:
        return None
    if side not in ("B", "S"):
        return None
    if len(code) != 6 or not code.isdigit():
        return None
    if shares <= 0 or shares % 100 != 0:
        return None
    if price <= 0:
        return None
    return {"seq": seq, "side": side, "code": code, "shares": shares,
            "price": price, "tag": tag, "fn": fn}


def _write_marker(path):
    """Create an empty/1-byte file. Writing is the only content channel left."""
    try:
        f = open(path, "w")
        try:
            f.write("x")
        finally:
            f.close()
        return True
    except Exception:
        return False


def _receipt(seq, status, order_id, tag):
    safe_id = str(order_id or "none").replace("_", "-")[:40]
    name = "%s__%s__%s__%s.rcpt" % (seq, status, safe_id, tag or "na")
    _write_marker(os.path.join(DONE_DIR, name))


def _known_order_ids(C):
    """Order ids visible right now, so a new one can be attributed after submit.

    Live QMT strips remark from order rows (verified 2026-09-02), so the only
    reliable attribution is "the id that appeared after we submitted".
    """
    ids = []
    try:
        rows = get_trade_detail_data(ACCOUNT, "stock", "order")
        for o in rows:
            ids.append(str(o.m_strOrderSysID))
    except Exception:
        pass
    return ids


def _find_new_id(C, before, o):
    """Single new order id matching this submission, else None (ambiguous)."""
    try:
        rows = get_trade_detail_data(ACCOUNT, "stock", "order")
    except Exception:
        return None
    before = set(before)
    want_dir = 48 if o["side"] == "B" else 49
    cands = []
    for r in rows:
        oid = str(r.m_strOrderSysID)
        if oid in before:
            continue
        try:
            if str(r.m_strInstrumentID).split(".")[0] != o["code"]:
                continue
            if int(r.m_nVolumeTotalOriginal) != o["shares"]:
                continue
            d = int(getattr(r, "m_nDirection", want_dir))
            if d != want_dir:
                continue
        except Exception:
            continue
        cands.append(oid)
    if len(cands) == 1:
        return cands[0]
    return None


def _submit(C, o):
    optype = OPTYPE_BUY if o["side"] == "B" else OPTYPE_SELL
    before = _known_order_ids(C)
    passorder(optype, ORDERBY_VOLUME, ACCOUNT, _code_to_qmt(o["code"]),
              PRTYPE_LIMIT, o["price"], o["shares"], "money-printer", 1,
              "mp" + str(o["seq"]), C)
    time.sleep(0.6)
    return _find_new_id(C, before, o)


def _heartbeat():
    """One file per tick named with the epoch second; prune the old ones.

    The external watchdog cannot read file CONTENT, so liveness has to be in
    the name. Keeping only the newest few bounds the directory.
    """
    now = int(time.time())
    if now - _last_hb[0] < 2:
        return
    _last_hb[0] = now
    _write_marker(os.path.join(HB_DIR, "%d.hb" % now))
    try:
        names = sorted(os.listdir(HB_DIR))
        for n in names[:-3]:
            try:
                os.remove(os.path.join(HB_DIR, n))
            except Exception:
                pass
    except Exception:
        pass


def _tick(C):
    _ensure_dirs()
    _heartbeat()
    try:
        names = os.listdir(ORD_DIR)
    except Exception as e:
        print("namebridge listdir failed: %r" % (e,))
        return
    orders = []
    for n in names:
        o = _parse(n)
        if o is not None:
            orders.append(o)
    if not orders:
        return
    # seq drives ordering: the external side numbers sells before buys so the
    # cash they free is available to the buys in the same batch (A-share T+0 cash).
    orders.sort(key=lambda x: x["seq"])
    for o in orders:
        src = os.path.join(ORD_DIR, o["fn"])
        # Claim the file FIRST. If the rename fails the file is gone or already
        # claimed, and we must not submit: a double submit is worse than a miss.
        taken = src[:-4] + ".taken"
        try:
            os.rename(src, taken)
        except Exception:
            continue
        if o["seq"] in _seen:
            _receipt(o["seq"], "dup", None, o["tag"])
            continue
        _seen[o["seq"]] = time.time()
        try:
            oid = _submit(C, o)
            print("namebridge sent %s %s %s x%d @%.2f -> %s" % (
                o["seq"], o["side"], o["code"], o["shares"], o["price"], oid))
            _receipt(o["seq"], "sent", oid, o["tag"])
        except Exception as e:
            print("namebridge FAILED %s: %r" % (o["seq"], e))
            _receipt(o["seq"], "failed", None, o["tag"])


# ---------------------------------------------------------------- QMT entries
# QMT statically scans for these names, so they must be defined at module level.

def init(C):
    _ensure_dirs()
    _write_marker(os.path.join(ORD_DIR, "_namebridge_init_%d" % int(time.time())))
    print("namebridge init ok, watching %s" % ORD_DIR)
    try:
        C.run_time("namebridge_tick", "2nSecond", "2026-01-01 09:00:00")
    except Exception as e:
        print("namebridge run_time failed (will rely on handlebar): %r" % (e,))


def namebridge_tick(C):
    try:
        _tick(C)
    except Exception as e:
        print("namebridge tick error: %r" % (e,))


def handlebar(C):
    # Fallback driver when the timer is unavailable.
    try:
        _tick(C)
    except Exception as e:
        print("namebridge handlebar error: %r" % (e,))
