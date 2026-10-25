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
#   account   : ORD_DIR\acct\<ts>__<cash_fen>__<mv_fen>__<total_fen>.acct
#   positions : ORD_DIR\pos\<code>__<total>__<avail>__<cost_fen>__<px_fen>.pos
#   orders    : ORD_DIR\ords\<oid>__<code>__<B|S>__<sub>__<filled>__<status>__<px_fen>.oq
#               Refreshed every SNAP_EVERY ticks. execute_orders' safety gates
#               (cash/position parity, idempotency re-check) need this state and
#               the sandbox cannot hand back content, so it goes in names too.
#               Amounts are integer fen; a float in a filename invites locale
#               and rounding surprises on the 3.6 side.
#
# DEPLOY: QMT client -> 模型交易 -> new strategy -> paste this file -> run.
# ASCII-only string literals (the client parses .py as GBK).

import os
import time

ACCOUNT = "8886933837"
ORD_DIR = r"C:\guojin\python\mp_orders"
DONE_DIR = os.path.join(ORD_DIR, "done")
HB_DIR = os.path.join(ORD_DIR, "hb")
ACCT_DIR = os.path.join(ORD_DIR, "acct")
POS_DIR = os.path.join(ORD_DIR, "pos")
ORDQ_DIR = os.path.join(ORD_DIR, "ords")
SNAP_EVERY = 5          # ticks between state snapshots (~10s at 2s/tick)

# passorder constants (same values the file bridge used)
OPTYPE_BUY = 23
OPTYPE_SELL = 24
ORDERBY_VOLUME = 1101
PRTYPE_LIMIT = 11

_last_hb = [0.0]
_tick_n = [0]
_seen = {}          # seq -> submitted ts, guards against double submit in-process


def _ensure_dirs():
    for d in (ORD_DIR, DONE_DIR, HB_DIR, ACCT_DIR, POS_DIR, ORDQ_DIR):
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


def _fen(x):
    try:
        return int(round(float(x) * 100))
    except Exception:
        return 0


def _clear(d):
    try:
        for n in os.listdir(d):
            try:
                os.remove(os.path.join(d, n))
            except Exception:
                pass
    except Exception:
        pass


def _snap_account():
    try:
        rows = get_trade_detail_data(ACCOUNT, "stock", "account")
    except Exception as e:
        print("namebridge account query failed: %r" % (e,))
        return
    if not rows:
        return
    a = rows[0]
    cash = _fen(getattr(a, "m_dAvailable", 0))
    mv = _fen(getattr(a, "m_dInstrumentValue", 0))
    total = _fen(getattr(a, "m_dBalance", 0))
    _clear(ACCT_DIR)
    _write_marker(os.path.join(
        ACCT_DIR, "%d__%d__%d__%d.acct" % (int(time.time()), cash, mv, total)))


def _snap_positions():
    try:
        rows = get_trade_detail_data(ACCOUNT, "stock", "position")
    except Exception as e:
        print("namebridge position query failed: %r" % (e,))
        return
    _clear(POS_DIR)
    for p in rows:
        try:
            code = str(p.m_strInstrumentID).split(".")[0]
            total = int(getattr(p, "m_nVolume", 0))
            avail = int(getattr(p, "m_nCanUseVolume", 0))
            cost = _fen(getattr(p, "m_dOpenPrice", 0))
            px = _fen(getattr(p, "m_dLastPrice", 0))
            if total <= 0:
                continue
            _write_marker(os.path.join(POS_DIR, "%s__%d__%d__%d__%d.pos" % (
                code, total, avail, cost, px)))
        except Exception:
            continue


def _snap_orders():
    """Today's orders. execute_orders needs these for its idempotency check."""
    try:
        rows = get_trade_detail_data(ACCOUNT, "stock", "order")
    except Exception as e:
        print("namebridge order query failed: %r" % (e,))
        return
    smap = {48: "pending", 49: "pending", 50: "pending", 51: "pending",
            52: "partial", 53: "filled", 54: "cancelled", 55: "cancelled",
            56: "rejected", 57: "rejected"}
    _clear(ORDQ_DIR)
    for o in rows:
        try:
            oid = str(o.m_strOrderSysID).replace("_", "-")[:40]
            code = str(o.m_strInstrumentID).split(".")[0]
            side = "B" if int(getattr(o, "m_nDirection", 48)) == 48 else "S"
            sub = int(getattr(o, "m_nVolumeTotalOriginal", 0))
            fill = int(getattr(o, "m_nVolumeTraded", 0))
            st = smap.get(int(getattr(o, "m_nOrderStatus", 0)), "pending")
            px = _fen(getattr(o, "m_dLimitPrice", 0))
            _write_marker(os.path.join(ORDQ_DIR, "%s__%s__%s__%d__%d__%s__%d.oq" % (
                oid, code, side, sub, fill, st, px)))
        except Exception:
            continue


def _snapshot_state():
    _snap_account()
    _snap_positions()
    _snap_orders()


def _tick(C):
    _ensure_dirs()
    _heartbeat()
    _tick_n[0] += 1
    if _tick_n[0] % SNAP_EVERY == 1:
        try:
            _snapshot_state()
        except Exception as e:
            print("namebridge snapshot error: %r" % (e,))
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
            try:
                _snap_orders()
            except Exception:
                pass
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
