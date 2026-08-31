# -*- coding: gbk -*-
# BigQMT built-in bridge strategy for money-printer (miniQMT retirement 2026-09-27).
#
# WHAT: runs INSIDE the full QMT client's built-in Python (3.6, GBK). Serves as
# a thin execution bridge: external Python 3.12 (FileBridgeBroker) writes request
# JSON files under BRIDGE_DIR; this strategy polls them on a timer, executes via
# built-in APIs (passorder / cancel / get_trade_detail_data), writes response
# JSON. All heavy logic (models, plans, reconcile, price rules) stays external.
#
# PROTOCOL (all files UTF-8 JSON, atomic write = tmp then os.rename):
#   req_<seq>.json  : {"seq": int, "cmd": "snapshot"|"order"|"cancel", "args": {...}}
#     order args : {"code":"600000","action":"buy"|"sell","shares":100,
#                   "limit_price":12.34,"remark":"mp-xxxx"}
#     cancel args: {"order_id": "123456"}
#   resp_<seq>.json : {"seq":..,"ok":bool,"error":str|None,"data":{...}}
#   heartbeat.json  : {"ts": epoch_sec, "acct": "..."} refreshed every timer tick.
#   processed req files are renamed to done_req_<seq>.json (idempotency).
#
# DEPLOY: QMT client -> new strategy -> paste this file -> set run_time timer.
# Uses ContextInfo C. Account id set below. ASCII-only strings (GBK safety).
#
# Order flow notes (from MiniQMT_2_BigQMT mapping, verified constraints):
#   - passorder(23=buy, 24=sell, 1101=by volume, prType 11=limit, ...)
#   - order id is NOT returned synchronously; we stamp userOrderId (=seq) via
#     strategy remark and resolve the exchange order id on the NEXT tick by
#     scanning today's orders for our remark. resp for "order" is therefore
#     written one tick late (external side polls up to 15s, fine).

import json
import os
import time

ACCOUNT = "8886933837"
BRIDGE_DIR = "C:\\money-printer\\data\\bridge"
HEARTBEAT = os.path.join(BRIDGE_DIR, "heartbeat.json")
PENDING_ORDERS_F = os.path.join(BRIDGE_DIR, "_pending_orders.json")

# passorder constants (normal cash account; margin would be 33/34)
OPTYPE_BUY = 23
OPTYPE_SELL = 24
ORDERBY_VOLUME = 1101
PRTYPE_LIMIT = 11


def _atomic_write(path, obj):
    tmp = path + ".tmp"
    f = open(tmp, "w")
    try:
        f.write(json.dumps(obj))
    finally:
        f.close()
    if os.path.exists(path):
        os.remove(path)
    os.rename(tmp, path)


def _load_pending():
    if os.path.exists(PENDING_ORDERS_F):
        try:
            f = open(PENDING_ORDERS_F, "r")
            try:
                return json.load(f)
            finally:
                f.close()
        except Exception:
            return []
    return []


def _save_pending(lst):
    _atomic_write(PENDING_ORDERS_F, lst)


def _code_to_qmt(code):
    # 6-digit -> QMT suffixed form
    if code.startswith("6"):
        return code + ".SH"
    return code + ".SZ"


def _plain(code_suffixed):
    return code_suffixed.split(".")[0]


INIT_MARKER = os.path.join(BRIDGE_DIR, "init_marker.json")
_last_tick = [0.0]


def init(C):
    # Write the marker FIRST so we can tell "init ran" from "timer dead" when
    # debugging from outside (2026-08-31: first paste produced an empty bridge
    # dir — init never proved it ran).
    if not os.path.isdir(BRIDGE_DIR):
        os.makedirs(BRIDGE_DIR)
    marker = {"ts": time.time(), "stage": "init-entered", "timer": None,
              "errors": []}
    _atomic_write(INIT_MARKER, marker)
    try:
        C.set_account(ACCOUNT)
    except Exception as e:
        marker["errors"].append("set_account: " + str(e))
    # run_time signature differs across QMT builds; try 4-arg then 3-arg.
    # handlebar() below is the always-works fallback driver either way.
    try:
        C.run_time("bridge_tick", "2nSecond", "2026-08-31 09:15:00", "SH")
        marker["timer"] = "4arg"
    except Exception as e1:
        marker["errors"].append("run_time 4arg: " + str(e1))
        try:
            C.run_time("bridge_tick", "2nSecond", "2026-08-31 09:15:00")
            marker["timer"] = "3arg"
        except Exception as e2:
            marker["errors"].append("run_time 3arg: " + str(e2))
            marker["timer"] = "none-handlebar-only"
    marker["stage"] = "init-done"
    _atomic_write(INIT_MARKER, marker)
    print("bridge init done, timer=" + str(marker["timer"]))


def handlebar(C):
    # Bar-driven fallback driver: fires on every bar/tick of the strategy's
    # benchmark instrument even when run_time is unsupported. Rate-limit to
    # one bridge pass per second.
    now = time.time()
    if now - _last_tick[0] >= 1.0:
        _last_tick[0] = now
        bridge_tick(C)


def _snapshot(C):
    acct_rows = get_trade_detail_data(ACCOUNT, "stock", "account")
    pos_rows = get_trade_detail_data(ACCOUNT, "stock", "position")
    ord_rows = get_trade_detail_data(ACCOUNT, "stock", "order")
    acct = {}
    for a in acct_rows:
        acct = {
            "cash_available": float(a.m_dAvailable),
            "cash_frozen": float(a.m_dFrozenCash),
            "market_value": float(a.m_dInstrumentValue),
            "total_assets": float(a.m_dBalance),
        }
    positions = []
    for p in pos_rows:
        positions.append({
            "code": _plain(p.m_strInstrumentID + "." + p.m_strExchangeID
                           if "." not in p.m_strInstrumentID else p.m_strInstrumentID),
            "name": "",
            "shares_total": int(p.m_nVolume),
            "shares_available": int(p.m_nCanUseVolume),
            "avg_cost": float(p.m_dOpenPrice),
            "market_price": float(p.m_dLastPrice),
            "market_value": float(p.m_dMarketValue),
        })
    orders = []
    # m_nOrderStatus: 48..57 range per QMT docs; map to our lifecycle
    smap = {48: "pending", 49: "pending", 50: "pending", 51: "pending",
            52: "pending", 53: "cancelled", 54: "cancelled", 55: "partial",
            56: "filled", 57: "rejected"}
    for o in ord_rows:
        side = "buy" if int(o.m_nOffsetFlag) == 48 else "sell"
        try:
            side = "buy" if int(o.m_nDirection) == 48 else "sell"
        except Exception:
            pass
        orders.append({
            "order_id": str(o.m_strOrderSysID),
            "code": _plain(o.m_strInstrumentID),
            "action": side,
            "shares_submitted": int(o.m_nVolumeTotalOriginal),
            "shares_filled": int(o.m_nVolumeTraded),
            "avg_fill_price": float(o.m_dTradedPrice) if int(o.m_nVolumeTraded) else 0.0,
            "status": smap.get(int(o.m_nOrderStatus), "pending"),
            "error_msg": None,
            "remark": getattr(o, "m_strRemark", ""),
        })
    return {"account": acct, "positions": positions, "orders": orders,
            "ts": time.time()}


def _handle_order(C, seq, args):
    code = _code_to_qmt(args["code"])
    action = args["action"]
    shares = int(args["shares"])
    price = float(args["limit_price"])
    remark = "mpseq" + str(seq)
    optype = OPTYPE_BUY if action == "buy" else OPTYPE_SELL
    # passorder 11-arg form: opType, orderType, account, code, prType, price,
    # volume, strategyName, quickTrade(1), userOrderId/remark, C
    passorder(optype, ORDERBY_VOLUME, ACCOUNT, code, PRTYPE_LIMIT, price,
              shares, "money-printer", 1, remark, C)
    # order id resolves next tick by remark scan
    pend = _load_pending()
    pend.append({"seq": seq, "remark": remark, "code": args["code"],
                 "action": action, "shares": shares, "limit_price": price,
                 "submitted_ts": time.time()})
    _save_pending(pend)


def _resolve_pending(C, snap):
    pend = _load_pending()
    if not pend:
        return
    remain = []
    by_remark = {}
    for o in snap["orders"]:
        r = o.get("remark") or ""
        if r:
            by_remark[r] = o
    for p in pend:
        o = by_remark.get(p["remark"])
        if o is not None:
            resp = {"seq": p["seq"], "ok": True, "error": None,
                    "data": {"order_id": o["order_id"], "code": p["code"],
                             "action": p["action"], "shares": p["shares"],
                             "limit_price": p["limit_price"]}}
            _atomic_write(os.path.join(BRIDGE_DIR, "resp_%d.json" % p["seq"]), resp)
        elif time.time() - p["submitted_ts"] > 20:
            resp = {"seq": p["seq"], "ok": False,
                    "error": "order not visible in today orders after 20s "
                             "(likely rejected pre-exchange)", "data": None}
            _atomic_write(os.path.join(BRIDGE_DIR, "resp_%d.json" % p["seq"]), resp)
        else:
            remain.append(p)
    _save_pending(remain)


def bridge_tick(C):
    try:
        _atomic_write(HEARTBEAT, {"ts": time.time(), "acct": ACCOUNT})
        # take a snapshot once per tick (also used to resolve pending orders)
        snap = None
        names = os.listdir(BRIDGE_DIR)
        reqs = []
        for n in names:
            if n.startswith("req_") and n.endswith(".json"):
                try:
                    reqs.append(int(n[4:-5]))
                except Exception:
                    pass
        reqs.sort()
        for seq in reqs:
            rp = os.path.join(BRIDGE_DIR, "req_%d.json" % seq)
            try:
                f = open(rp, "r")
                try:
                    req = json.load(f)
                finally:
                    f.close()
            except Exception as e:
                _atomic_write(os.path.join(BRIDGE_DIR, "resp_%d.json" % seq),
                              {"seq": seq, "ok": False,
                               "error": "bad req json: " + str(e), "data": None})
                os.rename(rp, os.path.join(BRIDGE_DIR, "done_req_%d.json" % seq))
                continue
            cmd = req.get("cmd")
            try:
                if cmd == "snapshot":
                    if snap is None:
                        snap = _snapshot(C)
                    _atomic_write(os.path.join(BRIDGE_DIR, "resp_%d.json" % seq),
                                  {"seq": seq, "ok": True, "error": None,
                                   "data": snap})
                elif cmd == "order":
                    _handle_order(C, seq, req["args"])
                    # resp deferred to _resolve_pending on later ticks
                elif cmd == "cancel":
                    oid = req["args"]["order_id"]
                    cancel(oid, ACCOUNT, "stock", C)
                    _atomic_write(os.path.join(BRIDGE_DIR, "resp_%d.json" % seq),
                                  {"seq": seq, "ok": True, "error": None,
                                   "data": {"order_id": oid}})
                else:
                    _atomic_write(os.path.join(BRIDGE_DIR, "resp_%d.json" % seq),
                                  {"seq": seq, "ok": False,
                                   "error": "unknown cmd: " + str(cmd),
                                   "data": None})
            except Exception as e:
                _atomic_write(os.path.join(BRIDGE_DIR, "resp_%d.json" % seq),
                              {"seq": seq, "ok": False, "error": str(e),
                               "data": None})
            os.rename(rp, os.path.join(BRIDGE_DIR, "done_req_%d.json" % seq))
        # resolve deferred order responses using a fresh snapshot
        pend = _load_pending()
        if pend:
            if snap is None:
                snap = _snapshot(C)
            _resolve_pending(C, snap)
    except Exception as e:
        print("bridge_tick error: " + str(e))
