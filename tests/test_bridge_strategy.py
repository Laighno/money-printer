"""Unit tests for scripts/bigqmt_bridge_strategy.py (BigQMT built-in side).

The module is importable on any platform: nothing at module level touches the
QMT built-ins (passorder / cancel / get_trade_detail_data), so we load it with
importlib and exercise the pure helpers with BRIDGE_DIR redirected to tmp_path.
"""
import importlib.util
import json
import os
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "scripts" / "bigqmt_bridge_strategy.py"


@pytest.fixture(scope="module")
def bqs():
    spec = importlib.util.spec_from_file_location("bigqmt_bridge_strategy", SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def bridge_dir(bqs, tmp_path, monkeypatch):
    d = tmp_path / "bridge"
    d.mkdir()
    monkeypatch.setattr(bqs, "BRIDGE_DIR", str(d))
    monkeypatch.setattr(bqs, "HEARTBEAT", str(d / "heartbeat.json"))
    monkeypatch.setattr(bqs, "PENDING_ORDERS_F", str(d / "_pending_orders.json"))
    return d


# ── source hygiene (Python 3.6 / GBK constraints) ────────────────────────────

def test_source_is_pure_ascii():
    raw = SRC.read_bytes()
    bad = [(i + 1, line) for i, line in enumerate(raw.splitlines())
           if any(b > 0x7F for b in line)]
    assert not bad, f"non-ASCII bytes in bridge strategy: {bad[:3]}"


def test_source_parses():
    import ast
    ast.parse(SRC.read_bytes())


# ── _atomic_write ────────────────────────────────────────────────────────────

def test_atomic_write_creates_and_replaces(bqs, tmp_path):
    p = str(tmp_path / "x.json")
    bqs._atomic_write(p, {"a": 1})
    assert json.loads(Path(p).read_text()) == {"a": 1}
    assert not os.path.exists(p + ".tmp")
    # overwrite an existing target (this is the path that used to remove+rename)
    bqs._atomic_write(p, {"a": 2})
    assert json.loads(Path(p).read_text()) == {"a": 2}
    assert not os.path.exists(p + ".tmp")


def test_atomic_write_uses_replace_not_remove(bqs, tmp_path, monkeypatch):
    """The target must never be removed before the new content lands."""
    p = str(tmp_path / "hb.json")
    bqs._atomic_write(p, {"ts": 1})
    calls = []

    def _boom(*a, **k):
        calls.append(("remove", a))
        raise AssertionError("os.remove must not be used by _atomic_write")

    monkeypatch.setattr(bqs.os, "remove", _boom)
    monkeypatch.setattr(bqs.os, "rename", _boom)
    real_replace = bqs.os.replace
    monkeypatch.setattr(bqs.os, "replace",
                        lambda a, b: (calls.append(("replace", a, b)), real_replace(a, b))[1])
    bqs._atomic_write(p, {"ts": 2})
    assert calls == [("replace", p + ".tmp", p)]
    assert json.loads(Path(p).read_text()) == {"ts": 2}


def test_atomic_write_fsyncs(bqs, tmp_path, monkeypatch):
    synced = []
    monkeypatch.setattr(bqs.os, "fsync", lambda fd: synced.append(fd))
    bqs._atomic_write(str(tmp_path / "y.json"), [1, 2])
    assert len(synced) == 1


# ── _match_new ───────────────────────────────────────────────────────────────

def _row(oid, code="600000", action="buy", shares=100, px=10.0):
    return {"order_id": oid, "code": code, "action": action,
            "shares_submitted": shares, "shares_filled": 0,
            "avg_fill_price": 0.0, "status": "pending", "error_msg": None,
            "remark": "", "limit_price": px}


def _pend(seq, known=(), code="600000", action="buy", shares=100, px=10.0):
    return {"seq": seq, "remark": "mpseq%d" % seq, "code": code,
            "action": action, "shares": shares, "limit_price": px,
            "known_ids": list(known), "submitted_ts": time.time()}


def test_match_new_unique(bqs):
    orders = [_row("OLD", px=10.0), _row("NEW", px=10.0), _row("OTHER", code="000001")]
    got = bqs._match_new(_pend(1, known=["OLD"]), orders)
    assert got is not None and got["order_id"] == "NEW"


def test_match_new_ambiguous_returns_none(bqs, capsys):
    orders = [_row("A"), _row("B")]
    assert bqs._match_new(_pend(1), orders) is None
    out = capsys.readouterr().out
    assert "ambiguous" in out and "A,B" in out
    assert "price_unchecked" not in out


def test_match_new_skips_claimed(bqs):
    orders = [_row("A"), _row("B")]
    got = bqs._match_new(_pend(1), orders, claimed={"A"})
    assert got is not None and got["order_id"] == "B"
    # everything claimed -> nothing
    assert bqs._match_new(_pend(1), orders, claimed={"A", "B"}) is None


def test_match_new_price_filter_and_unchecked_tag(bqs, capsys):
    # price mismatch beyond 1 fen excludes the row
    orders = [_row("FAR", px=10.05), _row("NEAR", px=10.01)]
    got = bqs._match_new(_pend(1, px=10.0), orders)
    assert got["order_id"] == "NEAR"
    # row without price -> degrade to not comparing, tagged in the ambiguity log
    orders = [_row("A", px=0.0), _row("B", px=0.0)]
    assert bqs._match_new(_pend(2, px=10.0), orders) is None
    assert "price_unchecked" in capsys.readouterr().out


def test_match_new_no_candidates(bqs):
    assert bqs._match_new(_pend(1, known=["X"]), [_row("X")]) is None


# ── _resolve_pending (claimed set across pendings) ───────────────────────────

def _read_resp(bridge_dir, seq):
    p = bridge_dir / ("resp_%d.json" % seq)
    return json.loads(p.read_text()) if p.exists() else None


def test_resolve_pending_two_identical_orders_no_double_claim(bqs, bridge_dir):
    # pending 1 was submitted when X already existed -> its only new id is Y
    # pending 2 was submitted before X appeared -> candidates X,Y; after 1
    # claims Y, 2 must get X (not Y again).
    bqs._save_pending([_pend(1, known=["X"]), _pend(2)])
    snap = {"orders": [_row("X"), _row("Y")], "account": {}, "positions": []}
    bqs._resolve_pending(None, snap)
    r1 = _read_resp(bridge_dir, 1)
    r2 = _read_resp(bridge_dir, 2)
    assert r1["ok"] and r1["data"]["order_id"] == "Y"
    assert r2["ok"] and r2["data"]["order_id"] == "X"
    assert bqs._load_pending() == []


def test_resolve_pending_ambiguous_stays_pending_then_times_out(bqs, bridge_dir, capsys):
    p1 = _pend(1)
    p2 = _pend(2)
    bqs._save_pending([p1, p2])
    snap = {"orders": [_row("X"), _row("Y")], "account": {}, "positions": []}
    bqs._resolve_pending(None, snap)
    # both ambiguous -> no resp, both remain
    assert _read_resp(bridge_dir, 1) is None and _read_resp(bridge_dir, 2) is None
    assert [p["seq"] for p in bqs._load_pending()] == [1, 2]
    assert capsys.readouterr().out.count("ambiguous") == 2
    # age them past the 20s budget -> failed resp
    aged = bqs._load_pending()
    for p in aged:
        p["submitted_ts"] = time.time() - 30
    bqs._save_pending(aged)
    bqs._resolve_pending(None, snap)
    assert _read_resp(bridge_dir, 1)["ok"] is False
    assert _read_resp(bridge_dir, 2)["ok"] is False
    assert bqs._load_pending() == []


def test_resolve_pending_remark_path_respects_claimed(bqs, bridge_dir):
    # If QMT ever returns remarks, a remark hit already claimed by an earlier
    # pending must not be handed out twice.
    rowA = _row("A"); rowA["remark"] = "mpseq1"
    bqs._save_pending([_pend(1), _pend(2)])
    snap = {"orders": [rowA], "account": {}, "positions": []}
    bqs._resolve_pending(None, snap)
    assert _read_resp(bridge_dir, 1)["data"]["order_id"] == "A"
    assert _read_resp(bridge_dir, 2) is None
    assert [p["seq"] for p in bqs._load_pending()] == [2]
