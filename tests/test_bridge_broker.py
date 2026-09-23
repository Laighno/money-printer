"""FileBridgeBroker.connect() / resp read retry against a fake bridge dir."""
import json
import threading
import time

import pytest

from mp.execution import bridge_broker as bb
from mp.execution.bridge_broker import FileBridgeBroker

ACCT = "8886933837"


@pytest.fixture
def fast_retry(monkeypatch):
    monkeypatch.setattr(bb, "READ_RETRY_INTERVAL", 0.02)
    monkeypatch.setattr(bb, "RESP_POLL_INTERVAL", 0.01)


def _hb(d, ts=None, acct=ACCT):
    (d / "heartbeat.json").write_text(
        json.dumps({"ts": time.time() if ts is None else ts, "acct": acct}),
        encoding="utf-8")


def test_connect_ok(tmp_path, fast_retry):
    _hb(tmp_path)
    b = FileBridgeBroker(ACCT, str(tmp_path))
    assert b.connect() is True
    assert b.is_connected()


def test_connect_missing_heartbeat_retries_then_fails(tmp_path, fast_retry, monkeypatch):
    sleeps = []
    monkeypatch.setattr(bb.time, "sleep", lambda s: sleeps.append(s))
    b = FileBridgeBroker(ACCT, str(tmp_path))
    assert b.connect() is False
    # 3 attempts -> 2 sleeps between them
    assert len(sleeps) == bb.READ_RETRIES - 1


def test_connect_survives_missing_then_present(tmp_path, fast_retry, monkeypatch):
    """Simulates the old remove()+rename() gap: first read hits ENOENT, the
    file shows up before the retry budget is exhausted."""
    calls = {"n": 0}
    real_sleep = time.sleep

    def _sleep(s):
        calls["n"] += 1
        if calls["n"] == 1:
            _hb(tmp_path)
        real_sleep(0)

    monkeypatch.setattr(bb.time, "sleep", _sleep)
    b = FileBridgeBroker(ACCT, str(tmp_path))
    assert b.connect() is True
    assert calls["n"] == 1


def test_connect_survives_half_written_json(tmp_path, fast_retry, monkeypatch):
    hb = tmp_path / "heartbeat.json"
    hb.write_text('{"ts": 17', encoding="utf-8")   # truncated
    calls = {"n": 0}

    def _sleep(s):
        calls["n"] += 1
        _hb(tmp_path)

    monkeypatch.setattr(bb.time, "sleep", _sleep)
    b = FileBridgeBroker(ACCT, str(tmp_path))
    assert b.connect() is True
    assert calls["n"] == 1


def test_connect_persistently_bad_json_fails(tmp_path, fast_retry):
    (tmp_path / "heartbeat.json").write_text("not json", encoding="utf-8")
    assert FileBridgeBroker(ACCT, str(tmp_path)).connect() is False


def test_connect_stale_and_wrong_account(tmp_path, fast_retry):
    _hb(tmp_path, ts=time.time() - 60)
    assert FileBridgeBroker(ACCT, str(tmp_path)).connect() is False
    _hb(tmp_path, acct="other")
    assert FileBridgeBroker(ACCT, str(tmp_path)).connect() is False


def test_rpc_resp_half_written_then_complete(tmp_path, fast_retry):
    _hb(tmp_path)
    b = FileBridgeBroker(ACCT, str(tmp_path))
    assert b.connect()

    def responder():
        # wait for the req, write a truncated resp, then the real one
        deadline = time.time() + 5
        while time.time() < deadline:
            reqs = list(tmp_path.glob("req_*.json"))
            if reqs:
                seq = int(reqs[0].stem[4:])
                rp = tmp_path / f"resp_{seq}.json"
                rp.write_text('{"seq": ', encoding="utf-8")
                time.sleep(0.05)
                rp.write_text(json.dumps({"seq": seq, "ok": True, "error": None,
                                          "data": {"account": {}, "positions": [],
                                                   "orders": []}}),
                              encoding="utf-8")
                return
            time.sleep(0.01)

    t = threading.Thread(target=responder, daemon=True)
    t.start()
    snap = b._rpc("snapshot", timeout=5)
    t.join(1)
    assert snap["ok"] is True
    assert not list(tmp_path.glob("resp_*.json"))


def test_rpc_timeout(tmp_path, fast_retry):
    _hb(tmp_path)
    b = FileBridgeBroker(ACCT, str(tmp_path))
    assert b.connect()
    with pytest.raises(TimeoutError):
        b._rpc("snapshot", timeout=0.1)


def test_prune_bridge_dir_removes_only_stale_archives(tmp_path):
    import os
    import time as _t
    from mp.execution.bridge_broker import prune_bridge_dir
    old = _t.time() - 30 * 3600
    for name in ["done_req_1.json", "resp_1.json", "done_req_2.json", "resp_2.json",
                 "heartbeat.json", "init_marker.json", "_pending_orders.json", "req_3.json"]:
        (tmp_path / name).write_text("{}", encoding="utf-8")
    for name in ["done_req_1.json", "resp_1.json", "heartbeat.json", "_pending_orders.json"]:
        os.utime(tmp_path / name, (old, old))
    removed = prune_bridge_dir(tmp_path, max_age_hours=24.0)
    assert removed == 2
    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert remaining == sorted(["done_req_2.json", "resp_2.json", "heartbeat.json",
                                "init_marker.json", "_pending_orders.json", "req_3.json"])
