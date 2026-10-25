"""Tests for the filename-channel broker (2026-10-25 QMT file-I/O ban).

The QMT built-in Python can no longer READ files but can still write them and
call os.listdir, so orders reach the strategy encoded in filenames. These tests
pin the wire format and the failure modes that matter for real money: never
double-submit, and never report success without a receipt.
"""

from __future__ import annotations

import time

import pytest

from mp.execution.namebridge_broker import (
    HEARTBEAT_MAX_AGE,
    NameBridgeBroker,
    decode_receipt,
    encode_order,
)


# ------------------------------------------------------------------ encoding
def test_encode_round_trips_through_the_strategy_parser():
    """The name this side writes must parse to the same order on the QMT side.

    The strategy's _parse is reimplemented here rather than imported: it lives
    in a GBK, Python-3.6 file that only the QMT client loads.
    """
    name = encode_order("003", "B", "600363", 200, 8.76, "mp1025")
    assert name == "003__B__600363__200__000876__mp1025.ord"

    parts = name[:-4].split("__")
    assert parts[0] == "003"
    assert parts[1] == "B"
    assert parts[2] == "600363"
    assert int(parts[3]) == 200
    assert int(parts[4]) / 100.0 == pytest.approx(8.76)


def test_price_survives_as_integer_fen():
    """Prices that bite in float land must come back exactly."""
    for price, fen in [(8.76, "000876"), (0.01, "000001"), (281.86, "028186"),
                       (10.00, "001000"), (1234.56, "123456")]:
        name = encode_order("1", "B", "600000", 100, price)
        assert name.split("__")[4] == fen
        assert int(fen) / 100.0 == pytest.approx(price)


def test_codes_are_zero_padded():
    assert encode_order("1", "S", "582", 100, 8.77).split("__")[2] == "000582"


def test_name_stays_far_under_the_measured_listdir_ceiling():
    """listdir returned a 184-char name intact; keep a wide margin."""
    name = encode_order("999999999", "B", "600363", 999900, 9999.99, "verylongtag")
    assert len(name) < 80, f"name grew to {len(name)} chars: {name}"


@pytest.mark.parametrize("kwargs", [
    {"side": "X"},                      # not B/S
    {"shares": 150},                    # not a round lot
    {"shares": 0},
    {"shares": -100},
    {"price": 0},
    {"price": -1.0},
])
def test_malformed_orders_are_rejected_before_they_reach_the_broker(kwargs):
    base = {"seq": "1", "side": "B", "code": "600363", "shares": 100, "price": 8.76}
    base.update(kwargs)
    with pytest.raises(ValueError):
        encode_order(**base)


def test_tag_is_sanitised_so_it_cannot_break_the_separator():
    """A tag containing '__' would desync every field after it."""
    name = encode_order("1", "B", "600363", 100, 8.76, "a__b/c:d")
    assert name.count("__") == 5
    assert "/" not in name and ":" not in name


# ------------------------------------------------------------------ receipts
def test_decode_receipt_handles_present_and_missing_order_ids():
    assert decode_receipt("003__sent__xt1090606341__mp1025.rcpt") == {
        "seq": "003", "status": "sent",
        "order_id": "xt1090606341", "tag": "mp1025"}
    assert decode_receipt("004__failed__none__mp1025.rcpt")["order_id"] is None
    assert decode_receipt("notareceipt.txt") is None


# ------------------------------------------------------------------ liveness
def test_heartbeat_age_reads_the_timestamp_from_the_filename(tmp_path):
    """The strategy cannot write readable content, so time lives in the name."""
    b = NameBridgeBroker(ord_dir=str(tmp_path))
    b.hb_dir.mkdir(parents=True)
    (b.hb_dir / f"{int(time.time()) - 5}.hb").touch()
    age = b.heartbeat_age()
    assert age is not None and 3 <= age <= 10
    assert b.is_connected()


def test_missing_or_stale_heartbeat_means_not_connected(tmp_path):
    b = NameBridgeBroker(ord_dir=str(tmp_path))
    b.hb_dir.mkdir(parents=True)
    assert b.heartbeat_age() is None
    assert not b.is_connected()
    assert not b.connect()

    (b.hb_dir / f"{int(time.time()) - int(HEARTBEAT_MAX_AGE) - 60}.hb").touch()
    assert not b.is_connected(), "a stale heartbeat must not count as alive"


def test_newest_heartbeat_wins(tmp_path):
    b = NameBridgeBroker(ord_dir=str(tmp_path))
    b.hb_dir.mkdir(parents=True)
    now = int(time.time())
    for offset in (600, 300, 3):
        (b.hb_dir / f"{now - offset}.hb").touch()
    assert b.heartbeat_age() < 10


# ------------------------------------------------------------------- orders
def test_order_file_is_created_and_receipt_is_honoured(tmp_path):
    b = NameBridgeBroker(ord_dir=str(tmp_path))
    b.done_dir.mkdir(parents=True)
    # Pre-stage the receipt the strategy would write.
    (b.done_dir / "777__sent__xt42__mp.rcpt").touch()

    res = b.place_limit_order("600363", "buy", 100, 8.76, seq="777", timeout=2)
    assert res.success and res.order_id == "xt42" and res.status == "sent"
    assert (tmp_path / "777__B__600363__100__000876__mp.ord").exists()


def test_failed_receipt_is_not_reported_as_success(tmp_path):
    b = NameBridgeBroker(ord_dir=str(tmp_path))
    b.done_dir.mkdir(parents=True)
    (b.done_dir / "778__failed__none__mp.rcpt").touch()

    res = b.place_limit_order("600363", "buy", 100, 8.76, seq="778", timeout=2)
    assert not res.success
    assert res.status == "failed"


def test_claimed_order_without_receipt_is_unknown_not_failed(tmp_path):
    """The strategy renames .ord -> .taken BEFORE calling passorder.

    So a vanished order file with no receipt means the order may be live at the
    exchange. Reporting that as a clean failure would invite a double submit on
    the next run, which is the one outcome real money cannot tolerate.
    """
    b = NameBridgeBroker(ord_dir=str(tmp_path))
    b.done_dir.mkdir(parents=True)

    import threading

    def claim():
        time.sleep(0.3)
        p = tmp_path / "779__B__600363__100__000876__mp.ord"
        if p.exists():
            p.rename(str(p)[:-4] + ".taken")

    threading.Thread(target=claim, daemon=True).start()
    res = b.place_limit_order("600363", "buy", 100, 8.76, seq="779", timeout=1.5)
    assert not res.success
    assert res.status == "unknown_submitted"


def test_untouched_order_without_receipt_is_plain_no_receipt(tmp_path):
    """File still sitting there = strategy never saw it = safe to retry."""
    b = NameBridgeBroker(ord_dir=str(tmp_path))
    b.done_dir.mkdir(parents=True)
    res = b.place_limit_order("600363", "buy", 100, 8.76, seq="780", timeout=1.0)
    assert not res.success
    assert res.status == "no_receipt"


def test_prune_removes_only_consumed_files(tmp_path):
    b = NameBridgeBroker(ord_dir=str(tmp_path))
    b.done_dir.mkdir(parents=True)
    b.hb_dir.mkdir(parents=True)
    old = time.time() - 48 * 3600

    pending = tmp_path / "900__B__600363__100__000876__mp.ord"
    pending.touch()
    for p in [tmp_path / "899__B__600363__100__000876__mp.taken",
              b.done_dir / "899__sent__xt1__mp.rcpt",
              b.hb_dir / "1000.hb"]:
        p.touch()
        import os as _os
        _os.utime(p, (old, old))

    removed = b.prune(max_age_hours=24.0)
    assert removed == 3
    assert pending.exists(), "an unsubmitted order must never be pruned"
