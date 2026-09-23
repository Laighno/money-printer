"""sync_portfolio_from_qmt: Rule #4.1 gate on the config/portfolio.yaml write
(2026-09-23).

PROJECT_ROOT is monkeypatched to tmp_path so ``tmp/config/portfolio.yaml``
is classified as a protected prod path without touching the real file.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mp.common.paths as paths  # noqa: E402
from scripts import sync_portfolio_from_qmt as spq  # noqa: E402


@pytest.fixture
def prot(tmp_path, monkeypatch):
    """tmp_path acts as PROJECT_ROOT; returns the (protected) yaml path."""
    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(paths, "AUDIT_LOG_PATH", tmp_path / "data" / "audit" / "prod_writes.log")
    monkeypatch.delenv("MP_ALLOW_PROD_WRITE", raising=False)
    p = tmp_path / "config" / "portfolio.yaml"
    p.parent.mkdir(parents=True)
    p.write_text("# header\naccount:\n  total_assets: 1.00\n", encoding="utf-8")
    assert paths.is_protected_prod_path(p)
    return p


def test_write_refused_without_env(prot):
    before = prot.read_text(encoding="utf-8")
    with pytest.raises(RuntimeError, match="REFUSED prod state write"):
        spq.write_portfolio_yaml(prot, "# new\naccount:\n  total_assets: 2.00\n")
    assert prot.read_text(encoding="utf-8") == before
    assert not prot.with_suffix(".yaml.tmp").exists()   # refused before tmp


def test_write_allowed_with_env_and_audited(prot, monkeypatch):
    monkeypatch.setenv("MP_ALLOW_PROD_WRITE", "1")
    spq.write_portfolio_yaml(prot, "# new\naccount:\n  total_assets: 2.00\n")
    assert "total_assets: 2.00" in prot.read_text(encoding="utf-8")
    assert not prot.with_suffix(".yaml.tmp").exists()
    log = paths.AUDIT_LOG_PATH.read_text(encoding="utf-8")
    assert "portfolio.yaml" in log and "sync_portfolio_from_qmt" in log


def test_unprotected_path_needs_no_env(tmp_path, monkeypatch):
    monkeypatch.delenv("MP_ALLOW_PROD_WRITE", raising=False)
    monkeypatch.setattr(paths, "AUDIT_LOG_PATH", tmp_path / "audit.log")
    p = tmp_path / "elsewhere.yaml"
    p.write_text("old", encoding="utf-8")
    spq.write_portfolio_yaml(p, "new")
    assert p.read_text(encoding="utf-8") == "new"


def _snapshot():
    return {
        "account": {"total_assets": 100000.0, "cash_available": 40000.0,
                    "market_value": 60000.0, "updated_at": "x"},
        "positions": [{"code": "000001", "name": "A", "shares": 1000,
                       "avg_cost": 50.0, "market_price": 60.0,
                       "market_value": 60000.0}],
    }


def test_main_dry_run_unaffected_by_gate(prot, monkeypatch, capsys):
    monkeypatch.setattr(spq, "fetch_qmt_snapshot_local", _snapshot)
    monkeypatch.setattr(spq, "fetch_names", lambda codes: {"000001": "A"})
    monkeypatch.setattr(sys, "argv", ["sync", "--local", "--dry-run",
                                      "--portfolio", str(prot)])
    before = prot.read_text(encoding="utf-8")
    spq.main()   # no raise even though env is unset
    assert prot.read_text(encoding="utf-8") == before
    assert "dry-run, not written" in capsys.readouterr().out


def test_main_real_write_refused_without_env(prot, monkeypatch):
    monkeypatch.setattr(spq, "fetch_qmt_snapshot_local", _snapshot)
    monkeypatch.setattr(spq, "fetch_names", lambda codes: {"000001": "A"})
    monkeypatch.setattr(sys, "argv", ["sync", "--local", "--portfolio", str(prot)])
    before = prot.read_text(encoding="utf-8")
    with pytest.raises(RuntimeError, match="MP_ALLOW_PROD_WRITE"):
        spq.main()
    assert prot.read_text(encoding="utf-8") == before


def test_main_real_write_with_env(prot, monkeypatch):
    monkeypatch.setenv("MP_ALLOW_PROD_WRITE", "1")
    monkeypatch.setattr(spq, "fetch_qmt_snapshot_local", _snapshot)
    monkeypatch.setattr(spq, "fetch_names", lambda codes: {"000001": "A"})
    monkeypatch.setattr(sys, "argv", ["sync", "--local", "--portfolio", str(prot)])
    spq.main()
    txt = prot.read_text(encoding="utf-8")
    assert "total_assets: 100000.00" in txt and "code: '000001'" in txt


@pytest.mark.parametrize("ps1", ["scripts/ecs_daily_report.ps1",
                                 "scripts/ecs_intraday_execute.ps1"])
def test_ecs_ps1_sets_gate_before_sync_step(ps1):
    """The scheduled-task wrappers must set MP_ALLOW_PROD_WRITE=1 before
    invoking sync_portfolio_from_qmt.py, and those lines must be ASCII
    (ECS PowerShell parses .ps1 as GBK)."""
    text = (ROOT / ps1).read_text(encoding="utf-8")
    i_env = text.find('$env:MP_ALLOW_PROD_WRITE = "1"')
    i_sync = text.find("sync_portfolio_from_qmt.py")
    assert i_env != -1, f"{ps1}: env gate line missing"
    assert i_sync != -1
    # first sync invocation must come after an env line
    sync_call = re.search(r"^\$syncOutput = .*$", text, re.M)
    assert sync_call and i_env < sync_call.start()
    for line in text.splitlines():
        if "MP_ALLOW_PROD_WRITE" in line:
            assert line.isascii(), f"{ps1}: non-ASCII in gate line: {line!r}"
