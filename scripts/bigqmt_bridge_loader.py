# -*- coding: gbk -*-
# PASTE THIS into the BigQMT strategy editor. It loads the real bridge strategy
# from disk so updates are just: git pull + restart strategy (no re-pasting).
#
# v2 (2026-08-31): QMT statically scans the pasted source for "def init" /
# "def handlebar" entry points; a bare exec() exposes none, so the strategy
# loads but never runs. Define the entries HERE and forward to the impl.
_SRC = "C:\\money-printer\\scripts\\bigqmt_bridge_strategy.py"
_g = {}
_f = open(_SRC, "rb")
_code = _f.read()
_f.close()
exec(compile(_code, _SRC, "exec"), _g)


def init(C):
    _g["init"](C)


def handlebar(C):
    _g["handlebar"](C)


def bridge_tick(C):
    # run_time resolves the callback by NAME in the strategy globals, so this
    # top-level forwarder must exist too.
    _g["bridge_tick"](C)
