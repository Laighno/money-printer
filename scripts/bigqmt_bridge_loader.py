# -*- coding: gbk -*-
# PASTE THIS (and only this) into the BigQMT strategy editor. It loads the real
# bridge strategy from disk, so future updates are just: git pull + restart
# strategy in QMT. No more re-pasting.
_SRC = "C:\\money-printer\\scripts\\bigqmt_bridge_strategy.py"
_f = open(_SRC, "rb")
_code = _f.read()
_f.close()
exec(compile(_code, _SRC, "exec"))
