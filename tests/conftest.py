# -*- coding: utf-8 -*-
"""pytest 公共设置：把 scripts/ 与 sim/ 加入导入路径。"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub in ("scripts", "sim", "tests"):
    path = os.path.join(ROOT, sub)
    if path not in sys.path:
        sys.path.insert(0, path)
