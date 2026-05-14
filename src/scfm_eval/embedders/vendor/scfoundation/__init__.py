"""Vendored scFoundation code (upstream: BioMap).

We keep upstream imports as-is (`from pretrainmodels import ...`, `from load import ...`)
by injecting this directory onto sys.path at package import time. This way the
upstream code remains a drop-in we can update from the original repo with no
manual patching.
"""
import os
import sys

_VENDOR_DIR = os.path.dirname(os.path.abspath(__file__))
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)
