# -*- coding: utf-8 -*-
"""Standalone test runner for the CARAS statistical core.

The core is deliberately free of QGIS imports, so the whole suite runs in plain
CPython with NumPy alone::

    python tests/run_tests.py

It is also a valid pytest suite::

    pytest tests -q

Exit code 0 means every check passed.
"""

from __future__ import print_function

import importlib
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

MODULES = ["test_estimators", "test_disagreement", "test_regression",
           "test_report",
           # runs only inside a QGIS environment; skips itself otherwise
           "test_qgis_integration"]


def main():
    passed, failed = 0, []
    for name in MODULES:
        try:
            mod = importlib.import_module(name)
        except Exception:
            failed.append((name, "<import>", traceback.format_exc()))
            continue
        funcs = sorted(f for f in dir(mod) if f.startswith("test_"))
        print("\n%s  (%d checks)" % (name, len(funcs)))
        for fname in funcs:
            fn = getattr(mod, fname)
            if not callable(fn):
                continue
            try:
                fn()
            except Exception:
                failed.append((name, fname, traceback.format_exc()))
                print("   FAIL  %s" % fname)
            else:
                passed += 1
                print("   ok    %s" % fname)

    print("\n" + "=" * 66)
    print("%d passed, %d failed" % (passed, len(failed)))
    print("=" * 66)
    for mod, fn, tb in failed:
        print("\n--- %s.%s ---\n%s" % (mod, fn, tb))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
