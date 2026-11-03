"""Offline guards tested without importing any project/database dependency."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ("baza-tekshir.py", "indeks-qayta-qur.py", "indeks-smoke.py", "indeks_tekshir.py")
HARNESS = r'''
import builtins
import importlib.util
import runpy
import sqlite3
import sys

class UnsafeDependency(BaseException):
    pass

original_import = builtins.__import__
def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    resolved = name
    if level:
        resolved = importlib.util.resolve_name('.' * level + name, (globals or {}).get('__package__'))
    if resolved.partition('.')[0] not in sys.stdlib_module_names:
        raise UnsafeDependency(resolved)
    return original_import(name, globals, locals, fromlist, level)

def no_database(*args, **kwargs):
    raise UnsafeDependency('sqlite3.connect')

builtins.__import__ = guarded_import
sqlite3.connect = no_database
try:
    # The maintenance main is never executed, even if its guard regresses.
    runpy.run_path(sys.argv[1], run_name='offline_guard_test')
except RuntimeError as exc:
    if not str(exc).startswith('OFFLINE_ONLY:'):
        raise
    print('OFFLINE_GUARD')
except UnsafeDependency as exc:
    print('DEPENDENCY_BLOCKED:' + str(exc))
else:
    raise AssertionError('Neither guard nor dependency interceptor triggered')
'''


@pytest.mark.parametrize("script", SCRIPTS)
@pytest.mark.parametrize("mode", ["http", "unknown", "local", None])
def test_offline_tools_stop_before_database_imports(script, mode):
    env = os.environ.copy()
    env.pop("GRAPHIFY_STORAGE_MODE", None)
    if mode is not None:
        env["GRAPHIFY_STORAGE_MODE"] = mode
    result = subprocess.run(
        [sys.executable, "-I", "-c", HARNESS, str(ROOT / script)],
        env=env, capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr
    if mode in ("local", None):
        expected = "numpy" if script == "indeks-smoke.py" else "chromadb" if script == "baza-tekshir.py" else "src"
        assert result.stdout.strip() == "DEPENDENCY_BLOCKED:" + expected
    else:
        assert result.stdout.strip() == "OFFLINE_GUARD"
