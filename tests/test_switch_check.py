"""Tests for Graphify v2 pre-switch verification tooling (Findings 8 and 9).

All tests use temporary directories, mock Ollama / network, and never touch live databases.
"""

import json
import sqlite3
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from migrate_v2.switch_check import (
    check_v2_identity,
    find_running_graphify_processes,
    print_checklist,
)


def _create_mock_v2_sqlite(db_dir: Path, collection_name: str = "graphify", metadata: dict = None):
    """Helper to create a mock Chroma SQLite database with collection metadata."""
    meta = metadata or {
        "schema_version": 2,
        "backend": "ollama",
        "model": "bge-m3",
        "model_digest": "sha256:test_digest_123",
        "dimension": 1024,
    }
    db_file = db_dir / "chroma.sqlite3"
    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()
    cur.execute("CREATE TABLE collections (id TEXT PRIMARY KEY, name TEXT);")
    cur.execute("CREATE TABLE collection_metadata (collection_id TEXT, key TEXT, str_value TEXT, int_value INT, float_value REAL, bool_value INT);")
    cur.execute("INSERT INTO collections VALUES ('col-1', ?)", (collection_name,))
    for k, v in meta.items():
        if isinstance(v, bool):
            cur.execute("INSERT INTO collection_metadata VALUES ('col-1', ?, NULL, NULL, NULL, ?)", (k, int(v)))
        elif isinstance(v, int):
            cur.execute("INSERT INTO collection_metadata VALUES ('col-1', ?, NULL, ?, NULL, NULL)", (k, v))
        elif isinstance(v, float):
            cur.execute("INSERT INTO collection_metadata VALUES ('col-1', ?, NULL, NULL, ?, NULL)", (k, v))
        else:
            cur.execute("INSERT INTO collection_metadata VALUES ('col-1', ?, ?, NULL, NULL, NULL)", (k, str(v)))
    conn.commit()
    conn.close()


def test_check_v2_identity_requires_db_and_detects_conflict_with_manifest(tmp_path):
    """Finding 8: check_v2_identity requires chroma.sqlite3 and compares collection identity with manifest."""
    v2_dir = tmp_path / "v2_test"
    v2_dir.mkdir()

    manifest_data = {
        "schema_version": 2,
        "backend": "ollama",
        "model": "bge-m3",
        "model_digest": "sha256:manifest_digest_abc",
        "dimension": 1024,
    }
    (v2_dir / "graphify_manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")

    # 1. DB file absent -> MUST FAIL
    res_no_db = check_v2_identity(v2_dir)
    assert res_no_db["status"] == "FAIL"
    assert "topilmadi" in res_no_db["error"]

    # 2. DB present but conflicting collection digest -> MUST FAIL
    conflicting_meta = {
        "schema_version": 2,
        "backend": "ollama",
        "model": "bge-m3",
        "model_digest": "sha256:contradictory_db_digest_xyz",
        "dimension": 1024,
    }
    _create_mock_v2_sqlite(v2_dir, metadata=conflicting_meta)

    res_conflict = check_v2_identity(v2_dir)
    assert res_conflict["status"] == "FAIL"
    assert "nomuvofiq" in res_conflict["error"] or "mos kelmadi" in res_conflict["error"]

    # 3. DB present with matching collection identity and mocked embedder -> PASS
    v2_dir_ok = tmp_path / "v2_ok"
    v2_dir_ok.mkdir()
    (v2_dir_ok / "graphify_manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")
    _create_mock_v2_sqlite(v2_dir_ok, metadata=manifest_data)

    mock_embedder = Mock()
    mock_embedder.encode.return_value = np.ones((1, 1024), dtype=np.float32)

    with patch("embedders.build_identity_embedder", return_value=mock_embedder):
        res_ok = check_v2_identity(v2_dir_ok)
        assert res_ok["status"] == "PASS"
        assert res_ok["dimension"] == 1024
        assert res_ok["model"] == "bge-m3"


def test_print_checklist_prints_pid_and_executable_only_without_token(monkeypatch):
    """Finding 9: switch_check prints PID and executable name only; never full command lines or secret tokens."""
    processes = [
        {
            "pid": 4321,
            "executable": "python.exe",
            "command": "python.exe server.py --token SECRET_SUPER_TOKEN_12345 --api-key SECRET_KEY",
        },
        {
            "pid": 5678,
            "command": "python server_http.py --token SECRET_HTTP_TOKEN",
        },
    ]
    drift = {"status": "OK", "total_drift": 0, "total_v1": 100}
    identity = {
        "status": "PASS",
        "model": "bge-m3",
        "backend": "ollama",
        "dimension": 1024,
        "digest": "sha256:testdigest1234567890",
    }
    launch_configs = []

    captured = StringIO()
    monkeypatch.setattr(sys, "stdout", captured)

    print_checklist(
        processes=processes,
        drift=drift,
        identity=identity,
        launch_configs=launch_configs,
        v1_dir=Path("kg_db"),
        v2_dir=Path("kg_db_v2"),
    )

    output = captured.getvalue()
    assert "PID 4321: python.exe" in output
    assert "PID 5678: python" in output
    # Strict secret redaction: NO token or secret string must appear in the output!
    assert "SECRET_SUPER_TOKEN" not in output
    assert "SECRET_KEY" not in output
    assert "SECRET_HTTP_TOKEN" not in output
    assert "--token" not in output
    assert "--api-key" not in output

def test_check_v2_identity_fails_when_graphify_collection_missing(tmp_path):
    """Finding 8 regression (TOPILMA 4): When DB has collections but lacks 'graphify', check_v2_identity must FAIL."""
    v2_dir = tmp_path / "v2_no_graphify"
    v2_dir.mkdir()
    manifest_data = {
        "schema_version": 2,
        "backend": "ollama",
        "model": "bge-m3",
        "model_digest": "sha256:manifest_digest_abc",
        "dimension": 1024,
    }
    (v2_dir / "graphify_manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")
    # Create SQLite with 'other_col' instead of 'graphify'
    _create_mock_v2_sqlite(v2_dir, collection_name="other_col", metadata=manifest_data)

    res = check_v2_identity(v2_dir)
    assert res["status"] == "FAIL"
    assert "graphify" in res["error"]


def test_check_v2_identity_fails_when_collection_missing_mandatory_identity_fields(tmp_path):
    """Finding 8 regression (TOPILMA 5): Collection with only schema_version=2 must FAIL due to missing mandatory identity fields."""
    v2_dir = tmp_path / "v2_sparse_meta"
    v2_dir.mkdir()
    manifest_data = {
        "schema_version": 2,
        "backend": "ollama",
        "model": "bge-m3",
        "model_digest": "sha256:manifest_digest_abc",
        "dimension": 1024,
    }
    (v2_dir / "graphify_manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")

    # Collection only has schema_version, missing backend/model/dimension/digest
    sparse_meta = {"schema_version": 2}
    _create_mock_v2_sqlite(v2_dir, collection_name="graphify", metadata=sparse_meta)

    res = check_v2_identity(v2_dir)
    assert res["status"] == "FAIL"
    assert "identifikatsiyasi topilmadi" in res["error"] or "topilmadi" in res["error"]


def test_find_running_graphify_processes_returns_pid_and_executable_only(monkeypatch):
    """Finding 9 regression: find_running_graphify_processes returns only pid and executable, never commandline or tokens."""
    import subprocess
    fake_ps_json = json.dumps([
        {
            "ProcessId": 1234,
            "Name": "python.exe",
            "CommandLine": "python.exe server.py --token VERY_SECRET_API_TOKEN_XYZ --port 8000"
        },
        {
            "ProcessId": 5678,
            "Name": "python3.exe",
            "CommandLine": "python3.exe bot.py --telegram-token SECRET_BOT_TOKEN_123"
        },
        {
            "ProcessId": 9999,
            "Name": "python.exe",
            "CommandLine": "python.exe unrelated_script.py"
        }
    ])

    mock_run = Mock()
    mock_run.returncode = 0
    mock_run.stdout = fake_ps_json

    with patch("subprocess.run", return_value=mock_run):
        procs = find_running_graphify_processes()
        assert len(procs) == 2
        for p in procs:
            assert set(p.keys()) == {"pid", "executable"}
            assert "token" not in str(p).lower()
            assert "commandline" not in p

