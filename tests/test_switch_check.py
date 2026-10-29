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
        "distance": "cosine",
        "normalization": False,
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
        "distance": "cosine",
        "normalization": False,
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
        "distance": "cosine",
        "normalization": False,
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
        "distance": "cosine",
        "normalization": False,
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
    monkeypatch.setattr(sys, "platform", "win32")
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


def test_check_v2_identity_and_conflict_check_compares_schema_version_distance_normalization_regression_finding_8(tmp_path):
    """Finding 8 regression: _check_identity_conflict and check_v2_identity compare schema_version, distance, and normalization and FAIL on missing/conflicting values."""
    from db import _check_identity_conflict
    from embedders import ModelIdentityError
    from migrate_v2.switch_check import check_v2_identity

    # 1. _check_identity_conflict detects conflict in distance
    id_a = {"model": "bge-m3", "model_digest": "sha256:abc", "dimension": 1024, "backend": "ollama", "schema_version": 2, "distance": "cosine", "normalization": False}
    id_b = {"model": "bge-m3", "model_digest": "sha256:abc", "dimension": 1024, "backend": "ollama", "schema_version": 2, "distance": "l2", "normalization": False}
    with pytest.raises(ModelIdentityError, match="distance"):
        _check_identity_conflict(id_a, "manifest", id_b, "kolleksiya")

    # 2. _check_identity_conflict detects conflict in normalization
    id_c = {"model": "bge-m3", "model_digest": "sha256:abc", "dimension": 1024, "backend": "ollama", "schema_version": 2, "distance": "cosine", "normalization": True}
    with pytest.raises(ModelIdentityError, match="normalization"):
        _check_identity_conflict(id_a, "manifest", id_c, "kolleksiya")

    # 3. _check_identity_conflict detects conflict in schema_version
    id_d = {"model": "bge-m3", "model_digest": "sha256:abc", "dimension": 1024, "backend": "ollama", "schema_version": 1, "distance": "cosine", "normalization": False}
    with pytest.raises(ModelIdentityError, match="schema_version"):
        _check_identity_conflict(id_a, "manifest", id_d, "kolleksiya")

    # 4. check_v2_identity FAILs on missing distance in collection
    v2_dir = tmp_path / "v2_missing_dist"
    v2_dir.mkdir()
    (v2_dir / "graphify_manifest.json").write_text(json.dumps(id_a), encoding="utf-8")
    meta_no_dist = dict(id_a)
    del meta_no_dist["distance"]
    _create_mock_v2_sqlite(v2_dir, metadata=meta_no_dist)
    res = check_v2_identity(v2_dir)
    assert res["status"] == "FAIL"
    assert "distance" in res["error"]

    # 5. check_v2_identity FAILs on conflicting normalization
    v2_dir2 = tmp_path / "v2_conflicting_norm"
    v2_dir2.mkdir()
    (v2_dir2 / "graphify_manifest.json").write_text(json.dumps(id_a), encoding="utf-8")
    meta_diff_norm = dict(id_a)
    meta_diff_norm["normalization"] = True
    _create_mock_v2_sqlite(v2_dir2, metadata=meta_diff_norm)
    res2 = check_v2_identity(v2_dir2)
    assert res2["status"] == "FAIL"
    assert "normalization" in res2["error"]

    # 6. TOPILMA 2: garbage normalization rejected in _check_identity_conflict and check_v2_identity
    id_garbage = dict(id_a)
    id_garbage["normalization"] = "garbage"
    with pytest.raises(ModelIdentityError, match="normalization"):
        _check_identity_conflict(id_a, "manifest", id_garbage, "kolleksiya")

    v2_dir_garbage = tmp_path / "v2_garbage_norm"
    v2_dir_garbage.mkdir()
    (v2_dir_garbage / "graphify_manifest.json").write_text(json.dumps(id_garbage), encoding="utf-8")
    _create_mock_v2_sqlite(v2_dir_garbage, metadata=id_a)
    res_garbage = check_v2_identity(v2_dir_garbage)
    assert res_garbage["status"] == "FAIL"
    assert "normalization" in res_garbage["error"]


def test_check_v2_identity_catches_malformed_dimension_within_fail_boundary_regression_finding_9(tmp_path):
    """Finding 9 regression: malformed dimension value in manifest or collection returns FAIL instead of raising unhandled ValueError/OverflowError."""
    from migrate_v2.switch_check import check_v2_identity

    valid_id = {
        "schema_version": 2,
        "backend": "ollama",
        "model": "bge-m3",
        "model_digest": "sha256:abc",
        "dimension": 1024,
        "distance": "cosine",
        "normalization": False,
    }

    # Manifest with malformed dimension "not_a_number"
    v2_dir = tmp_path / "v2_bad_dim_manifest"
    v2_dir.mkdir()
    bad_manifest = dict(valid_id)
    bad_manifest["dimension"] = "not_a_number"
    (v2_dir / "graphify_manifest.json").write_text(json.dumps(bad_manifest), encoding="utf-8")
    _create_mock_v2_sqlite(v2_dir, metadata=valid_id)

    res = check_v2_identity(v2_dir)
    assert res["status"] == "FAIL"
    assert "dimension" in res["error"]

    # Collection with malformed dimension
    v2_dir2 = tmp_path / "v2_bad_dim_col"
    v2_dir2.mkdir()
    (v2_dir2 / "graphify_manifest.json").write_text(json.dumps(valid_id), encoding="utf-8")
    bad_col = dict(valid_id)
    bad_col["dimension"] = "invalid_int"
    _create_mock_v2_sqlite(v2_dir2, metadata=bad_col)

    res2 = check_v2_identity(v2_dir2)
    assert res2["status"] == "FAIL"
    assert "dimension" in res2["error"]

    # TOPILMA 3: dimension=1e309 raises OverflowError inside int() and must return FAIL within boundary
    v2_dir_inf = tmp_path / "v2_overflow_dim_manifest"
    v2_dir_inf.mkdir()
    overflow_manifest = dict(valid_id)
    overflow_manifest["dimension"] = 1e309
    (v2_dir_inf / "graphify_manifest.json").write_text(json.dumps(overflow_manifest), encoding="utf-8")
    _create_mock_v2_sqlite(v2_dir_inf, metadata=valid_id)

    res_inf = check_v2_identity(v2_dir_inf)
    assert res_inf["status"] == "FAIL"
    assert "dimension" in res_inf["error"]

    v2_dir_inf_col = tmp_path / "v2_overflow_dim_col"
    v2_dir_inf_col.mkdir()
    (v2_dir_inf_col / "graphify_manifest.json").write_text(json.dumps(valid_id), encoding="utf-8")
    overflow_col = dict(valid_id)
    overflow_col["dimension"] = 1e309
    _create_mock_v2_sqlite(v2_dir_inf_col, metadata=overflow_col)

    res_inf_col = check_v2_identity(v2_dir_inf_col)
    assert res_inf_col["status"] == "FAIL"
    assert "dimension" in res_inf_col["error"]

