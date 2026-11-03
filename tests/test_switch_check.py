"""Tests for Graphify v2 pre-switch verification tooling (Findings 8 and 9).

All tests use temporary directories, mock Ollama / network, and never touch live databases.
"""

import json
import sqlite3
import subprocess
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch
import chromadb
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from migrate_v2.switch_check import (
    check_v2_completeness,
    check_v2_identity,
    check_v2_normalization,
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


def test_print_checklist_prints_pid_and_executable_only_without_token(monkeypatch, tmp_path):
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
        v1_dir=tmp_path / "kg_db",
        v2_dir=tmp_path / "kg_db_v2",
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


def test_print_checklist_isolation_from_cwd(monkeypatch, tmp_path):
    """G7 / TOPILMA 3: print_checklist with tmp_path never opens or creates kg_db/kg_db_v2 in working directory."""
    isolated_cwd = tmp_path / "isolated_cwd"
    isolated_cwd.mkdir()
    monkeypatch.chdir(isolated_cwd)

    cwd_kg_db = isolated_cwd / "kg_db"
    cwd_kg_db_v2 = isolated_cwd / "kg_db_v2"
    assert not cwd_kg_db.exists()
    assert not cwd_kg_db_v2.exists()

    opened_db_paths = []
    real_persistent_client = chromadb.PersistentClient

    def tracking_persistent_client(path=None, *args, **kwargs):
        if path is not None:
            opened_db_paths.append(str(Path(path).resolve()))
        return real_persistent_client(path=path, *args, **kwargs)

    captured = StringIO()
    monkeypatch.setattr(sys, "stdout", captured)

    with patch("chromadb.PersistentClient", side_effect=tracking_persistent_client):
        print_checklist(
            processes=[],
            drift={"status": "OK", "total_drift": 0, "total_v1": 0},
            identity={"status": "PASS", "model": "bge-m3", "backend": "ollama", "dimension": 1024, "digest": "sha256:abc"},
            launch_configs=[],
            v1_dir=tmp_path / "iso_v1",
            v2_dir=tmp_path / "iso_v2",
        )

    assert not cwd_kg_db.exists()
    assert not cwd_kg_db_v2.exists()
    for p in opened_db_paths:
        assert str(isolated_cwd) not in p or p.startswith(str(tmp_path.resolve()))


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


def test_plain_and_prefixed_graphify_keys_conflict_fails_closed_regression_item_4(tmp_path):
    """Finding 4 regression: when both <key> and graphify:<key> exist with different values, fail closed."""
    from db import _check_identity_conflict
    from embedders import ModelIdentityError
    from migrate_v2.build_v2 import validate_destination_persisted_identity

    valid_id = {
        "schema_version": 2,
        "backend": "ollama",
        "model": "bge-m3",
        "model_digest": "sha256:abc",
        "dimension": 1024,
        "distance": "l2",
        "normalization": False,
    }

    # 1. _check_identity_conflict fails closed when intra-source keys conflict
    conflicting_src_a = dict(valid_id, **{"graphify:model": "other-model"})
    with pytest.raises(ModelIdentityError, match="mos kelmadi"):
        _check_identity_conflict(conflicting_src_a, "manifest", valid_id, "kolleksiya")

    conflicting_src_b = dict(valid_id, **{"graphify:dimension": 512})
    with pytest.raises(ModelIdentityError, match="mos kelmadi"):
        _check_identity_conflict(valid_id, "manifest", conflicting_src_b, "kolleksiya")

    # 2. check_v2_identity returns FAIL when collection has conflicting plain and prefixed keys
    col_dir = tmp_path / "f4_col_conflict"
    col_dir.mkdir()
    (col_dir / "graphify_manifest.json").write_text(json.dumps(valid_id), encoding="utf-8")
    conflicting_col = dict(valid_id, **{"graphify:model": "different-model"})
    _create_mock_v2_sqlite(col_dir, metadata=conflicting_col)
    res = check_v2_identity(col_dir)
    assert res["status"] == "FAIL"
    assert "mos kelmadi" in res["error"]

    # Manifest with conflicting plain and prefixed keys
    m_dir = tmp_path / "f4_manifest_conflict"
    m_dir.mkdir()
    conflicting_m = dict(valid_id, **{"graphify:backend": "openai"})
    (m_dir / "graphify_manifest.json").write_text(json.dumps(conflicting_m), encoding="utf-8")
    _create_mock_v2_sqlite(m_dir, metadata=valid_id)
    res_m = check_v2_identity(m_dir)
    assert res_m["status"] == "FAIL"
    assert "mos kelmadi" in res_m["error"]

    # 3. validate_destination_persisted_identity raises ModelIdentityError on prefix conflict
    dest_dir = tmp_path / "f4_build_v2_dest"
    dest_dir.mkdir()
    (dest_dir / "graphify_manifest.json").write_text(json.dumps(conflicting_m), encoding="utf-8")
    with pytest.raises(ModelIdentityError, match="mos kelmadi"):
        validate_destination_persisted_identity(
            out_dir=dest_dir,
            col_name="graphify",
            expected_model="bge-m3",
            expected_digest="sha256:abc",
            expected_dimension=1024,
            client=None,
        )


def test_dimension_and_schema_version_reject_bool_and_non_integer_floats_regression_item_5(tmp_path):
    """Finding 5 regression: dimension and schema_version reject bool and non-integer floats across switch_check, db, and build_v2."""
    from db import _check_identity_conflict
    from embedders import ModelIdentityError
    from migrate_v2.build_v2 import validate_destination_persisted_identity

    valid_id = {
        "schema_version": 2,
        "backend": "ollama",
        "model": "bge-m3",
        "model_digest": "sha256:abc",
        "dimension": 1024,
        "distance": "l2",
        "normalization": False,
    }

    # 1. switch_check check_v2_identity rejects bool and non-integer floats
    for bad_dim in [True, False, 1024.9]:
        d_dir = tmp_path / f"dim_m_{type(bad_dim).__name__}_{bad_dim}"
        d_dir.mkdir()
        bad_m = dict(valid_id, dimension=bad_dim)
        (d_dir / "graphify_manifest.json").write_text(json.dumps(bad_m), encoding="utf-8")
        _create_mock_v2_sqlite(d_dir, metadata=valid_id)
        res = check_v2_identity(d_dir)
        assert res["status"] == "FAIL"
        assert "dimension" in res["error"]

        c_dir = tmp_path / f"dim_c_{type(bad_dim).__name__}_{bad_dim}"
        c_dir.mkdir()
        (c_dir / "graphify_manifest.json").write_text(json.dumps(valid_id), encoding="utf-8")
        bad_c = dict(valid_id, dimension=bad_dim)
        _create_mock_v2_sqlite(c_dir, metadata=bad_c)
        res_c = check_v2_identity(c_dir)
        assert res_c["status"] == "FAIL"
        assert "dimension" in res_c["error"]

    for bad_schema in [True, False, 2.5]:
        s_dir = tmp_path / f"schema_m_{type(bad_schema).__name__}_{bad_schema}"
        s_dir.mkdir()
        bad_m = dict(valid_id, schema_version=bad_schema)
        (s_dir / "graphify_manifest.json").write_text(json.dumps(bad_m), encoding="utf-8")
        _create_mock_v2_sqlite(s_dir, metadata=valid_id)
        res = check_v2_identity(s_dir)
        assert res["status"] == "FAIL"
        assert "schema_version" in res["error"]

    # 2. _check_identity_conflict rejects bool and non-integer floats
    for bad_dim in [True, 1024.9]:
        with pytest.raises(ModelIdentityError, match="dimension"):
            _check_identity_conflict(dict(valid_id, dimension=bad_dim), "manifest", valid_id, "kolleksiya")

    for bad_schema in [True, 2.5]:
        with pytest.raises(ModelIdentityError, match="schema_version"):
            _check_identity_conflict(dict(valid_id, schema_version=bad_schema), "manifest", valid_id, "kolleksiya")

    # 3. validate_destination_persisted_identity rejects bool and non-integer floats
    v_dir = tmp_path / "val_dest_dim_float"
    v_dir.mkdir()
    (v_dir / "graphify_manifest.json").write_text(json.dumps(dict(valid_id, dimension=1024.9)), encoding="utf-8")
    with pytest.raises(ModelIdentityError, match="dimension"):
        validate_destination_persisted_identity(
            out_dir=v_dir,
            col_name="graphify",
            expected_model="bge-m3",
            expected_digest="sha256:abc",
            expected_dimension=1024,
            client=None,
        )

    v_dir2 = tmp_path / "val_dest_dim_bool"
    v_dir2.mkdir()
    (v_dir2 / "graphify_manifest.json").write_text(json.dumps(dict(valid_id, dimension=True)), encoding="utf-8")
    with pytest.raises(ModelIdentityError, match="dimension"):
        validate_destination_persisted_identity(
            out_dir=v_dir2,
            col_name="graphify",
            expected_model="bge-m3",
            expected_digest="sha256:abc",
            expected_dimension=1024,
            client=None,
        )


def test_check_v2_identity_rejects_float_dimension_and_conflicting_prefixed_fields(tmp_path):
    """Finding 3 & 4: check_v2_identity rejects float dimension and intra-source conflicting prefixed fields."""
    v2_dir = tmp_path / "v2_bad_dim"
    v2_dir.mkdir()
    valid_id = {
        "schema_version": 2,
        "backend": "ollama",
        "model": "bge-m3",
        "model_digest": "sha256:abc",
        "dimension": 1024,
        "distance": "cosine",
        "normalization": False,
    }
    (v2_dir / "graphify_manifest.json").write_text(json.dumps(valid_id), encoding="utf-8")
    _create_mock_v2_sqlite(v2_dir, metadata=dict(valid_id, dimension=1024.9))

    res = check_v2_identity(v2_dir)
    assert res["status"] == "FAIL"
    assert "dimension" in res["error"]


def test_conftest_db_isolation_regression():
    """T1 regression: conftest sets GRAPHIFY_DB_PATH at module level to temp dir,
    ensuring db.DB_PATH and keyword_index.DB_PATH never point to checkout kg_db.
    """
    import db
    import keyword_index
    import tempfile
    from pathlib import Path

    checkout_kg_db = Path(__file__).resolve().parents[1] / "kg_db"
    assert str(db.DB_PATH) != str(checkout_kg_db)
    assert str(keyword_index.DB_PATH) != str(checkout_kg_db / "keyword_index.sqlite3")

    temp_dir = tempfile.gettempdir()
    assert str(db.DB_PATH).startswith(temp_dir) or "graphify-test-" in str(db.DB_PATH)
    assert str(keyword_index.DB_PATH).startswith(temp_dir) or "graphify-test-" in str(keyword_index.DB_PATH)
    assert not checkout_kg_db.exists()


def test_find_running_graphify_processes_fail_closed_rc_nonzero(monkeypatch):
    """T2 regression: returncode != 0 causes find_running_graphify_processes to return None and checklist to FAIL."""
    monkeypatch.setattr(sys, "platform", "win32")
    mock_run = Mock()
    mock_run.returncode = 1
    mock_run.stdout = ""
    mock_run.stderr = "PowerShell error"

    with patch("subprocess.run", return_value=mock_run):
        procs = find_running_graphify_processes()
        assert procs is None

    captured = StringIO()
    monkeypatch.setattr(sys, "stdout", captured)
    ready = print_checklist(
        processes=procs,
        drift={"status": "OK", "total_drift": 0, "total_v1": 10},
        identity={"status": "PASS", "model": "m", "backend": "b", "dimension": 10, "digest": "d"},
        launch_configs=[],
        v1_dir=Path("v1"),
        v2_dir=Path("v2"),
        completeness={"status": "PASS", "count": 10},
    )
    output = captured.getvalue()
    assert "[FAIL] Yozuvchi jarayonlarni tekshirib bo'lmadi" in output
    assert "tayyor emas" in output
    assert ready is False


def test_find_running_graphify_processes_fail_closed_timeout(monkeypatch):
    """T2 regression: subprocess.TimeoutExpired causes find_running_graphify_processes to return None and checklist to FAIL."""
    monkeypatch.setattr(sys, "platform", "win32")

    def _timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="powershell", timeout=10)

    with patch("subprocess.run", side_effect=_timeout):
        procs = find_running_graphify_processes()
        assert procs is None

    captured = StringIO()
    monkeypatch.setattr(sys, "stdout", captured)
    ready = print_checklist(
        processes=procs,
        drift={"status": "OK", "total_drift": 0, "total_v1": 10},
        identity={"status": "PASS", "model": "m", "backend": "b", "dimension": 10, "digest": "d"},
        launch_configs=[],
        v1_dir=Path("v1"),
        v2_dir=Path("v2"),
        completeness={"status": "PASS", "count": 10},
    )
    output = captured.getvalue()
    assert "[FAIL] Yozuvchi jarayonlarni tekshirib bo'lmadi" in output
    assert "timeout" in output.lower()
    assert ready is False


def test_find_running_graphify_processes_fail_closed_invalid_json(monkeypatch):
    """T2 regression: invalid JSON from PowerShell causes find_running_graphify_processes to return None and checklist to FAIL."""
    monkeypatch.setattr(sys, "platform", "win32")
    mock_run = Mock()
    mock_run.returncode = 0
    mock_run.stdout = "This is not JSON at all!"

    with patch("subprocess.run", return_value=mock_run):
        procs = find_running_graphify_processes()
        assert procs is None

    captured = StringIO()
    monkeypatch.setattr(sys, "stdout", captured)
    ready = print_checklist(
        processes=procs,
        drift={"status": "OK", "total_drift": 0, "total_v1": 10},
        identity={"status": "PASS", "model": "m", "backend": "b", "dimension": 10, "digest": "d"},
        launch_configs=[],
        v1_dir=Path("v1"),
        v2_dir=Path("v2"),
        completeness={"status": "PASS", "count": 10},
    )
    output = captured.getvalue()
    assert "[FAIL] Yozuvchi jarayonlarni tekshirib bo'lmadi" in output
    assert ready is False


def test_find_running_graphify_processes_non_windows_returns_none(monkeypatch):
    """T2 regression: non-Windows OS returns None (fail-closed, not [])."""
    monkeypatch.setattr(sys, "platform", "linux")
    procs = find_running_graphify_processes()
    assert procs is None


def test_check_v2_completeness_missing_build_complete_fails(tmp_path):
    """T3 regression: missing build_complete.json causes check_v2_completeness to FAIL."""
    v2_dir = tmp_path / "v2_no_complete"
    v2_dir.mkdir()
    (v2_dir / "graphify_manifest.json").write_text("{}", encoding="utf-8")

    res = check_v2_completeness(v2_dir)
    assert res["status"] == "FAIL"
    assert "build tugallanmagan" in res["error"]
    assert "build_complete.json topilmadi" in res["error"]


def test_check_v2_completeness_identity_mismatch_fails(tmp_path):
    """T3 regression: model identity mismatch between build_complete.json and graphify_manifest.json FAILS."""
    v2_dir = tmp_path / "v2_mismatch"
    v2_dir.mkdir()
    manifest_id = {
        "schema_version": 2,
        "backend": "ollama",
        "model": "bge-m3",
        "model_digest": "sha256:manifest_digest",
        "dimension": 1024,
        "distance": "cosine",
        "normalization": False,
    }
    complete_id = dict(manifest_id, model="different-model")

    (v2_dir / "graphify_manifest.json").write_text(json.dumps(manifest_id), encoding="utf-8")
    (v2_dir / "build_complete.json").write_text(json.dumps(complete_id), encoding="utf-8")

    res = check_v2_completeness(v2_dir)
    assert res["status"] == "FAIL"
    assert "mos kelmadi" in res["error"]


def test_check_v2_completeness_record_count_mismatch_fails(tmp_path):
    """T3 regression: collection.count() != manifest.jsonl lines FAILS."""
    v2_dir = tmp_path / "v2_count_mismatch"
    v2_dir.mkdir()
    manifest_id = {
        "schema_version": 2,
        "backend": "ollama",
        "model": "bge-m3",
        "model_digest": "sha256:digest",
        "dimension": 1024,
        "distance": "cosine",
        "normalization": False,
    }
    (v2_dir / "graphify_manifest.json").write_text(json.dumps(manifest_id), encoding="utf-8")
    (v2_dir / "build_complete.json").write_text(json.dumps(manifest_id), encoding="utf-8")

    # 3 lines in manifest.jsonl
    manifest_lines = ['{"id": "doc1"}\n', '{"id": "doc2"}\n', '{"id": "doc3"}\n']
    (v2_dir / "manifest.jsonl").write_text("".join(manifest_lines), encoding="utf-8")

    # Mock collection with count=2 (mismatch!)
    mock_col = Mock()
    mock_col.count.return_value = 2
    mock_client = Mock()
    mock_client.get_collection.return_value = mock_col

    with patch("chromadb.PersistentClient", return_value=mock_client):
        res = check_v2_completeness(v2_dir)
        assert res["status"] == "FAIL"
        assert "teng emas" in res["error"]


def test_check_v2_completeness_success(tmp_path):
    """T3 regression: valid build_complete.json matching manifest and equal record counts PASSES."""
    v2_dir = tmp_path / "v2_ok"
    v2_dir.mkdir()
    manifest_id = {
        "schema_version": 2,
        "backend": "ollama",
        "model": "bge-m3",
        "model_digest": "sha256:digest",
        "dimension": 1024,
        "distance": "cosine",
        "normalization": False,
    }
    (v2_dir / "graphify_manifest.json").write_text(json.dumps(manifest_id), encoding="utf-8")
    (v2_dir / "build_complete.json").write_text(json.dumps(manifest_id), encoding="utf-8")

    manifest_lines = ['{"id": "doc1"}\n', '{"id": "doc2"}\n']
    (v2_dir / "manifest.jsonl").write_text("".join(manifest_lines), encoding="utf-8")

    mock_col = Mock()
    mock_col.count.return_value = 2
    mock_client = Mock()
    mock_client.get_collection.return_value = mock_col

    with patch("chromadb.PersistentClient", return_value=mock_client):
        res = check_v2_completeness(v2_dir)
        assert res["status"] == "PASS"
        assert res["count"] == 2


def test_print_checklist_none_digest_does_not_crash(monkeypatch):
    """T4 regression: None digest does not raise TypeError and prints '-'."""
    captured = StringIO()
    monkeypatch.setattr(sys, "stdout", captured)

    identity = {
        "status": "PASS",
        "model": "bge-m3",
        "backend": "local",
        "dimension": 1024,
        "digest": None,
    }
    ready = print_checklist(
        processes=[],
        drift={"status": "OK", "total_drift": 0, "total_v1": 5},
        identity=identity,
        launch_configs=[],
        v1_dir=Path("v1"),
        v2_dir=Path("v2"),
        completeness={"status": "PASS", "count": 5},
    )
    output = captured.getvalue()
    assert "- Digest: -" in output
    assert ready is True


def test_find_running_graphify_processes_fail_closed_empty_stdout(monkeypatch):
    """Finding 152 / T2 regression: empty stdout with rc=0 returns None (fail-closed, not [])."""
    monkeypatch.setattr(sys, "platform", "win32")
    mock_run = Mock()
    mock_run.returncode = 0
    mock_run.stdout = "   \n  "
    mock_run.stderr = ""

    with patch("subprocess.run", return_value=mock_run):
        procs = find_running_graphify_processes()
        assert procs is None


def test_find_running_graphify_processes_fail_closed_null_json(monkeypatch):
    """Finding 153 / T2 regression: 'null' JSON from PowerShell returns None."""
    monkeypatch.setattr(sys, "platform", "win32")
    mock_run = Mock()
    mock_run.returncode = 0
    mock_run.stdout = "null"

    with patch("subprocess.run", return_value=mock_run):
        procs = find_running_graphify_processes()
        assert procs is None


def test_find_running_graphify_processes_fail_closed_non_dict_element(monkeypatch):
    """Finding 153 / T2 regression: non-dict element e.g. [1, 2] returns None."""
    monkeypatch.setattr(sys, "platform", "win32")
    mock_run = Mock()
    mock_run.returncode = 0
    mock_run.stdout = "[1, 2]"

    with patch("subprocess.run", return_value=mock_run):
        procs = find_running_graphify_processes()
        assert procs is None


def test_find_running_graphify_processes_valid_empty_array_returns_empty_list(monkeypatch):
    """Finding 152 / T2 regression: '[]' JSON from PowerShell returns [] (success, 0 processes)."""
    monkeypatch.setattr(sys, "platform", "win32")
    mock_run = Mock()
    mock_run.returncode = 0
    mock_run.stdout = "[]"

    with patch("subprocess.run", return_value=mock_run):
        procs = find_running_graphify_processes()
        assert procs == []


def test_check_v2_completeness_manifest_list_format_fails(tmp_path):
    """Finding 154 / T3 regression: graphify_manifest.json containing [] fails gracefully with FAIL status."""
    v2_dir = tmp_path / "v2_list_manifest"
    v2_dir.mkdir()
    (v2_dir / "build_complete.json").write_text("{}", encoding="utf-8")
    (v2_dir / "graphify_manifest.json").write_text("[]", encoding="utf-8")

    res = check_v2_completeness(v2_dir)
    assert res["status"] == "FAIL"
    assert "formati noto'g'ri" in res["error"]


def test_check_v2_completeness_missing_required_identity_keys_fails(tmp_path):
    """Finding 151 / T3 regression: build_complete.json missing required identity fields FAILS."""
    v2_dir = tmp_path / "v2_missing_keys"
    v2_dir.mkdir()
    manifest_id = {
        "schema_version": 2,
        "backend": "ollama",
        "model": "bge-m3",
        "model_digest": "sha256:digest",
        "dimension": 1024,
        "distance": "cosine",
        "normalization": False,
    }
    (v2_dir / "graphify_manifest.json").write_text(json.dumps(manifest_id), encoding="utf-8")
    (v2_dir / "build_complete.json").write_text(json.dumps({"model": "bge-m3"}), encoding="utf-8")

    res = check_v2_completeness(v2_dir)
    assert res["status"] == "FAIL"
    assert "majburiy identity maydoni yetishmaydi" in res["error"]


def test_check_v2_completeness_read_only_isolated_copy(tmp_path):
    """Finding 150 / T3 regression: collection.count() runs on isolated copy and does not mutate v2_dir."""
    v2_dir = tmp_path / "v2_empty_check"
    v2_dir.mkdir()
    manifest_id = {
        "schema_version": 2,
        "backend": "ollama",
        "model": "bge-m3",
        "model_digest": "sha256:digest",
        "dimension": 1024,
        "distance": "cosine",
        "normalization": False,
    }
    (v2_dir / "graphify_manifest.json").write_text(json.dumps(manifest_id), encoding="utf-8")
    (v2_dir / "build_complete.json").write_text(json.dumps(manifest_id), encoding="utf-8")
    (v2_dir / "manifest.jsonl").write_text('{"id": "doc1"}\n', encoding="utf-8")

    assert not (v2_dir / "chroma.sqlite3").exists()

    res = check_v2_completeness(v2_dir)
    assert res["status"] == "FAIL"
    assert not (v2_dir / "chroma.sqlite3").exists()


def test_conftest_checkout_kg_db_never_created_regression():
    """Finding 155 / T1 regression: verifies checkout kg_db was not created and is guarded by conftest sessionfinish."""
    checkout_kg_db = Path(__file__).resolve().parents[1] / "kg_db"
    assert not checkout_kg_db.exists()


def test_check_v2_completeness_closes_client_after_count(tmp_path):
    """T1: chromadb.PersistentClient is closed after col.count() before tempdir cleanup."""
    v2_dir = tmp_path / "v2_close_test"
    v2_dir.mkdir()
    manifest_id = {
        "schema_version": 2,
        "backend": "ollama",
        "model": "bge-m3",
        "model_digest": "sha256:digest",
        "dimension": 1024,
        "distance": "cosine",
        "normalization": False,
    }
    (v2_dir / "graphify_manifest.json").write_text(json.dumps(manifest_id), encoding="utf-8")
    (v2_dir / "build_complete.json").write_text(json.dumps(manifest_id), encoding="utf-8")
    (v2_dir / "manifest.jsonl").write_text('{"id": "doc1"}\n{"id": "doc2"}\n', encoding="utf-8")

    # 1. Test client.close() called after count()
    call_order = []
    mock_col = Mock()
    mock_col.count.side_effect = lambda: (call_order.append("count"), 2)[1]
    mock_client = Mock()
    mock_client.get_collection.return_value = mock_col
    mock_client.close.side_effect = lambda: call_order.append("close")

    with patch("chromadb.PersistentClient", return_value=mock_client):
        res = check_v2_completeness(v2_dir)
        assert res["status"] == "PASS"
        assert res["count"] == 2
        assert mock_client.close.called
        assert call_order == ["count", "close"]

    # 2. Test fallback to client._system.stop() when client.close is absent
    call_order_fallback = []
    mock_col_fallback = Mock()
    mock_col_fallback.count.side_effect = lambda: (call_order_fallback.append("count"), 2)[1]
    mock_client_fallback = Mock(spec=["get_collection", "_system"])
    mock_client_fallback.get_collection.return_value = mock_col_fallback
    mock_client_fallback._system = Mock()
    mock_client_fallback._system.stop.side_effect = lambda: call_order_fallback.append("stop")

    with patch("chromadb.PersistentClient", return_value=mock_client_fallback):
        res_fb = check_v2_completeness(v2_dir)
        assert res_fb["status"] == "PASS"
        assert res_fb["count"] == 2
        assert mock_client_fallback._system.stop.called
        assert call_order_fallback == ["count", "stop"]


def test_check_v2_completeness_temporary_directory_cleanup_permission_error_passes(tmp_path):
    """T1: PermissionError during TemporaryDirectory cleanup does not fail a valid check."""
    v2_dir = tmp_path / "v2_cleanup_perm_err"
    v2_dir.mkdir()
    manifest_id = {
        "schema_version": 2,
        "backend": "ollama",
        "model": "bge-m3",
        "model_digest": "sha256:digest",
        "dimension": 1024,
        "distance": "cosine",
        "normalization": False,
    }
    (v2_dir / "graphify_manifest.json").write_text(json.dumps(manifest_id), encoding="utf-8")
    (v2_dir / "build_complete.json").write_text(json.dumps(manifest_id), encoding="utf-8")
    (v2_dir / "manifest.jsonl").write_text('{"id": "doc1"}\n{"id": "doc2"}\n', encoding="utf-8")

    mock_col = Mock()
    mock_col.count.return_value = 2
    mock_client = Mock()
    mock_client.get_collection.return_value = mock_col

    import tempfile
    orig_td = tempfile.TemporaryDirectory

    class FailingCleanupTempDir(orig_td):
        def __exit__(self, exc_type, exc_val, exc_tb):
            super().__exit__(exc_type, exc_val, exc_tb)
            raise PermissionError("[WinError 32] The process cannot access the file because it is being used by another process")

    with patch("chromadb.PersistentClient", return_value=mock_client):
        with patch("tempfile.TemporaryDirectory", FailingCleanupTempDir):
            res = check_v2_completeness(v2_dir)
            assert res["status"] == "PASS"
            assert res["count"] == 2


def test_check_v2_completeness_get_collection_error_remains_fail(tmp_path):
    """T1: get_collection failure remains FAIL and client is still closed in finally."""
    v2_dir = tmp_path / "v2_get_col_err"
    v2_dir.mkdir()
    manifest_id = {
        "schema_version": 2,
        "backend": "ollama",
        "model": "bge-m3",
        "model_digest": "sha256:digest",
        "dimension": 1024,
        "distance": "cosine",
        "normalization": False,
    }
    (v2_dir / "graphify_manifest.json").write_text(json.dumps(manifest_id), encoding="utf-8")
    (v2_dir / "build_complete.json").write_text(json.dumps(manifest_id), encoding="utf-8")
    (v2_dir / "manifest.jsonl").write_text('{"id": "doc1"}\n', encoding="utf-8")

    mock_client = Mock()
    mock_client.get_collection.side_effect = ValueError("Collection 'graphify' does not exist")

    with patch("chromadb.PersistentClient", return_value=mock_client):
        res = check_v2_completeness(v2_dir)
        assert res["status"] == "FAIL"
        assert "v2 kolleksiyasini ochishda xatolik" in res["error"]
        assert "Collection 'graphify' does not exist" in res["error"]
        assert mock_client.close.called


def test_check_v2_completeness_client_close_permission_error_fails(tmp_path):
    """T1 / Review Finding: PermissionError during client.close() returns FAIL and gc.collect() runs."""
    v2_dir = tmp_path / "v2_close_perm_err"
    v2_dir.mkdir()
    manifest_id = {
        "schema_version": 2,
        "backend": "ollama",
        "model": "bge-m3",
        "model_digest": "sha256:digest",
        "dimension": 1024,
        "distance": "cosine",
        "normalization": False,
    }
    (v2_dir / "graphify_manifest.json").write_text(json.dumps(manifest_id), encoding="utf-8")
    (v2_dir / "build_complete.json").write_text(json.dumps(manifest_id), encoding="utf-8")
    (v2_dir / "manifest.jsonl").write_text('{"id": "doc1"}\n', encoding="utf-8")

    mock_col = Mock()
    mock_col.count.return_value = 1
    mock_client = Mock()
    mock_client.get_collection.return_value = mock_col
    mock_client.close.side_effect = PermissionError("[WinError 32] client.close lock error")

    with patch("chromadb.PersistentClient", return_value=mock_client), \
         patch("gc.collect") as mock_gc:
        res = check_v2_completeness(v2_dir)
        assert res["status"] == "FAIL"
        assert "v2 kolleksiyasini ochishda xatolik" in res["error"]
        assert "client.close lock error" in res["error"]
        assert mock_client.close.called
        assert mock_gc.called


def test_check_v2_completeness_system_stop_permission_error_fails(tmp_path):
    """T1 / Review Finding: PermissionError during client._system.stop() returns FAIL and gc.collect() runs."""
    v2_dir = tmp_path / "v2_stop_perm_err"
    v2_dir.mkdir()
    manifest_id = {
        "schema_version": 2,
        "backend": "ollama",
        "model": "bge-m3",
        "model_digest": "sha256:digest",
        "dimension": 1024,
        "distance": "cosine",
        "normalization": False,
    }
    (v2_dir / "graphify_manifest.json").write_text(json.dumps(manifest_id), encoding="utf-8")
    (v2_dir / "build_complete.json").write_text(json.dumps(manifest_id), encoding="utf-8")
    (v2_dir / "manifest.jsonl").write_text('{"id": "doc1"}\n', encoding="utf-8")

    mock_col = Mock()
    mock_col.count.return_value = 1
    mock_client = Mock(spec=["get_collection", "_system"])
    mock_client.get_collection.return_value = mock_col
    mock_client._system = Mock()
    mock_client._system.stop.side_effect = PermissionError("[WinError 32] _system.stop lock error")

    with patch("chromadb.PersistentClient", return_value=mock_client), \
         patch("gc.collect") as mock_gc:
        res = check_v2_completeness(v2_dir)
        assert res["status"] == "FAIL"
        assert "v2 kolleksiyasini ochishda xatolik" in res["error"]
        assert "_system.stop lock error" in res["error"]
        assert mock_client._system.stop.called
        assert mock_gc.called


def test_check_v2_completeness_prefixed_keys_ollama_missing_digest_fails(tmp_path):
    """T2: graphify: prefixed keys for Ollama backend without model_digest in marker FAILS."""
    v2_dir = tmp_path / "v2_prefixed_missing_digest"
    v2_dir.mkdir()
    manifest_data = {
        "graphify:schema_version": 2,
        "graphify:backend": "ollama",
        "graphify:model": "bge-m3",
        "graphify:model_digest": "sha256:digest_123",
        "graphify:dimension": 1024,
        "graphify:distance": "cosine",
        "graphify:normalization": False,
    }
    complete_data = {
        "graphify:schema_version": 2,
        "graphify:backend": "ollama",
        "graphify:model": "bge-m3",
        "graphify:dimension": 1024,
        "graphify:distance": "cosine",
        "graphify:normalization": False,
    }
    (v2_dir / "graphify_manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")
    (v2_dir / "build_complete.json").write_text(json.dumps(complete_data), encoding="utf-8")

    res = check_v2_completeness(v2_dir)
    assert res["status"] == "FAIL"
    assert "majburiy identity maydoni yetishmaydi: 'model_digest'" in res["error"]


def test_check_v2_completeness_prefixed_keys_ollama_with_digest_passes(tmp_path):
    """T2: graphify: prefixed keys for Ollama backend with matching model_digest in marker PASSES."""
    v2_dir = tmp_path / "v2_prefixed_ok"
    v2_dir.mkdir()
    manifest_data = {
        "graphify:schema_version": 2,
        "graphify:backend": "ollama",
        "graphify:model": "bge-m3",
        "graphify:model_digest": "sha256:digest_123",
        "graphify:dimension": 1024,
        "graphify:distance": "cosine",
        "graphify:normalization": False,
    }
    complete_data = dict(manifest_data)
    (v2_dir / "graphify_manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")
    (v2_dir / "build_complete.json").write_text(json.dumps(complete_data), encoding="utf-8")
    (v2_dir / "manifest.jsonl").write_text('{"id": "doc1"}\n{"id": "doc2"}\n', encoding="utf-8")

    mock_col = Mock()
    mock_col.count.return_value = 2
    mock_client = Mock()
    mock_client.get_collection.return_value = mock_col

    with patch("chromadb.PersistentClient", return_value=mock_client):
        res = check_v2_completeness(v2_dir)
        assert res["status"] == "PASS"
        assert res["count"] == 2


def test_g1_switch_check_normalization_sample_nan_fails(tmp_path):
    """G1: check_v2_normalization returns FAIL when vector sample contains [nan, 0]."""
    v2_dir = tmp_path / "v2_nan_check"
    v2_dir.mkdir()
    (v2_dir / "chroma.sqlite3").write_text("", encoding="utf-8")
    manifest_data = {
        "schema_version": 2,
        "backend": "ollama",
        "model": "bge-m3",
        "dimension": 2,
        "distance": "cosine",
        "normalization": True,
    }
    (v2_dir / "graphify_manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")

    mock_col = Mock()
    mock_col.count.return_value = 1
    mock_col.get.return_value = {"embeddings": [[float("nan"), 0.0]]}
    mock_client = Mock()
    mock_client.get_collection.return_value = mock_col

    with patch("chromadb.PersistentClient", return_value=mock_client):
        res = check_v2_normalization(v2_dir)
        assert res["status"] == "FAIL"
        assert "Namunaviy yozuv normasi 1±1e-3 oralig'ida emas" in res.get("error", "")


def test_g1_switch_check_normalization_sample_inf_fails(tmp_path):
    """G1: check_v2_normalization returns FAIL when vector sample contains [inf, 0]."""
    v2_dir = tmp_path / "v2_inf_check"
    v2_dir.mkdir()
    (v2_dir / "chroma.sqlite3").write_text("", encoding="utf-8")
    manifest_data = {
        "schema_version": 2,
        "backend": "ollama",
        "model": "bge-m3",
        "dimension": 2,
        "distance": "cosine",
        "normalization": True,
    }
    (v2_dir / "graphify_manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")

    mock_col = Mock()
    mock_col.count.return_value = 1
    mock_col.get.return_value = {"embeddings": [[float("inf"), 0.0]]}
    mock_client = Mock()
    mock_client.get_collection.return_value = mock_col

    with patch("chromadb.PersistentClient", return_value=mock_client):
        res = check_v2_normalization(v2_dir)
        assert res["status"] == "FAIL"
        assert "Namunaviy yozuv normasi 1±1e-3 oralig'ida emas" in res.get("error", "")


def test_g1_switch_check_normalization_sample_unit_passes(tmp_path):
    """G1: check_v2_normalization returns PASS when vector sample has unit norm."""
    v2_dir = tmp_path / "v2_unit_check"
    v2_dir.mkdir()
    (v2_dir / "chroma.sqlite3").write_text("", encoding="utf-8")
    manifest_data = {
        "schema_version": 2,
        "backend": "ollama",
        "model": "bge-m3",
        "dimension": 2,
        "distance": "cosine",
        "normalization": True,
    }
    (v2_dir / "graphify_manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")

    mock_col = Mock()
    mock_col.count.return_value = 1
    mock_col.get.return_value = {"embeddings": [[1.0, 0.0]]}
    mock_client = Mock()
    mock_client.get_collection.return_value = mock_col

    with patch("chromadb.PersistentClient", return_value=mock_client):
        res = check_v2_normalization(v2_dir)
        assert res["status"] == "PASS"
        assert res["sample_count"] == 1




