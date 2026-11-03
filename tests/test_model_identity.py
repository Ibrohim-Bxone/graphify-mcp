"""Tests for model identity binding and fail-closed behavior (Finding 4).

All tests use temporary directories, mock Ollama / network, and never touch live databases.
"""

import json
import sqlite3
import sys
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import embedders
from embedders import (
    ModelIdentityError,
    build_identity_embedder,
    build_chroma_embedding_function,
    get_ollama_model_digest,
)
import db


@pytest.fixture
def mock_ollama_tags():
    fake_digest = "sha256:7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab"
    tags_response = {
        "models": [
            {
                "name": "bge-m3:latest",
                "model": "bge-m3:latest",
                "digest": fake_digest,
                "details": {"embedding_length": 1024},
            }
        ]
    }
    return fake_digest, tags_response


def test_get_ollama_model_digest_success(mock_ollama_tags):
    expected_digest, tags_data = mock_ollama_tags
    mock_resp = Mock()
    mock_resp.read.return_value = json.dumps(tags_data).encode("utf-8")
    mock_resp.__enter__ = Mock(return_value=mock_resp)
    mock_resp.__exit__ = Mock(return_value=None)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        digest = get_ollama_model_digest("http://localhost:11434", "bge-m3")
        assert digest == expected_digest


def test_get_ollama_model_digest_missing_server_raises():
    with patch("urllib.request.urlopen", side_effect=OSError("Connection refused")):
        with pytest.raises(ModelIdentityError, match="Ollama serveriga ulanib bo'lmadi"):
            get_ollama_model_digest("http://localhost:11434", "bge-m3")


def test_identity_embedder_builds_and_validates_successfully(mock_ollama_tags):
    digest, tags_data = mock_ollama_tags
    mock_resp = Mock()
    mock_resp.read.return_value = json.dumps(tags_data).encode("utf-8")
    mock_resp.__enter__ = Mock(return_value=mock_resp)
    mock_resp.__exit__ = Mock(return_value=None)

    identity = {
        "schema_version": 2,
        "backend": "ollama",
        "model": "bge-m3",
        "model_digest": digest,
        "dimension": 1024,
        "distance": "l2",
    }

    fake_vec = np.ones((1, 1024), dtype=np.float32)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with patch.object(embedders.OllamaEmbedder, "encode", return_value=fake_vec):
            embedder = build_identity_embedder(identity)
            assert embedder is not None
            fn = build_chroma_embedding_function(identity=identity)
            assert fn is not None
            assert fn.name() == "graphify-ollama"


def test_model_digest_mismatch_raises_model_identity_error(mock_ollama_tags):
    _, tags_data = mock_ollama_tags
    mock_resp = Mock()
    mock_resp.read.return_value = json.dumps(tags_data).encode("utf-8")
    mock_resp.__enter__ = Mock(return_value=mock_resp)
    mock_resp.__exit__ = Mock(return_value=None)

    identity = {
        "schema_version": 2,
        "backend": "ollama",
        "model": "bge-m3",
        "model_digest": "sha256:different_corrupted_digest_123456",
        "dimension": 1024,
    }

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with pytest.raises(ModelIdentityError, match="Model digest mos kelmadi"):
            build_identity_embedder(identity)


def test_dimension_mismatch_raises_model_identity_error(mock_ollama_tags):
    digest, tags_data = mock_ollama_tags
    mock_resp = Mock()
    mock_resp.read.return_value = json.dumps(tags_data).encode("utf-8")
    mock_resp.__enter__ = Mock(return_value=mock_resp)
    mock_resp.__exit__ = Mock(return_value=None)

    identity = {
        "schema_version": 2,
        "backend": "ollama",
        "model": "bge-m3",
        "model_digest": digest,
        "dimension": 1024,
    }

    # Model returns 384 dimensions instead of expected 1024
    wrong_dim_vec = np.ones((1, 384), dtype=np.float32)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with patch.object(embedders.OllamaEmbedder, "encode", return_value=wrong_dim_vec):
            with pytest.raises(ModelIdentityError, match="Model o'lchami mos kelmadi"):
                build_identity_embedder(identity)


def test_missing_ollama_server_raises_and_never_falls_back():
    identity = {
        "schema_version": 2,
        "backend": "ollama",
        "model": "bge-m3",
        "model_digest": "sha256:some_digest",
        "dimension": 1024,
    }

    with patch("urllib.request.urlopen", side_effect=OSError("Connection refused")):
        with pytest.raises(ModelIdentityError):
            build_chroma_embedding_function(identity=identity)


def test_failed_self_test_raises_model_identity_error(mock_ollama_tags):
    digest, tags_data = mock_ollama_tags
    mock_resp = Mock()
    mock_resp.read.return_value = json.dumps(tags_data).encode("utf-8")
    mock_resp.__enter__ = Mock(return_value=mock_resp)
    mock_resp.__exit__ = Mock(return_value=None)

    identity = {
        "schema_version": 2,
        "backend": "ollama",
        "model": "bge-m3",
        "model_digest": digest,
        "dimension": 1024,
    }

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with patch.object(embedders.OllamaEmbedder, "encode", side_effect=RuntimeError("GPU OOM")):
            with pytest.raises(ModelIdentityError, match="Embedder 1-query self-test xatoligi"):
                build_identity_embedder(identity)


def test_db_without_identity_metadata_preserves_fallback_behavior(tmp_path, monkeypatch):
    """v1 database with no identity metadata keeps existing behavior."""
    v1_dir = tmp_path / "v1_db"
    v1_dir.mkdir()
    monkeypatch.setattr(db, "DB_PATH", str(v1_dir))

    client = Mock()
    # Simulate ValueError when custom embedder fails in get_or_create_collection
    client.get_or_create_collection.side_effect = [
        ValueError("collection already exists with different settings"),
        Mock(name="fallback_collection"),
    ]

    with patch.object(db, "build_chroma_embedding_function", return_value=Mock()):
        col = db._open_collection(client)
        assert col is not None
        # Assert that client.get_or_create_collection was called second time with fallback metadata
        assert client.get_or_create_collection.call_count == 2
        second_call = client.get_or_create_collection.call_args_list[1]
        assert second_call[1]["metadata"] == {"hnsw:space": "cosine"}


def test_db_with_identity_fails_closed_in_db_open_collection(tmp_path, monkeypatch):
    """v2 database with identity metadata fails closed: raises and NEVER falls back."""
    v2_dir = tmp_path / "v2_db"
    v2_dir.mkdir()
    manifest_file = v2_dir / "graphify_manifest.json"
    manifest_data = {
        "schema_version": 2,
        "backend": "ollama",
        "model": "bge-m3",
        "model_digest": "sha256:test_digest",
        "dimension": 1024,
    }
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")
    monkeypatch.setattr(db, "DB_PATH", str(v2_dir))

    client = Mock()

    # Simulate Ollama down
    with patch("urllib.request.urlopen", side_effect=OSError("Connection refused")):
        with pytest.raises(ModelIdentityError):
            db._open_collection(client)
        # Verify that get_or_create_collection was NEVER called with a fallback!
        client.get_or_create_collection.assert_not_called()


def test_identity_missing_mandatory_fields_raises_model_identity_error():
    """Finding 7: Identity requires mandatory fields (model, digest, dimension) and raises ModelIdentityError if missing."""
    # Missing model_digest for Ollama
    id_no_digest = {
        "backend": "ollama",
        "model": "bge-m3",
        "dimension": 1024,
    }
    with pytest.raises(ModelIdentityError, match="model_digest ko'rsatilmagan"):
        build_identity_embedder(id_no_digest)

    # Missing dimension
    id_no_dim = {
        "backend": "ollama",
        "model": "bge-m3",
        "model_digest": "sha256:fake",
    }
    with pytest.raises(ModelIdentityError, match="dimension ko'rsatilmagan"):
        build_identity_embedder(id_no_dim)

    # Missing model
    id_no_model = {
        "backend": "ollama",
        "dimension": 1024,
        "model_digest": "sha256:fake",
    }
    with pytest.raises(ModelIdentityError, match="model nomi ko'rsatilmagan"):
        build_identity_embedder(id_no_model)


def test_get_ollama_model_digest_normalizes_and_rejects_other_tags():
    """Finding 8: Tags like bge-m3:other listed first do not match when bge-m3 is requested; matches bge-m3:latest."""
    mock_resp_data = {
        "models": [
            {
                "name": "bge-m3:other",
                "model": "bge-m3:other",
                "digest": "sha256:wrong_other_digest",
            },
            {
                "name": "bge-m3:latest",
                "model": "bge-m3:latest",
                "digest": "sha256:correct_latest_digest",
            },
        ]
    }
    mock_resp = Mock()
    mock_resp.read.return_value = json.dumps(mock_resp_data).encode("utf-8")
    mock_resp.__enter__ = Mock(return_value=mock_resp)
    mock_resp.__exit__ = Mock(return_value=None)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        # Requesting "bge-m3" normalizes to "bge-m3:latest"
        digest = get_ollama_model_digest("http://localhost:11434", "bge-m3")
        assert digest == "sha256:correct_latest_digest"


def test_get_db_identity_raises_on_manifest_collection_conflict(tmp_path, monkeypatch):
    """Finding 9: Manifest and collection identity conflict raises ModelIdentityError."""
    db_dir = tmp_path / "conflict_db"
    db_dir.mkdir()
    manifest_file = db_dir / "graphify_manifest.json"
    manifest_data = {
        "schema_version": 2,
        "backend": "ollama",
        "model": "bge-m3",
        "model_digest": "sha256:manifest_digest",
        "dimension": 1024,
    }
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    mock_client = Mock()
    mock_col = Mock()
    mock_col.metadata = {
        "schema_version": 2,
        "backend": "ollama",
        "model": "bge-m3",
        "model_digest": "sha256:conflicting_collection_digest",
        "dimension": 1024,
    }
    mock_client.get_collection.return_value = mock_col

    with pytest.raises(ModelIdentityError, match="Model identifikatsiyasi nomuvofiq"):
        db.get_db_identity(client=mock_client, db_path=str(db_dir))


def test_align_chroma_collection_configuration_removes_default(tmp_path):
    """Finding 1: Align Chroma collection configuration replaces default EF with legacy to prevent conflict."""
    import sqlite3
    db_file = tmp_path / "chroma.sqlite3"
    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()
    cur.execute("CREATE TABLE collections (id TEXT PRIMARY KEY, name TEXT, schema_str TEXT);")
    schema_with_default = json.dumps({
        "keys": {
            "#embedding": {
                "float_list": {
                    "vector_index": {
                        "embedding_function": {
                            "type": "known",
                            "name": "default",
                            "config": {},
                        }
                    }
                }
            }
        }
    })
    cur.execute("INSERT INTO collections VALUES ('col-1', 'graphify', ?)", (schema_with_default,))
    conn.commit()
    conn.close()

    db._align_chroma_collection_configuration(tmp_path, "graphify")

    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()
    cur.execute("SELECT schema_str FROM collections WHERE name = 'graphify'")
    updated_schema_str = cur.fetchone()[0]
    conn.close()

    updated = json.loads(updated_schema_str)
    ef = updated["keys"]["#embedding"]["float_list"]["vector_index"]["embedding_function"]
    assert ef == {"type": "legacy"}
    assert "default" not in updated_schema_str

def test_get_db_identity_propagates_read_errors_and_never_falls_back():
    """Finding 3: Read errors during identity lookup must be propagated, never swallowed into legacy fallback."""
    mock_client = Mock()
    mock_client.get_collection.side_effect = RuntimeError("Disk I/O error reading SQLite")

    with pytest.raises(RuntimeError, match="Disk I/O error"):
        db.get_db_identity(client=mock_client)

    # When opening collection with a read error, it propagates out and never calls get_or_create_collection
    with pytest.raises(RuntimeError, match="Disk I/O error"):
        db._open_collection(mock_client)
    mock_client.get_or_create_collection.assert_not_called()

    # Confirmed absent collection does NOT raise and takes legacy path (returns None)
    import chromadb.errors
    absent_client = Mock()
    absent_client.get_collection.side_effect = chromadb.errors.NotFoundError("Collection graphify does not exist")
    assert db.get_db_identity(client=absent_client) is None


def test_build_identity_embedder_rejects_unsupported_local_model():
    """Finding 4: Local identity with an unsupported model name must be rejected."""
    bad_id = {
        "schema_version": 2,
        "backend": "local",
        "model": "unsupported-transformer-model",
        "dimension": 384,
    }
    with pytest.raises(ModelIdentityError, match="Qo'llab-quvvatlanmagan lokal model"):
        build_identity_embedder(bad_id)

    # Supported local model name succeeds (with mocked encode)
    good_id = {
        "schema_version": 2,
        "backend": "local",
        "model": "all-MiniLM-L6-v2",
        "dimension": 384,
    }
    with patch.object(embedders.LocalEmbedder, "encode", return_value=np.ones((1, 384), dtype=np.float32)):
        embedder = build_identity_embedder(good_id)
        assert embedder is not None

def test_get_db_identity_propagates_tenant_not_found_errors():
    """Finding 3 regression (TOPILMA 1): RuntimeError('tenant not found') must be propagated, never swallowed into None."""
    mock_client = Mock()
    mock_client.get_collection.side_effect = RuntimeError("tenant not found")

    with pytest.raises(RuntimeError, match="tenant not found"):
        db.get_db_identity(client=mock_client)

    with pytest.raises(RuntimeError, match="tenant not found"):
        db._open_collection(mock_client)
    mock_client.get_or_create_collection.assert_not_called()

    # Also test NotFoundError containing tenant
    import chromadb.errors
    tenant_err_client = Mock()
    tenant_err_client.get_collection.side_effect = chromadb.errors.NotFoundError("tenant default_tenant not found")

    with pytest.raises(chromadb.errors.NotFoundError, match="tenant default_tenant not found"):
        db.get_db_identity(client=tenant_err_client)
def test_build_identity_embedder_local_accepts_canonical_variants_and_rejects_others():
    """Finding 4 regression: Canonical local model names succeed while any other name raises ModelIdentityError."""
    supported = ["all-MiniLM-L6-v2", "sentence-transformers/all-minilm-l6-v2", "ALL-MINILM-L6-V2 "]
    for model_name in supported:
        identity = {
            "schema_version": 2,
            "backend": "local",
            "model": model_name,
            "dimension": 384,
        }
        with patch.object(embedders.LocalEmbedder, "encode", return_value=np.ones((1, 384), dtype=np.float32)):
            emb = build_identity_embedder(identity)
            assert emb is not None

    unsupported = ["bert-base-uncased", "text-embedding-ada-002", "nomic-embed-text"]
    for model_name in unsupported:
        bad_id = {
            "schema_version": 2,
            "backend": "local",
            "model": model_name,
            "dimension": 384,
        }
        with pytest.raises(ModelIdentityError, match="Qo'llab-quvvatlanmagan lokal model"):
            build_identity_embedder(bad_id)


def test_get_db_identity_raises_when_collections_table_has_graphify_but_metadata_table_missing_regression_finding_6(tmp_path):
    """Finding 6 regression: only confirmed absent collection bypasses identity lookup; missing collection_metadata raises OperationalError when collection exists."""
    db_file = tmp_path / "chroma.sqlite3"
    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()
    # collections table exists and has graphify collection, but collection_metadata table is missing
    cur.execute("CREATE TABLE collections (id TEXT PRIMARY KEY, name TEXT);")
    cur.execute("INSERT INTO collections VALUES ('col-1', 'graphify');")
    conn.commit()
    conn.close()

    # Must NOT swallow into None! Must raise OperationalError because collection exists!
    with pytest.raises(sqlite3.OperationalError, match="no such table: collection_metadata"):
        db.get_db_identity(db_path=str(tmp_path))

    # Conversely, if collections has no graphify collection, it is confirmed absent -> returns None
    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()
    cur.execute("DELETE FROM collections WHERE name = 'graphify';")
    cur.execute("INSERT INTO collections VALUES ('col-other', 'other_collection');")
    conn.commit()
    conn.close()

    res = db.get_db_identity(db_path=str(tmp_path))
    assert res is None


def test_identity_fields_validated_independently_dimension_float_fails(tmp_path):
    """Regression test (Finding 3): validate every present identity field independently (dimension 1024.9 must fail)."""
    # 1. Manifest with float dimension fails independently (even without collection or graphify prefix)
    m_dir = tmp_path / "float_dim_manifest"
    m_dir.mkdir()
    manifest_data = {
        "schema_version": 2,
        "backend": "ollama",
        "model": "bge-m3",
        "model_digest": "sha256:abc",
        "dimension": 1024.9,
    }
    (m_dir / "graphify_manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")

    with pytest.raises(ModelIdentityError, match="dimension"):
        db.get_db_identity(db_path=str(m_dir))

    # 2. Prefixed graphify:dimension float fails independently
    m_dir2 = tmp_path / "float_dim_prefixed"
    m_dir2.mkdir()
    manifest_data2 = {
        "graphify:schema_version": 2,
        "graphify:backend": "ollama",
        "graphify:model": "bge-m3",
        "graphify:model_digest": "sha256:abc",
        "graphify:dimension": 1024.9,
    }
    (m_dir2 / "graphify_manifest.json").write_text(json.dumps(manifest_data2), encoding="utf-8")

    with pytest.raises(ModelIdentityError, match="dimension"):
        db.get_db_identity(db_path=str(m_dir2))

    # 3. Collection metadata with float dimension fails independently
    c_client = Mock()
    c_col = Mock()
    c_col.metadata = {
        "schema_version": 2,
        "backend": "ollama",
        "model": "bge-m3",
        "model_digest": "sha256:abc",
        "dimension": 1024.9,
    }
    c_client.get_collection.return_value = c_col

    with pytest.raises(ModelIdentityError, match="dimension"):
        db.get_db_identity(client=c_client, db_path=str(tmp_path / "nonexistent"))


def test_http_storage_mode_checks_plain_vs_prefixed_conflict_before_early_return(monkeypatch):
    """Regression test (Finding 4): plain-vs-graphify: conflict check runs before HTTP early return in db.py."""
    monkeypatch.setattr(db, "STORAGE_MODE", "http")

    mock_client = Mock()
    mock_col = Mock()
    # Conflicting model names
    mock_col.metadata = {
        "schema_version": 2,
        "model": "bge-m3",
        "graphify:model": "all-MiniLM-L6-v2",
        "model_digest": "sha256:abc",
        "dimension": 1024,
    }
    mock_client.get_collection.return_value = mock_col

    with pytest.raises(ModelIdentityError, match="Model identifikatsiyasi nomuvofiq"):
        db.get_db_identity(client=mock_client)

    # Conflicting dimension
    mock_col.metadata = {
        "schema_version": 2,
        "model": "bge-m3",
        "model_digest": "sha256:abc",
        "dimension": 1024,
        "graphify:dimension": 384,
    }
    with pytest.raises(ModelIdentityError, match="dimension"):
        db.get_db_identity(client=mock_client)

    # Invalid field in HTTP mode (dimension float)
    mock_col.metadata = {
        "schema_version": 2,
        "model": "bge-m3",
        "model_digest": "sha256:abc",
        "dimension": 1024.9,
    }
    with pytest.raises(ModelIdentityError, match="dimension"):
        db.get_db_identity(client=mock_client)


def test_http_storage_mode_validates_raw_c_meta_independently_before_early_returns(monkeypatch):
    """Regression test: HTTP storage mode validates raw c_meta fields independently before early returns."""
    monkeypatch.setattr(db, "STORAGE_MODE", "http")

    mock_client = Mock()
    mock_col = Mock()
    mock_client.get_collection.return_value = mock_col

    # 1. schema_version as float 2.9 must raise ModelIdentityError, not return None
    mock_col.metadata = {"schema_version": 2.9}
    with pytest.raises(ModelIdentityError, match="schema_version"):
        db.get_db_identity(client=mock_client)

    # 2. normalization as 'invalid' must raise ModelIdentityError, not return None
    mock_col.metadata = {"normalization": "invalid"}
    with pytest.raises(ModelIdentityError, match="normalization"):
        db.get_db_identity(client=mock_client)

    # 3. graphify:normalization as 'invalid' must raise ModelIdentityError, not return None
    mock_col.metadata = {"graphify:normalization": "invalid"}
    with pytest.raises(ModelIdentityError, match="normalization"):
        db.get_db_identity(client=mock_client)


def test_local_sources_validate_raw_metadata_independently(tmp_path):
    """Regression test: Local manifest and collection metadata validate raw fields independently."""
    # 1. Manifest with normalization 'invalid' raises ModelIdentityError
    m_dir = tmp_path / "bad_norm_manifest"
    m_dir.mkdir()
    (m_dir / "graphify_manifest.json").write_text(json.dumps({"normalization": "invalid"}), encoding="utf-8")
    with pytest.raises(ModelIdentityError, match="normalization"):
        db.get_db_identity(db_path=str(m_dir))

    # 2. Manifest with schema_version 2.9 raises ModelIdentityError
    s_dir = tmp_path / "bad_schema_manifest"
    s_dir.mkdir()
    (s_dir / "graphify_manifest.json").write_text(json.dumps({"schema_version": 2.9}), encoding="utf-8")
    with pytest.raises(ModelIdentityError, match="schema_version"):
        db.get_db_identity(db_path=str(s_dir))

    # 3. Local client collection with normalization 'invalid' raises ModelIdentityError
    c_client = Mock()
    c_col = Mock()
    c_col.metadata = {"normalization": "invalid"}
    c_client.get_collection.return_value = c_col
    with pytest.raises(ModelIdentityError, match="normalization"):
        db.get_db_identity(client=c_client, db_path=str(tmp_path / "nonexistent"))

    # 4. Local client collection with schema_version 2.9 raises ModelIdentityError
    c_col.metadata = {"schema_version": 2.9}
    with pytest.raises(ModelIdentityError, match="schema_version"):
        db.get_db_identity(client=c_client, db_path=str(tmp_path / "nonexistent"))



