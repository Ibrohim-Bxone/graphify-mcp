"""Tests for model identity binding and fail-closed behavior (Finding 4).

All tests use temporary directories, mock Ollama / network, and never touch live databases.
"""

import json
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
