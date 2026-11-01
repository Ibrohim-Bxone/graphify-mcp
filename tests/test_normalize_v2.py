"""Tests for Graphify v2 embedding normalization, renormalize mode, and verification (E1-E5)."""

import hashlib
import json
import os
import sqlite3
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import chromadb
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import db
import embedders
from embedders import (
    ModelIdentityError,
    OllamaEmbedder,
    build_identity_embedder,
)
from migrate_v2.build_v2 import run_renormalize, run_sync
from migrate_v2.switch_check import check_v2_normalization


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _dir_hashes(directory: Path) -> dict[str, str]:
    """Compute sha256 hashes of all files in directory."""
    hashes = {}
    for root, _, files in os.walk(directory):
        for f in files:
            p = Path(root) / f
            hashes[str(p.relative_to(directory))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return hashes


def _create_mock_v2_with_embeddings(
    target_dir: Path,
    vectors: list[list[float]],
    normalization: bool = False,
    distance: str = "l2",
) -> None:
    """Create a mock v2 database with given embeddings, metadata, and manifest."""
    target_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(target_dir))
    meta = {
        "schema_version": 2,
        "backend": "ollama",
        "model": "bge-m3",
        "model_digest": "sha256:fake_digest_123",
        "dimension": len(vectors[0]),
        "distance": distance,
        "normalization": normalization,
        "hnsw:space": distance,
    }
    col = client.create_collection("graphify", metadata=meta)
    ids = [f"doc_{i}" for i in range(len(vectors))]
    docs = [f"Text content for document {i}" for i in range(len(vectors))]
    metadatas = [{"source": f"doc_{i}.md"} for i in range(len(vectors))]
    col.add(ids=ids, documents=docs, metadatas=metadatas, embeddings=vectors)

    manifest = dict(meta)
    del manifest["hnsw:space"]
    manifest["total_records"] = len(vectors)
    manifest["corpus_hash"] = "fake_corpus_hash"
    (target_dir / "graphify_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# E1: OllamaEmbedder normalization and zero vector handling
# ---------------------------------------------------------------------------
def test_e1_ollama_embedder_normalize_true_produces_unit_vectors():
    """E1: OllamaEmbedder(normalize=True) normalizes vectors to unit L2 length (1.0)."""
    raw_vec = [3.0, 4.0] + [0.0] * 1022  # norm = 5.0
    mock_resp = Mock()
    mock_resp.json.return_value = {"embedding": raw_vec}
    mock_resp.raise_for_status = Mock()

    with patch("requests.post", return_value=mock_resp) as mock_post:
        embedder = OllamaEmbedder(normalize=True)
        assert embedder.normalize is True
        result = embedder.encode(["test query"])

        mock_post.assert_called_once()
        assert "/api/embeddings" in mock_post.call_args[0][0]

        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 1024)
        norm = float(np.linalg.norm(result[0]))
        assert np.isclose(norm, 1.0, atol=1e-5)
        assert np.isclose(result[0][0], 0.6, atol=1e-5)
        assert np.isclose(result[0][1], 0.8, atol=1e-5)


def test_e1_ollama_embedder_normalize_false_preserves_raw_vectors():
    """E1: OllamaEmbedder(normalize=False) preserves original unnormalized vectors."""
    raw_vec = [3.0, 4.0] + [0.0] * 1022  # norm = 5.0
    mock_resp = Mock()
    mock_resp.json.return_value = {"embedding": raw_vec}
    mock_resp.raise_for_status = Mock()

    with patch("requests.post", return_value=mock_resp):
        embedder = OllamaEmbedder(normalize=False)
        assert embedder.normalize is False
        result = embedder.encode(["test query"])

        norm = float(np.linalg.norm(result[0]))
        assert np.isclose(norm, 5.0, atol=1e-5)
        assert np.isclose(result[0][0], 3.0, atol=1e-5)


def test_e1_ollama_embedder_zero_vector_raises_model_identity_error():
    """E1: Zero vector with normalize=True raises ModelIdentityError (no silent pass)."""
    zero_vec = [0.0] * 1024
    mock_resp = Mock()
    mock_resp.json.return_value = {"embedding": zero_vec}
    mock_resp.raise_for_status = Mock()

    with patch("requests.post", return_value=mock_resp):
        embedder = OllamaEmbedder(normalize=True)
        with pytest.raises(ModelIdentityError, match="Nol vektor normallashtirib bo'lmaydi"):
            embedder.encode(["test zero"])


# ---------------------------------------------------------------------------
# E2: Identity -> embedder flag and db.py fail-closed behavior
# ---------------------------------------------------------------------------
def test_e2_identity_true_sets_embedder_normalize_flag():
    """E2: Database identity with normalization=True creates OllamaEmbedder with normalize=True."""
    identity = {
        "schema_version": 2,
        "backend": "ollama",
        "model": "bge-m3",
        "model_digest": "sha256:digest_123",
        "dimension": 1024,
        "distance": "cosine",
        "normalization": True,
    }
    with patch.object(embedders, "get_ollama_model_digest", return_value="sha256:digest_123"), \
         patch.object(OllamaEmbedder, "encode", return_value=np.ones((1, 1024), dtype=np.float32)):
        embedder = build_identity_embedder(identity)
        assert isinstance(embedder, OllamaEmbedder)
        assert embedder.normalize is True


def test_e2_identity_false_sets_embedder_normalize_flag():
    """E2: Database identity with normalization=False creates OllamaEmbedder with normalize=False."""
    identity = {
        "schema_version": 2,
        "backend": "ollama",
        "model": "bge-m3",
        "model_digest": "sha256:digest_123",
        "dimension": 1024,
        "distance": "l2",
        "normalization": False,
    }
    with patch.object(embedders, "get_ollama_model_digest", return_value="sha256:digest_123"), \
         patch.object(OllamaEmbedder, "encode", return_value=np.ones((1, 1024), dtype=np.float32)):
        embedder = build_identity_embedder(identity)
        assert isinstance(embedder, OllamaEmbedder)
        assert embedder.normalize is False


def test_e2_db_open_collection_fails_closed_on_missing_or_invalid_normalization():
    """E2: db._open_collection fails closed if v2 identity lacks normalization or is non-bool."""
    # 1. Missing normalization in v2 identity
    client_missing = Mock()
    mock_col_missing = Mock()
    mock_col_missing.metadata = {
        "schema_version": 2,
        "backend": "ollama",
        "model": "bge-m3",
        "model_digest": "sha256:digest_123",
        "dimension": 1024,
    }
    client_missing.get_collection.return_value = mock_col_missing

    with patch.object(db, "STORAGE_MODE", "http"), \
         pytest.raises(ModelIdentityError, match="normalization"):
        db._open_collection(client_missing)

    # 2. Invalid non-bool normalization
    client_invalid = Mock()
    mock_col_invalid = Mock()
    mock_col_invalid.metadata = {
        "schema_version": 2,
        "backend": "ollama",
        "model": "bge-m3",
        "model_digest": "sha256:digest_123",
        "dimension": 1024,
        "normalization": "not_a_valid_boolean",
    }
    client_invalid.get_collection.return_value = mock_col_invalid

    with patch.object(db, "STORAGE_MODE", "http"), \
         pytest.raises(ModelIdentityError, match="normalization"):
        db._open_collection(client_invalid)


# ---------------------------------------------------------------------------
# E3: Renormalize mode
# ---------------------------------------------------------------------------
def test_e3_renormalize_full_workflow_without_calling_ollama(tmp_path):
    """E3: build_v2.py --renormalize copies collection, normalizes embeddings, preserves source hashes, and NEVER calls Ollama."""
    src_dir = tmp_path / "v2_src"
    out_dir = tmp_path / "v2_out"

    # Create 6 unnormalized vectors with norm ≈ 25.0 (similar to bge-m3 raw)
    vectors = []
    for i in range(6):
        vec = [(i + 1) * 1.0] * 1024
        vectors.append(vec)

    _create_mock_v2_with_embeddings(src_dir, vectors, normalization=False, distance="l2")

    # Record source hashes before renormalize
    before_src_hashes = _dir_hashes(src_dir)

    # Mock keyword rebuild subprocess to create keyword_index.sqlite3
    def _mock_rebuild_kw(target_dir):
        from keyword_index import tokenize
        client = chromadb.PersistentClient(path=str(target_dir))
        col = client.get_collection("graphify")
        data = col.get(include=["documents", "metadatas"])
        kw_file = Path(target_dir) / "keyword_index.sqlite3"
        conn = sqlite3.connect(str(kw_file))
        cur = conn.cursor()
        cur.execute("CREATE TABLE docs (id TEXT PRIMARY KEY, project TEXT, kind TEXT, parent_id TEXT, token_count INT, text TEXT);")
        cur.execute("CREATE VIRTUAL TABLE fts_idx USING fts5(id UNINDEXED, text);")
        for did, doc, meta in zip(data["ids"], data["documents"], data["metadatas"]):
            text = doc or ""
            meta = meta or {}
            project = meta.get("project", "")
            kind = meta.get("kind", "")
            parent_id = meta.get("parent_id", "")
            if not parent_id and "shortcode" in meta:
                parent_id = meta["shortcode"]
            if not parent_id and "source" in meta:
                parent_id = meta["source"]
            tokens = tokenize((meta.get("title") or "") + "\n" + text)
            cur.execute("INSERT INTO docs VALUES (?, ?, ?, ?, ?, ?)", (did, project, kind, parent_id, len(tokens), text))
            cur.execute("INSERT INTO fts_idx VALUES (?, ?)", (did, " ".join(tokens)))
        conn.commit()
        conn.close()

    # Mock Ollama network calls to FAIL if called
    with patch("requests.post", side_effect=AssertionError("Ollama requests.post was called!")), \
         patch("urllib.request.urlopen", side_effect=AssertionError("Ollama urlopen was called!")), \
         patch("migrate_v2.build_v2.rebuild_keyword_index_subprocess", side_effect=_mock_rebuild_kw):
        run_renormalize(src_dir, out_dir, batch_size=2)

    # 1. Source files hashes are 100% unchanged
    after_src_hashes = _dir_hashes(src_dir)
    assert before_src_hashes == after_src_hashes

    # 2. Output record count equals source record count
    out_client = chromadb.PersistentClient(path=str(out_dir))
    out_col = out_client.get_collection("graphify")
    assert out_col.count() == 6

    # 3. Output vectors equal source / norm (1e-5 precision) and have L2 norm == 1.0
    out_data = out_col.get(include=["embeddings"])
    for orig_vec, out_vec in zip(vectors, out_data["embeddings"]):
        v_orig = np.array(orig_vec, dtype=np.float32)
        v_out = np.array(out_vec, dtype=np.float32)
        norm_orig = float(np.linalg.norm(v_orig))
        expected_vec = v_orig / norm_orig

        assert np.allclose(v_out, expected_vec, atol=1e-5, rtol=1e-5)
        assert np.isclose(float(np.linalg.norm(v_out)), 1.0, atol=1e-3)

    # 4. Markers and manifest files exist and specify normalization: True and distance: cosine
    assert (out_dir / "build_complete.json").exists()
    assert (out_dir / "graphify_manifest.json").exists()
    assert (out_dir / "manifest.jsonl").exists()
    assert (out_dir / "keyword_index.sqlite3").exists()

    with open(out_dir / "graphify_manifest.json", "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    assert manifest_data["normalization"] is True
    assert manifest_data["distance"] == "cosine"

    with open(out_dir / "build_complete.json", "r", encoding="utf-8") as f:
        complete_data = json.load(f)
    assert complete_data["normalization"] is True
    assert complete_data["distance"] == "cosine"


def test_e3_renormalize_refuses_to_overwrite_or_use_same_dir(tmp_path):
    """E3: run_renormalize raises ValueError if out == src or out is a non-empty directory."""
    src_dir = tmp_path / "src"
    _create_mock_v2_with_embeddings(src_dir, [[1.0] * 1024], normalization=False)

    # out == src
    with pytest.raises(ValueError, match="bir xil bo'lishi mumkin emas"):
        run_renormalize(src_dir, src_dir)

    # out exists and is non-empty
    out_dir = tmp_path / "non_empty_out"
    out_dir.mkdir()
    (out_dir / "pre_existing.txt").write_text("already here", encoding="utf-8")

    with pytest.raises(ValueError, match="ustidan yozish taqiqlangan"):
        run_renormalize(src_dir, out_dir)


# ---------------------------------------------------------------------------
# E4: Sync compatibility with normalized destination
# ---------------------------------------------------------------------------
def test_e4_sync_with_normalized_destination_normalizes_new_vectors(tmp_path):
    """E4: --sync into a database with normalization=True normalizes newly added vectors."""
    v1_dir = tmp_path / "v1_db"
    v1_dir.mkdir()
    # Create SQLite v1 with 2 records
    v1_db_file = v1_dir / "chroma.sqlite3"
    conn = sqlite3.connect(str(v1_db_file))
    cur = conn.cursor()
    cur.execute("CREATE TABLE collections (id TEXT PRIMARY KEY, name TEXT, config_json_str TEXT);")
    cur.execute("CREATE TABLE collection_metadata (collection_id TEXT, key TEXT, str_value TEXT, int_value INT, float_value REAL, bool_value INT);")
    cur.execute("CREATE TABLE segments (id TEXT PRIMARY KEY, type TEXT, scope TEXT, collection TEXT);")
    cur.execute("CREATE TABLE embeddings (id INTEGER PRIMARY KEY, segment_id TEXT, embedding_id TEXT, seq_id BLOB);")
    cur.execute("CREATE TABLE embedding_metadata (id INTEGER, key TEXT, string_value TEXT, int_value INT, float_value REAL, bool_value INT);")
    cur.execute("CREATE TABLE embedding_fulltext_search_content (id INTEGER PRIMARY KEY, c0 TEXT);")

    cur.execute("INSERT INTO collections VALUES ('col1', 'graphify', '{}');")
    cur.execute("INSERT INTO collection_metadata VALUES ('col1', 'hnsw:space', 'cosine', NULL, NULL, NULL);")
    cur.execute("INSERT INTO segments VALUES ('seg1', 'meta', 'METADATA', 'col1');")

    # Record 1 (already in dest) and Record 2 (new in v1)
    cur.execute("INSERT INTO embeddings VALUES (1, 'seg1', 'doc_1', X'01');")
    cur.execute("INSERT INTO embeddings VALUES (2, 'seg1', 'doc_2', X'02');")
    cur.execute("INSERT INTO embedding_fulltext_search_content VALUES (1, 'Doc 1 content');")
    cur.execute("INSERT INTO embedding_fulltext_search_content VALUES (2, 'Doc 2 new content');")
    cur.execute("INSERT INTO embedding_metadata VALUES (1, 'source', 'doc_1.md', NULL, NULL, NULL);")
    cur.execute("INSERT INTO embedding_metadata VALUES (2, 'source', 'doc_2.md', NULL, NULL, NULL);")
    conn.commit()
    conn.close()

    # Destination v2 has record 1 and normalization=True
    v2_dir = tmp_path / "v2_db"
    unit_vec = [1.0 / np.sqrt(1024)] * 1024
    _create_mock_v2_with_embeddings(v2_dir, [unit_vec], normalization=True, distance="cosine")
    # Set doc id to match doc_1
    client = chromadb.PersistentClient(path=str(v2_dir))
    col = client.get_collection("graphify")
    col.delete(ids=["doc_0"])
    col.add(ids=["doc_1"], documents=["Doc 1 content"], metadatas=[{"source": "doc_1.md"}], embeddings=[unit_vec])

    # Raw unnormalized vector returned by embedder for doc_2 (norm = 10.0)
    raw_unnorm_vec = [10.0 / np.sqrt(1024)] * 1024
    fake_digest = "sha256:fake_digest_123"

    def _mock_rebuild_kw(target_dir):
        from keyword_index import tokenize
        client = chromadb.PersistentClient(path=str(target_dir))
        col = client.get_collection("graphify")
        data = col.get(include=["documents", "metadatas"])
        kw_file = Path(target_dir) / "keyword_index.sqlite3"
        conn = sqlite3.connect(str(kw_file))
        cur = conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS docs (id TEXT PRIMARY KEY, project TEXT, kind TEXT, parent_id TEXT, token_count INT, text TEXT);")
        cur.execute("CREATE VIRTUAL TABLE IF NOT EXISTS fts_idx USING fts5(id UNINDEXED, text);")
        cur.execute("DELETE FROM docs;")
        cur.execute("DELETE FROM fts_idx;")
        for did, doc, meta in zip(data["ids"], data["documents"], data["metadatas"]):
            text = doc or ""
            meta = meta or {}
            project = meta.get("project", "")
            kind = meta.get("kind", "")
            parent_id = meta.get("parent_id", "")
            if not parent_id and "shortcode" in meta:
                parent_id = meta["shortcode"]
            if not parent_id and "source" in meta:
                parent_id = meta["source"]
            tokens = tokenize((meta.get("title") or "") + "\n" + text)
            cur.execute("INSERT INTO docs VALUES (?, ?, ?, ?, ?, ?)", (did, project, kind, parent_id, len(tokens), text))
            cur.execute("INSERT INTO fts_idx VALUES (?, ?)", (did, " ".join(tokens)))
        conn.commit()
        conn.close()

    with patch("migrate_v2.build_v2.get_ollama_model_digest", return_value=fake_digest), \
         patch("embedders.get_ollama_model_digest", return_value=fake_digest), \
         patch("requests.post") as mock_req, \
         patch("migrate_v2.build_v2.rebuild_keyword_index_subprocess", side_effect=_mock_rebuild_kw):
        mock_req.return_value.json.return_value = {"embedding": raw_unnorm_vec}
        mock_req.return_value.raise_for_status = Mock()

        run_sync(v1_dir, v2_dir, batch_size=10)

    # Verify doc_2 was added and has normalized unit length (norm == 1.0)
    col = client.get_collection("graphify")
    assert col.count() == 2
    res = col.get(ids=["doc_2"], include=["embeddings"])
    added_emb = res["embeddings"][0]
    norm = float(np.linalg.norm(added_emb))
    assert np.isclose(norm, 1.0, atol=1e-3)

    # Manifest and build_complete maintain normalization: True
    with open(v2_dir / "graphify_manifest.json", "r", encoding="utf-8") as f:
        m = json.load(f)
    assert m["normalization"] is True

    with open(v2_dir / "build_complete.json", "r", encoding="utf-8") as f:
        c = json.load(f)
    assert c["normalization"] is True


# ---------------------------------------------------------------------------
# E5: switch_check normalization verification (FAIL / WARN / PASS)
# ---------------------------------------------------------------------------
def test_e5_switch_check_normalization_warn_on_false(tmp_path):
    """E5: switch_check returns WARN for database with normalization=False."""
    v2_dir = tmp_path / "v2_unnorm"
    raw_vec = [25.0 / np.sqrt(1024)] * 1024
    _create_mock_v2_with_embeddings(v2_dir, [raw_vec], normalization=False)

    chk = check_v2_normalization(v2_dir)
    assert chk["status"] == "WARN"
    assert "vektorlar normallashtirilmagan — reyting normaga og'adi" in chk["message"]


def test_e5_switch_check_normalization_pass_on_true_with_unit_vectors(tmp_path):
    """E5: switch_check returns PASS for database with normalization=True and unit vectors (norm 1.0)."""
    v2_dir = tmp_path / "v2_norm_pass"
    unit_vec = [1.0 / np.sqrt(1024)] * 1024
    _create_mock_v2_with_embeddings(v2_dir, [unit_vec] * 5, normalization=True, distance="cosine")

    chk = check_v2_normalization(v2_dir)
    assert chk["status"] == "PASS"
    assert chk["sample_count"] == 5


def test_e5_switch_check_normalization_fail_on_true_with_unnormalized_vectors(tmp_path):
    """E5: switch_check returns FAIL if normalization=True but sample vectors violate 1±1e-3."""
    v2_dir = tmp_path / "v2_norm_fail"
    unnorm_vec = [25.0 / np.sqrt(1024)] * 1024
    _create_mock_v2_with_embeddings(v2_dir, [unnorm_vec] * 3, normalization=True, distance="cosine")

    chk = check_v2_normalization(v2_dir)
    assert chk["status"] == "FAIL"
    assert "Namunaviy yozuv normasi 1±1e-3 oralig'ida emas" in chk["error"]


def test_a3_renormalize_preserves_real_persistent_client_source_hashes(tmp_path):
    """A3: renormalize copies source to temporary directory, leaving ALL real PersistentClient source files and hashes untouched."""
    src_dir = tmp_path / "real_src"
    out_dir = tmp_path / "real_out"

    # Create real source database with PersistentClient
    client = chromadb.PersistentClient(path=str(src_dir))
    meta = {
        "schema_version": 2,
        "backend": "ollama",
        "model": "bge-m3",
        "model_digest": "sha256:fake_digest_456",
        "dimension": 4,
        "distance": "l2",
        "normalization": False,
        "hnsw:space": "l2",
    }
    col = client.create_collection("graphify", metadata=meta)
    docs = ["Doc alpha", "Doc beta", "Doc gamma"]
    metas = [{"title": d, "project": "test", "kind": "doc"} for d in docs]
    vectors = [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
    ]
    col.add(ids=["doc_1", "doc_2", "doc_3"], documents=docs, metadatas=metas, embeddings=vectors)

    manifest = dict(meta)
    del manifest["hnsw:space"]
    manifest["total_records"] = 3
    manifest["corpus_hash"] = "fake_corpus_hash_3"
    (src_dir / "graphify_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if hasattr(client, "close"):
        client.close()

    # Record hashes of all files in src_dir
    before_hashes = _dir_hashes(src_dir)
    assert len(before_hashes) > 0

    def _mock_rebuild_kw(target_dir):
        from keyword_index import tokenize
        c = chromadb.PersistentClient(path=str(target_dir))
        k_col = c.get_collection("graphify")
        data = k_col.get(include=["documents", "metadatas"])
        kw_file = Path(target_dir) / "keyword_index.sqlite3"
        conn = sqlite3.connect(str(kw_file))
        cur = conn.cursor()
        cur.execute("CREATE TABLE docs (id TEXT PRIMARY KEY, project TEXT, kind TEXT, parent_id TEXT, token_count INT, text TEXT);")
        cur.execute("CREATE VIRTUAL TABLE fts_idx USING fts5(id UNINDEXED, text);")
        for did, doc, m_meta in zip(data["ids"], data["documents"], data["metadatas"]):
            text = doc or ""
            m_meta = m_meta or {}
            tokens = tokenize((m_meta.get("title") or "") + "\n" + text)
            cur.execute("INSERT INTO docs VALUES (?, ?, ?, ?, ?, ?)", (did, "test", "doc", "", len(tokens), text))
            cur.execute("INSERT INTO fts_idx VALUES (?, ?)", (did, " ".join(tokens)))
        conn.commit()
        conn.close()
        if hasattr(c, "close"):
            c.close()

    with patch("requests.post", side_effect=AssertionError("Ollama was called!")), \
         patch("urllib.request.urlopen", side_effect=AssertionError("Ollama urlopen was called!")), \
         patch("migrate_v2.build_v2.rebuild_keyword_index_subprocess", side_effect=_mock_rebuild_kw):
        run_renormalize(src_dir, out_dir, batch_size=2)

    # 1. Source files hashes are 100% unchanged
    after_hashes = _dir_hashes(src_dir)
    assert before_hashes == after_hashes, "Manba bazasidagi fayllar xeshi o'zgargan!"

    # 2. Output collection verified
    out_client = chromadb.PersistentClient(path=str(out_dir))
    out_col = out_client.get_collection("graphify")
    assert out_col.count() == 3
    out_data = out_col.get(include=["embeddings"])
    for orig_vec, out_vec in zip(vectors, out_data["embeddings"]):
        v_orig = np.array(orig_vec, dtype=np.float32)
        v_out = np.array(out_vec, dtype=np.float32)
        norm_orig = float(np.linalg.norm(v_orig))
        expected_vec = v_orig / norm_orig
        assert np.allclose(v_out, expected_vec, atol=1e-5, rtol=1e-5)
        assert np.isclose(float(np.linalg.norm(v_out)), 1.0, atol=1e-3)
    if hasattr(out_client, "close"):
        out_client.close()
