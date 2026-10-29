"""Real-Chroma integration tests for Graphify v2 migration tooling.

Runs against real chromadb.PersistentClient in tmp_path with a deterministic dummy
embedding function (no Ollama, no network). Verifies full build, sync after a change,
switch_check identity check, configuration space preservation, and build_complete.json
handling on forced failure.
"""

import json
import os
import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import chromadb
from migrate_v2.build_v2 import (
    main,
    run_sync,
    EXPECTED_DIMENSION,
)
from migrate_v2.eval_v2 import get_effective_space
from migrate_v2.switch_check import check_v2_identity


class DeterministicDummyEmbeddingFunction:
    """Deterministic dummy embedding function without network or Ollama."""

    def __init__(self, dimension: int = EXPECTED_DIMENSION):
        self.dimension = dimension

    def _embed_single(self, text: str) -> list[float]:
        val = (sum(ord(c) for c in (text or "")) % 997 + 1) / 1000.0
        return [float((int(val * 1000) + i) % 100) / 100.0 for i in range(self.dimension)]

    def __call__(self, input):
        return [self._embed_single(t) for t in input]

    def encode(self, input):
        return self(input)

    def name(self):
        return "graphify-dummy-deterministic"

    def is_legacy(self) -> bool:
        return True


def create_v1_fixture(db_dir: Path, collection_name: str = "graphify", space: str = "cosine") -> None:
    """Creates a minimal real SQLite Chroma v1 database."""
    db_file = db_dir / "chroma.sqlite3"
    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()
    cur.execute("CREATE TABLE collections (id TEXT PRIMARY KEY, name TEXT, config_json_str TEXT);")
    cur.execute(
        "CREATE TABLE collection_metadata (collection_id TEXT, key TEXT, str_value TEXT, int_value INT, float_value REAL, bool_value INT);"
    )
    cur.execute("CREATE TABLE segments (id TEXT PRIMARY KEY, type TEXT, scope TEXT, collection TEXT);")
    cur.execute("CREATE TABLE embeddings (id INTEGER PRIMARY KEY, segment_id TEXT, embedding_id TEXT, seq_id BLOB);")
    cur.execute("CREATE TABLE embedding_fulltext_search_content (id INTEGER PRIMARY KEY, c0 TEXT);")
    cur.execute(
        "CREATE TABLE embedding_metadata (id INTEGER, key TEXT, string_value TEXT, int_value INT, float_value REAL, bool_value INT);"
    )

    cur.execute("INSERT INTO collections VALUES ('col-1', ?, '{\"sync_threshold\": 500}')", (collection_name,))
    cur.execute("INSERT INTO collection_metadata VALUES ('col-1', 'hnsw:space', ?, NULL, NULL, NULL)", (space,))
    cur.execute("INSERT INTO segments VALUES ('seg-1', 'meta', 'METADATA', 'col-1')")

    # Record 1: standard document
    cur.execute("INSERT INTO embeddings VALUES (1, 'seg-1', 'doc-1', X'01')")
    cur.execute("INSERT INTO embedding_fulltext_search_content VALUES (1, 'Birinchi test hujjati matni')")
    cur.execute("INSERT INTO embedding_metadata VALUES (1, 'title', 'Hujjat 1', NULL, NULL, NULL)")
    cur.execute("INSERT INTO embedding_metadata VALUES (1, 'project', 'default', NULL, NULL, NULL)")

    # Record 2: documentless record
    cur.execute("INSERT INTO embeddings VALUES (2, 'seg-1', 'doc-2', X'02')")
    cur.execute("INSERT INTO embedding_metadata VALUES (2, 'title', 'Hujjatsiz yozuv 2', NULL, NULL, NULL)")
    cur.execute("INSERT INTO embedding_metadata VALUES (2, 'project', 'default', NULL, NULL, NULL)")

    conn.commit()
    conn.close()


def add_v1_record(db_dir: Path, row_id: int, doc_id: str, text: str, title: str) -> None:
    """Adds a new record to the v1 database to test incremental sync."""
    db_file = db_dir / "chroma.sqlite3"
    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()
    cur.execute("INSERT INTO embeddings VALUES (?, 'seg-1', ?, X'03')", (row_id, doc_id))
    cur.execute("INSERT INTO embedding_fulltext_search_content VALUES (?, ?)", (row_id, text))
    cur.execute("INSERT INTO embedding_metadata VALUES (?, 'title', ?, NULL, NULL, NULL)", (row_id, title))
    cur.execute("INSERT INTO embedding_metadata VALUES (?, 'project', 'default', NULL, NULL, NULL)", (row_id,))
    conn.commit()
    conn.close()


def sync_keyword_index_from_real_col(out_dir: Path, timeout: float = None) -> None:
    """Builds keyword_index.sqlite3 from the real Chroma collection in out_dir."""
    from keyword_index import tokenize

    client = chromadb.PersistentClient(path=str(out_dir))
    col = client.get_collection("graphify")
    data = col.get(include=["documents", "metadatas"])

    kw_file = out_dir / "keyword_index.sqlite3"
    if kw_file.exists():
        kw_file.unlink()

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


def test_real_chroma_full_build_sync_and_switch_check_workflow(tmp_path):
    """Full workflow: build v1 into v2, sync after change, switch_check pass, space preserved."""
    v1_dir = tmp_path / "v1_source"
    v1_dir.mkdir()
    create_v1_fixture(v1_dir, space="cosine")
    v2_dir = tmp_path / "v2_dest"

    dummy_ef = DeterministicDummyEmbeddingFunction()
    fake_digest = "sha256:dummy_bge_m3_digest_abc123"

    # Step 1: Full build of small v1 fixture into v2 dir
    with patch("migrate_v2.build_v2.get_ollama_model_digest", return_value=fake_digest), \
         patch("embedders.get_ollama_model_digest", return_value=fake_digest), \
         patch("migrate_v2.build_v2.build_chroma_embedding_function", return_value=dummy_ef), \
         patch("embedders.build_chroma_embedding_function", return_value=dummy_ef), \
         patch("migrate_v2.build_v2.build_embedder", return_value=dummy_ef), \
         patch("embedders.build_identity_embedder", return_value=dummy_ef), \
         patch("migrate_v2.build_v2.rebuild_keyword_index_subprocess", side_effect=sync_keyword_index_from_real_col), \
         patch.object(sys, "argv", ["build_v2.py", "--v1-db", str(v1_dir), "--out", str(v2_dir), "--batch-size", "2"]):
        main()

    # Verify real Chroma collection was created and persisted
    client = chromadb.PersistentClient(path=str(v2_dir))
    col = client.get_collection("graphify")
    assert col.count() == 2

    # Collection keeps its space in configuration_json
    assert col.configuration_json["hnsw"]["space"] == "cosine"
    assert get_effective_space(col.configuration_json, col.metadata) == "cosine"

    # Identity fields are in metadata
    meta = col.metadata
    assert meta["schema_version"] == 2
    assert meta["backend"] == "ollama"
    assert meta["model"] == "bge-m3"
    assert meta["model_digest"] == fake_digest
    assert meta["dimension"] == 1024
    assert meta["distance"] == "cosine"
    assert meta["normalization"] is False
    assert not any(k.startswith("hnsw:") for k in meta.keys())

    # Markers exist
    assert (v2_dir / "build_complete.json").exists()
    assert (v2_dir / "graphify_manifest.json").exists()

    # Step 2: Incremental sync after change in v1
    add_v1_record(v1_dir, row_id=3, doc_id="doc-3", text="Uchinchi yangi hujjat matni", title="Hujjat 3")

    with patch("migrate_v2.build_v2.get_ollama_model_digest", return_value=fake_digest), \
         patch("embedders.get_ollama_model_digest", return_value=fake_digest), \
         patch("migrate_v2.build_v2.build_chroma_embedding_function", return_value=dummy_ef), \
         patch("embedders.build_chroma_embedding_function", return_value=dummy_ef), \
         patch("migrate_v2.build_v2.build_embedder", return_value=dummy_ef), \
         patch("embedders.build_identity_embedder", return_value=dummy_ef), \
         patch("migrate_v2.build_v2.rebuild_keyword_index_subprocess", side_effect=sync_keyword_index_from_real_col):
        run_sync(v1_dir, v2_dir, batch_size=2)

    # Re-fetch from real Chroma client and verify record count
    col = client.get_collection("graphify")
    assert col.count() == 3
    assert col.configuration_json["hnsw"]["space"] == "cosine"
    assert get_effective_space(col.configuration_json, col.metadata) == "cosine"
    assert (v2_dir / "build_complete.json").exists()

    # Step 3: Run switch_check — succeeds
    with patch("embedders.build_identity_embedder", return_value=dummy_ef):
        chk = check_v2_identity(v2_dir)
    assert chk["status"] == "PASS"
    assert chk["model"] == "bge-m3"
    assert chk["backend"] == "ollama"
    assert chk["dimension"] == 1024
    assert chk["digest"] == fake_digest


def test_real_chroma_build_complete_removed_on_forced_failure(tmp_path):
    """build_complete.json handling is correct on a forced failure (marker is removed and not left)."""
    v1_dir = tmp_path / "v1_fail"
    v1_dir.mkdir()
    create_v1_fixture(v1_dir, space="cosine")
    v2_dir = tmp_path / "v2_fail"

    dummy_ef = DeterministicDummyEmbeddingFunction()
    fake_digest = "sha256:dummy_bge_m3_digest_abc123"

    # Initial successful build
    with patch("migrate_v2.build_v2.get_ollama_model_digest", return_value=fake_digest), \
         patch("embedders.get_ollama_model_digest", return_value=fake_digest), \
         patch("migrate_v2.build_v2.build_chroma_embedding_function", return_value=dummy_ef), \
         patch("embedders.build_chroma_embedding_function", return_value=dummy_ef), \
         patch("migrate_v2.build_v2.build_embedder", return_value=dummy_ef), \
         patch("embedders.build_identity_embedder", return_value=dummy_ef), \
         patch("migrate_v2.build_v2.rebuild_keyword_index_subprocess", side_effect=sync_keyword_index_from_real_col), \
         patch.object(sys, "argv", ["build_v2.py", "--v1-db", str(v1_dir), "--out", str(v2_dir)]):
        main()

    complete_marker = v2_dir / "build_complete.json"
    assert complete_marker.exists()

    # Add record to trigger mutation during sync
    add_v1_record(v1_dir, row_id=3, doc_id="doc-fail", text="Xatolik matni", title="Fail")

    # Invalidate completion status and simulate forced failure during embedding
    with patch("migrate_v2.build_v2.get_ollama_model_digest", return_value=fake_digest), \
         patch("embedders.get_ollama_model_digest", return_value=fake_digest), \
         patch("migrate_v2.build_v2.build_chroma_embedding_function", return_value=dummy_ef), \
         patch("embedders.build_chroma_embedding_function", return_value=dummy_ef), \
         patch("migrate_v2.build_v2.build_embedder", return_value=dummy_ef), \
         patch("migrate_v2.build_v2.embed_text_with_retry", side_effect=RuntimeError("Forced embedding crash")):
        with pytest.raises(RuntimeError, match="Forced embedding crash"):
            run_sync(v1_dir, v2_dir)

    # build_complete.json must not exist after failure
    assert not complete_marker.exists()


def test_real_chroma_modify_strips_hnsw_keys_to_avoid_changing_distance_error(tmp_path):
    """Finding 1 on real Chroma: col.modify fails if hnsw:space is in dict, succeeds when stripped."""
    db_dir = tmp_path / "real_chroma_modify"
    db_dir.mkdir()
    client = chromadb.PersistentClient(path=str(db_dir))
    col = client.get_or_create_collection(
        name="graphify",
        metadata={"hnsw:space": "cosine", "initial_key": "val1"},
    )
    assert col.configuration_json["hnsw"]["space"] == "cosine"

    # With hnsw:space included (pre-fix build_v2.py:768), real Chroma raises ValueError:
    with pytest.raises(
        ValueError, match="Changing the distance function of a collection once it is created is not supported currently"
    ):
        col.modify(metadata={"hnsw:space": "cosine", "model": "bge-m3"})

    # With hnsw:* keys stripped (fixed in Finding 1), col.modify succeeds and preserves space:
    new_metadata = {
        k: v
        for k, v in {"hnsw:space": "cosine", "model": "bge-m3", "schema_version": 2}.items()
        if not k.startswith("hnsw:")
    }
    col.modify(metadata=new_metadata)
    assert col.configuration_json["hnsw"]["space"] == "cosine"
    assert col.metadata["model"] == "bge-m3"
    assert "hnsw:space" not in col.metadata
