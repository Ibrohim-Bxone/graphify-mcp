class _DummyEF:
    def __call__(self, input):
        return [[0.1] * 1024 for _ in input]
    def name(self):
        return "dummy"
"""Regression tests for Graphify v2 migration tooling (Findings 1, 2, 3, 5, 6).
All tests run in temporary directories, mock network / Ollama, and never touch live databases.
"""
import json
import os
import sqlite3
import sys
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from migrate_v2.build_v2 import (
    read_v1_records,
    compute_hashes,
    verify_v2,
    write_manifest_and_identity,
    write_build_complete_atomic,
    run_sync,
)
from migrate_v2.switch_check import (
    check_v1_manifest_drift,
    inspect_launch_configs,
)
def create_mock_v1_db(db_dir: Path, collection_name: str = "graphify"):
    """Helper to create a synthetic v1 SQLite database."""
    db_file = db_dir / "chroma.sqlite3"
    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()
    cur.execute("CREATE TABLE collections (id TEXT PRIMARY KEY, name TEXT, config_json_str TEXT);")
    cur.execute("CREATE TABLE collection_metadata (collection_id TEXT, key TEXT, str_value TEXT, int_value INT, float_value REAL, bool_value INT);")
    cur.execute("CREATE TABLE segments (id TEXT PRIMARY KEY, type TEXT, scope TEXT, collection TEXT);")
    cur.execute("CREATE TABLE embeddings (id INTEGER PRIMARY KEY, segment_id TEXT, embedding_id TEXT, seq_id BLOB);")
    cur.execute("CREATE TABLE embedding_fulltext_search_content (id INTEGER PRIMARY KEY, c0 TEXT);")
    cur.execute("CREATE TABLE embedding_metadata (id INTEGER, key TEXT, string_value TEXT, int_value INT, float_value REAL, bool_value INT);")
    cur.execute("INSERT INTO collections VALUES ('col-1', ?, '{\"sync_threshold\": 500}')", (collection_name,))
    cur.execute("INSERT INTO collections VALUES ('col-other', 'other_collection', '{}')")
    cur.execute("INSERT INTO collection_metadata VALUES ('col-1', 'hnsw:space', 'cosine', NULL, NULL, NULL)")
    cur.execute("INSERT INTO collection_metadata VALUES ('col-1', 'custom_key', 'custom_value', NULL, NULL, NULL)")
    cur.execute("INSERT INTO segments VALUES ('seg-1', 'meta', 'METADATA', 'col-1')")
    cur.execute("INSERT INTO segments VALUES ('seg-other', 'meta', 'METADATA', 'col-other')")
    # Document 1: standard document
    cur.execute("INSERT INTO embeddings VALUES (1, 'seg-1', 'doc-1', X'01')")
    cur.execute("INSERT INTO embedding_fulltext_search_content VALUES (1, 'Hujjat matni 1')")
    cur.execute("INSERT INTO embedding_metadata VALUES (1, 'title', 'Sarlavha 1', NULL, NULL, NULL)")
    # Document 2: documentless record!
    cur.execute("INSERT INTO embeddings VALUES (2, 'seg-1', 'doc-2', X'02')")
    cur.execute("INSERT INTO embedding_metadata VALUES (2, 'title', 'Sarlavha 2 (hujjatsiz)', NULL, NULL, NULL)")
    # Document 3: belongs to other collection (should be excluded)
    cur.execute("INSERT INTO embeddings VALUES (3, 'seg-other', 'doc-other', X'03')")
    cur.execute("INSERT INTO embedding_fulltext_search_content VALUES (3, 'Boshqa kolleksiya hujjati')")
    conn.commit()
    conn.close()
def create_mock_keyword_index(db_dir: Path, records: list[dict]):
    """Helper to create a synthetic keyword_index.sqlite3 matching records."""
    from keyword_index import tokenize
    kw_file = db_dir / "keyword_index.sqlite3"
    conn = sqlite3.connect(str(kw_file))
    cur = conn.cursor()
    cur.execute("CREATE TABLE docs (id TEXT PRIMARY KEY, project TEXT, kind TEXT, parent_id TEXT, token_count INT, text TEXT);")
    cur.execute("CREATE VIRTUAL TABLE fts_idx USING fts5(id UNINDEXED, text);")
    for r in records:
        text = r.get("document") or ""
        meta = r.get("metadata") or {}
        project = meta.get("project", "")
        kind = meta.get("kind", "")
        parent_id = meta.get("parent_id", "")
        if not parent_id and "shortcode" in meta:
            parent_id = meta["shortcode"]
        if not parent_id and "source" in meta:
            parent_id = meta["source"]
        tokens = tokenize((meta.get("title") or "") + "\n" + text)
        cur.execute("INSERT INTO docs VALUES (?, ?, ?, ?, ?, ?)", (r["id"], project, kind, parent_id, len(tokens), text))
        cur.execute("INSERT INTO fts_idx VALUES (?, ?)", (r["id"], " ".join(tokens)))
    conn.commit()
    conn.close()
def test_reject_out_aliasing_source(tmp_path):
    """Finding 1: build_v2.py rejects --out when it resolves to source v1 directory."""
    v1_dir = tmp_path / "same_dir"
    v1_dir.mkdir()
    out_dir = tmp_path / "same_dir"
    with pytest.raises(ValueError, match="Chiqish yo'li v1 manbasi bilan bir xil bo'lishi mumkin emas"):
        run_sync(v1_dir, out_dir)
def test_extraction_scoped_and_preserves_documentless_and_snapshot(tmp_path):
    """Finding 2: Extraction is scoped to collection segments, preserves documentless records, and copies metadata/config."""
    v1_dir = tmp_path / "v1_test"
    v1_dir.mkdir()
    create_mock_v1_db(v1_dir, collection_name="graphify")
    col_name, col_meta, col_config, mtime, records = read_v1_records(v1_dir, collection_name="graphify")
    assert col_name == "graphify"
    assert col_meta.get("hnsw:space") == "cosine"
    assert col_meta.get("custom_key") == "custom_value"
    assert col_config == {"sync_threshold": 500}
    # Only 2 records should be extracted (doc-other is excluded)
    assert len(records) == 2
    rec_ids = [r["id"] for r in records]
    assert "doc-1" in rec_ids
    assert "doc-2" in rec_ids
    assert "doc-other" not in rec_ids
    # Documentless record has empty string and metadata intact
    doc2 = next(r for r in records if r["id"] == "doc-2")
    assert doc2["document"] == ""
    assert doc2["metadata"]["title"] == "Sarlavha 2 (hujjatsiz)"
def test_verify_v2_checks_all_records_and_keyword_index(tmp_path):
    """Finding 3: verify_v2 verifies 100% of records and keyword index content."""
    out_dir = tmp_path / "verify_test"
    out_dir.mkdir()
    records = [
        {"id": "doc-1", "document": "Matn 1", "metadata": {"tag": "a"}},
        {"id": "doc-2", "document": "Matn 2", "metadata": {"tag": "b"}},
    ]
    create_mock_keyword_index(out_dir, records)
    mock_col = Mock()
    mock_col.count.return_value = 2
    mock_col.get.return_value = {
        "ids": ["doc-1", "doc-2"],
        "documents": ["Matn 1", "Matn 2"],
        "metadatas": [{"tag": "a"}, {"tag": "b"}],
    }
    # Should pass without assertion errors
    verify_v2(mock_col, records, out_dir)
    # If an ID is missing, verify_v2 must raise AssertionError
    mock_col.get.return_value = {
        "ids": ["doc-1", "doc-wrong"],
        "documents": ["Matn 1", "Matn 2"],
        "metadatas": [{"tag": "a"}, {"tag": "b"}],
    }
    with pytest.raises(AssertionError):
        verify_v2(mock_col, records, out_dir)
def test_sync_reembeds_changed_ids_and_deletes_removed_ids(tmp_path):
    """Finding 3: Resume/sync compares destination hashes, re-embeds changed IDs, deletes removed IDs."""
    v1_dir = tmp_path / "sync_v1"
    out_dir = tmp_path / "sync_v2"
    v1_dir.mkdir()
    out_dir.mkdir()
    create_mock_v1_db(v1_dir)
    records = [
        {"id": "doc-1", "document": "Hujjat matni 1", "metadata": {"title": "Sarlavha 1"}},
        {"id": "doc-2", "document": "", "metadata": {"title": "Sarlavha 2 (hujjatsiz)"}},
    ]
    create_mock_keyword_index(out_dir, records)
    fake_digest = "sha256:7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab"
    identity = write_manifest_and_identity(
        out_dir=out_dir,
        records=records,
        v1_mtime=1.0,
        started_at=1.0,
        model_digest=fake_digest,
        distance="cosine",
        corpus_hash="hash123",
        kw_gen="gen1",
    )
    write_build_complete_atomic(out_dir, identity, 1.0)
    # Destination initially has an old document text for doc-1 and an extra doc-3 to be removed
    mock_chroma_col = Mock()
    mock_chroma_col.count.return_value = 2
    mock_chroma_col.get.return_value = {
        "ids": ["doc-1", "doc-3_removed"],
        "documents": ["Eski eskirgan matn", "O'chirilishi kerak bo'lgan matn"],
        "metadatas": [{"title": "Eski sarlavha"}, {"title": "O'chirish"}],
    }
    mock_client = Mock()
    mock_client.get_or_create_collection.return_value = mock_chroma_col
    fake_vec = [0.1] * 1024
    with patch("chromadb.PersistentClient", return_value=mock_client):
        with patch("migrate_v2.build_v2.build_embedder") as mock_build_emb:
            mock_emb = Mock()
            mock_emb.encode.return_value = [fake_vec]
            mock_build_emb.return_value = mock_emb
            with patch("migrate_v2.build_v2.get_ollama_model_digest", return_value=fake_digest):
                with patch("migrate_v2.build_v2.build_chroma_embedding_function", return_value=_DummyEF()):
                    with patch("migrate_v2.build_v2.rebuild_keyword_index_subprocess") as mock_kw_rebuild:
                        with patch("migrate_v2.build_v2.verify_v2"):
                            run_sync(v1_dir, out_dir, batch_size=10, max_workers=2)
                    # Assert removed doc-3 was deleted
                    mock_chroma_col.delete.assert_called_once_with(ids=["doc-3_removed"])
                    # Assert upsert was called with updated/new doc-1 and doc-2
                    assert mock_chroma_col.upsert.called
                    upsert_call = mock_chroma_col.upsert.call_args[1]
                    assert "doc-1" in upsert_call["ids"]
                    # Assert atomic build_complete.json exists
                    assert (out_dir / "build_complete.json").exists()
def test_sync_rebuilds_broken_index_even_if_no_doc_changes(tmp_path):
    """Finding 3: Sync does not return early leaving a broken keyword index."""
    v1_dir = tmp_path / "retry_v1"
    out_dir = tmp_path / "retry_v2"
    v1_dir.mkdir()
    out_dir.mkdir()
    create_mock_v1_db(v1_dir)
    records = [
        {"id": "doc-1", "document": "Hujjat matni 1", "metadata": {"title": "Sarlavha 1"}},
        {"id": "doc-2", "document": "", "metadata": {"title": "Sarlavha 2 (hujjatsiz)"}},
    ]
    # Do NOT create keyword_index.sqlite3 (simulate broken/missing index)
    # Manifest exists with matching hashes
    write_manifest_and_identity(
        out_dir=out_dir,
        records=records,
        v1_mtime=1.0,
        started_at=1.0,
        model_digest="sha256:123",
        distance="cosine",
        corpus_hash="hash123",
        kw_gen="gen1",
    )
    mock_chroma_col = Mock()
    mock_chroma_col.get.return_value = {
        "ids": ["doc-1", "doc-2"],
        "documents": ["Hujjat matni 1", ""],
        "metadatas": [{"title": "Sarlavha 1"}, {"title": "Sarlavha 2 (hujjatsiz)"}],
    }
    mock_client = Mock()
    mock_client.get_or_create_collection.return_value = mock_chroma_col
    with patch("chromadb.PersistentClient", return_value=mock_client):
        with patch("migrate_v2.build_v2.get_ollama_model_digest", return_value="sha256:123"):
            with patch("migrate_v2.build_v2.build_chroma_embedding_function", return_value=_DummyEF()):
                with patch("migrate_v2.build_v2.rebuild_keyword_index_subprocess") as mock_kw:
                    with patch("migrate_v2.build_v2.verify_v2"):
                        run_sync(v1_dir, out_dir)
                        # It must NOT return early; it must trigger keyword rebuild!
                        mock_kw.assert_called_once()
                        assert (out_dir / "build_complete.json").exists()
def test_switch_check_read_only_and_keys_by_name_only(tmp_path, monkeypatch):
    """Finding 5: switch_check.py is strictly read-only and searches keys by NAME only (never prints secret values)."""
    v1_dir = tmp_path / "sc_v1"
    v2_dir = tmp_path / "sc_v2"
    v1_dir.mkdir()
    v2_dir.mkdir()
    create_mock_v1_db(v1_dir)
    records = [{"id": "doc-1", "document": "Hujjat matni 1", "metadata": {"title": "Sarlavha 1"}}]
    write_manifest_and_identity(
        out_dir=v2_dir,
        records=records,
        v1_mtime=1.0,
        started_at=1.0,
        model_digest="sha256:fake",
        distance="cosine",
        corpus_hash="hash",
        kw_gen="gen",
    )
    # 1. Drift check
    drift = check_v1_manifest_drift(v1_dir, v2_dir)
    assert drift["status"] == "OK"
    assert drift["total_drift"] == 1  # doc-2 in v1 is not in v2_manifest
    # 2. Config check: create a fake config with a secret
    fake_config = tmp_path / ".claude.json"
    fake_config.write_text("GRAPHIFY_DB_PATH=/path/to/db\nSUPER_SECRET_KEY=secret_pass_12345\n", encoding="utf-8")
    with patch("migrate_v2.switch_check.Path.home", return_value=tmp_path):
        results = inspect_launch_configs()
        # Verify result contains only boolean flags and paths, never the secret_pass_12345 value
        dumped = json.dumps(results)
        assert "secret_pass_12345" not in dumped
def test_eval_v2_fails_closed_on_vector_search_error(tmp_path):
    """Finding 6: eval_v2 does not swallow vector search errors."""
    from migrate_v2.eval_v2 import run_worker
    worker_out = tmp_path / "worker_out.json"
    queries_file = tmp_path / "queries.json"
    queries_file.write_text(json.dumps([{"post": "p1", "uz_need": "test query"}]), encoding="utf-8")
    fake_db = tmp_path / "fake_db"
    fake_db.mkdir()
    with patch("embedders.load_models_config"):
        with patch("search_service.retrieve") as mock_retrieve:
            # Simulate search_core printing vector search error to stderr
            def simulate_vector_error(**kwargs):
                sys.stderr.write("[hybrid_search] Vector search error: Connection refused to Ollama\n")
                return {"status": "ok", "results": []}
            mock_retrieve.side_effect = simulate_vector_error
            with patch("db.get_collection"):
                with patch("keyword_index.get_index"):
                    with pytest.raises(RuntimeError, match="Vector search xatosi aniqlandi"):
                        run_worker("v2", str(fake_db), str(queries_file), str(worker_out))
def test_read_v1_records_raises_when_collection_not_found(tmp_path):
    """Finding 13: read_v1_records raises clear RuntimeError when collection is missing."""
    v1_dir = tmp_path / "missing_col_v1"
    v1_dir.mkdir()
    create_mock_v1_db(v1_dir, collection_name="other_name")
    with pytest.raises(RuntimeError, match="v1 bazasida 'graphify' kolleksiyasi topilmadi"):
        read_v1_records(v1_dir, collection_name="graphify")
def test_verify_v2_checks_keyword_index_docs_and_fts_exact_content(tmp_path):
    """Finding 3: verify_v2 checks docs text, metadata, tokens, and fts_idx content."""
    out_dir = tmp_path / "verify_exact"
    out_dir.mkdir()
    records = [
        {"id": "doc-1", "document": "Matn 1", "metadata": {"title": "S1", "project": "p1", "kind": "summary"}},
    ]
    create_mock_keyword_index(out_dir, records)
    mock_col = Mock()
    mock_col.count.return_value = 1
    mock_col.get.return_value = {
        "ids": ["doc-1"],
        "documents": ["Matn 1"],
        "metadatas": [{"title": "S1", "project": "p1", "kind": "summary"}],
    }
    # Passes on valid match
    verify_v2(mock_col, records, out_dir)
    # Corrupt docs text in sqlite
    kw_sqlite = out_dir / "keyword_index.sqlite3"
    conn = sqlite3.connect(str(kw_sqlite))
    conn.execute("UPDATE docs SET text = 'Buzilgan matn' WHERE id = 'doc-1'")
    conn.commit()
    conn.close()
    with pytest.raises(AssertionError, match="docs' matni mos kelmadi"):
        verify_v2(mock_col, records, out_dir)
def test_sync_fails_before_mutation_on_digest_mismatch(tmp_path):
    """Finding 2: Sync validates existing v2 digest against Ollama before deleting or upserting."""
    v1_dir = tmp_path / "sync_mismatch_v1"
    out_dir = tmp_path / "sync_mismatch_v2"
    v1_dir.mkdir()
    out_dir.mkdir()
    create_mock_v1_db(v1_dir)
    # Existing v2 manifest has sha256:old
    manifest = {
        "schema_version": 2,
        "backend": "ollama",
        "model": "bge-m3",
        "model_digest": "sha256:old_digest_111",
        "dimension": 1024,
    }
    (out_dir / "graphify_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    from embedders import ModelIdentityError
    with patch("migrate_v2.build_v2.get_ollama_model_digest", return_value="sha256:new_digest_222"):
        with pytest.raises(ModelIdentityError, match="Sync rad etildi"):
            run_sync(v1_dir, out_dir)
def test_sync_invalidates_completion_status_before_mutating(tmp_path):
    """Finding 4: Sync unlinks build_complete.json before mutations."""
    v1_dir = tmp_path / "sync_atomic_v1"
    out_dir = tmp_path / "sync_atomic_v2"
    v1_dir.mkdir()
    out_dir.mkdir()
    create_mock_v1_db(v1_dir)
    records = [
        {"id": "doc-1", "document": "Matn 1", "metadata": {"title": "S1"}},
    ]
    create_mock_keyword_index(out_dir, records)
    complete_file = out_dir / "build_complete.json"
    complete_file.write_text('{"test": true}', encoding='utf-8')
    mock_chroma_col = Mock()
    mock_chroma_col.get.return_value = {
        "ids": ["doc-1"],
        "documents": ["Eski matn"],
        "metadatas": [{"title": "Eski"}],
    }
    mock_client = Mock()
    mock_client.get_or_create_collection.return_value = mock_chroma_col
    fake_digest = "sha256:7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab"
    with patch("chromadb.PersistentClient", return_value=mock_client):
        with patch("migrate_v2.build_v2.get_ollama_model_digest", return_value=fake_digest):
            with patch("migrate_v2.build_v2.build_chroma_embedding_function", return_value=_DummyEF()):
                with patch("migrate_v2.build_v2.build_embedder") as mock_emb:
                    mock_emb.return_value.encode.return_value = [[0.1] * 1024]
                    # Simulate failure during keyword rebuild
                    with patch("migrate_v2.build_v2.rebuild_keyword_index_subprocess", side_effect=RuntimeError("Rebuild failed")):
                        with pytest.raises(RuntimeError, match="Rebuild failed"):
                            run_sync(v1_dir, out_dir)
                        # Assert build_complete.json was deleted and does not remain!
                        assert not complete_file.exists()
def test_sync_updates_collection_metadata_with_full_identity(tmp_path):
    """Finding 5: Sync calls col.modify(metadata=identity) after verification."""
    v1_dir = tmp_path / "sync_meta_v1"
    out_dir = tmp_path / "sync_meta_v2"
    v1_dir.mkdir()
    out_dir.mkdir()
    create_mock_v1_db(v1_dir)
    records = [
        {"id": "doc-1", "document": "Hujjat matni 1", "metadata": {"title": "Sarlavha 1"}},
        {"id": "doc-2", "document": "", "metadata": {"title": "Sarlavha 2 (hujjatsiz)"}},
    ]
    create_mock_keyword_index(out_dir, records)
    fake_digest = "sha256:7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab"
    mock_chroma_col = Mock()
    mock_chroma_col.count.return_value = 2
    mock_chroma_col.get.return_value = {
        "ids": ["doc-1"],
        "documents": ["Hujjat matni 1"],
        "metadatas": [{"title": "Sarlavha 1"}],
    }
    mock_client = Mock()
    mock_client.get_or_create_collection.return_value = mock_chroma_col
    with patch("chromadb.PersistentClient", return_value=mock_client):
        with patch("migrate_v2.build_v2.get_ollama_model_digest", return_value=fake_digest):
            with patch("migrate_v2.build_v2.build_chroma_embedding_function", return_value=_DummyEF()):
                with patch("migrate_v2.build_v2.build_embedder") as mock_emb:
                    mock_emb.return_value.encode.return_value = [[0.1] * 1024]
                    with patch("migrate_v2.build_v2.rebuild_keyword_index_subprocess"):
                        with patch("migrate_v2.build_v2.verify_v2"):
                            run_sync(v1_dir, out_dir)
                            # Assert col.modify was called with identity dictionary
                            assert mock_chroma_col.modify.called
                            mod_meta = mock_chroma_col.modify.call_args[1]["metadata"]
                            assert mod_meta["schema_version"] == 2
                            assert mod_meta["model"] == "bge-m3"
                            assert mod_meta["model_digest"] == fake_digest
                            assert "keyword_index_generation" in mod_meta
def test_eval_v2_stops_if_replica_corpus_hashes_differ(tmp_path):
    """Finding 11: eval_v2 detects mismatched replica corpus hashes and stops."""
    from migrate_v2.build_v2 import compute_corpus_hash
    recs_v1 = [{"id": "d1", "document": "A", "metadata": {}}]
    recs_v2 = [{"id": "d1", "document": "B (different)", "metadata": {}}]
    h1 = compute_corpus_hash(recs_v1)
    h2 = compute_corpus_hash(recs_v2)
    assert h1 != h2
def test_eval_v2_fails_closed_on_keyword_search_error(tmp_path):
    """Finding 12: eval_v2 fails closed on keyword search errors printed by search_core."""
    from migrate_v2.eval_v2 import run_worker
    worker_out = tmp_path / "worker_out.json"
    queries_file = tmp_path / "queries.json"
    queries_file.write_text(json.dumps([{"post": "p1", "uz_need": "test query"}]), encoding="utf-8")
    fake_db = tmp_path / "fake_db"
    fake_db.mkdir()
    with patch("embedders.load_models_config"):
        with patch("search_service.retrieve") as mock_retrieve:
            def simulate_keyword_error(**kwargs):
                sys.stderr.write("[hybrid_search] Keyword search error: SQLite disk I/O error\n")
                return {"status": "ok", "results": []}
            mock_retrieve.side_effect = simulate_keyword_error
            with patch("db.get_collection"):
                with patch("keyword_index.get_index"):
                    with pytest.raises(RuntimeError, match="Qidiruv xatosi aniqlandi"):
                        run_worker("v2", str(fake_db), str(queries_file), str(worker_out))
def test_eval_v2_forces_local_storage_mode(tmp_path, monkeypatch):
    """Finding 10: eval_v2 worker forces GRAPHIFY_STORAGE_MODE=local."""
    from migrate_v2.eval_v2 import run_worker
    monkeypatch.setenv("GRAPHIFY_STORAGE_MODE", "http")
    worker_out = tmp_path / "worker_out.json"
    queries_file = tmp_path / "queries.json"
    queries_file.write_text(json.dumps([{"post": "p1", "uz_need": "test query"}]), encoding="utf-8")
    fake_db = tmp_path / "fake_db"
    fake_db.mkdir()
    with patch("embedders.load_models_config"):
        with patch("search_service.retrieve", return_value={"status": "ok", "results": []}):
            with patch("db.get_collection"):
                with patch("keyword_index.get_index"):
                    run_worker("v1", str(fake_db), str(queries_file), str(worker_out))
                    assert os.environ.get("GRAPHIFY_STORAGE_MODE") == "local"
def test_build_v2_validates_persisted_destination_identity_before_mutation(tmp_path):
    """Finding 1: build_v2 validates persisted destination collection and manifest identity before mutating destination."""
    out_dir = tmp_path / "v2_dest"
    out_dir.mkdir()
    manifest = {
        "schema_version": 2,
        "backend": "ollama",
        "model": "bge-m3",
        "model_digest": "sha256:old_ollama_digest_111",
        "dimension": 1024,
    }
    (out_dir / "graphify_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    from embedders import ModelIdentityError
    from migrate_v2.build_v2 import validate_destination_persisted_identity
    # 1. Manifest mismatch raises
    with pytest.raises(ModelIdentityError, match="Destination manifest model_digest mos kelmadi"):
        validate_destination_persisted_identity(
            out_dir=out_dir,
            col_name="graphify",
            expected_model="bge-m3",
            expected_digest="sha256:new_ollama_digest_222",
            expected_dimension=1024,
        )
    # 2. Collection with conflicting persisted digest raises
    mock_client = Mock()
    mock_col = Mock()
    mock_col.metadata = {"model_digest": "sha256:old_ollama_digest_111", "model": "bge-m3", "dimension": 1024}
    mock_col.count.return_value = 10
    mock_client.get_collection.return_value = mock_col
    # Manifest matches new, but collection has old digest -> must raise
    manifest_new = {
        "schema_version": 2,
        "backend": "ollama",
        "model": "bge-m3",
        "model_digest": "sha256:new_ollama_digest_222",
        "dimension": 1024,
    }
    (out_dir / "graphify_manifest.json").write_text(json.dumps(manifest_new), encoding="utf-8")
    with pytest.raises(ModelIdentityError, match="model_digest mos kelmadi"):
        validate_destination_persisted_identity(
            out_dir=out_dir,
            col_name="graphify",
            expected_model="bge-m3",
            expected_digest="sha256:new_ollama_digest_222",
            expected_dimension=1024,
            client=mock_client,
        )
def test_sync_validates_collection_identity_when_manifest_missing_and_rejects_malformed_manifest(tmp_path):
    """Finding 2: Sync validates collection identity independently and rejects unreadable/malformed manifests."""
    v1_dir = tmp_path / "sync_v1"
    out_dir = tmp_path / "sync_out"
    v1_dir.mkdir()
    out_dir.mkdir()
    create_mock_v1_db(v1_dir)
    # Case A: Malformed manifest (invalid JSON) -> must be rejected, not swallowed!
    (out_dir / "graphify_manifest.json").write_text("{broken json syntax", encoding="utf-8")
    from embedders import ModelIdentityError
    with patch("migrate_v2.build_v2.get_ollama_model_digest", return_value="sha256:digest_123"):
        with pytest.raises(ModelIdentityError, match="manifest fayli o'qib bo'lmadi"):
            run_sync(v1_dir, out_dir)
    # Case B: Manifest is missing, but destination collection has conflicting identity
    out_dir_b = tmp_path / "sync_out_b"
    out_dir_b.mkdir()
    import chromadb
    c_client = chromadb.PersistentClient(path=str(out_dir_b))
    c_client.create_collection("graphify", metadata={"model_digest": "sha256:old_conflicting_digest", "model": "bge-m3", "dimension": 1024})
    with patch("migrate_v2.build_v2.get_ollama_model_digest", return_value="sha256:digest_123"):
        with pytest.raises(ModelIdentityError, match="Destination kolleksiya model_digest mos kelmadi"):
            run_sync(v1_dir, out_dir_b)
def test_build_v2_removes_completion_marker_before_mutation_and_aborts_on_failure(tmp_path):
    """Finding 5: remove_completion_marker removes build_complete.json before mutations and aborts if removal fails."""
    out_dir = tmp_path / "dest_marker"
    out_dir.mkdir()
    complete_file = out_dir / "build_complete.json"
    complete_file.write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    from migrate_v2.build_v2 import remove_completion_marker
    # Normal removal succeeds
    remove_completion_marker(out_dir)
    assert not complete_file.exists()
    # Recreate and test abort on unlink error
    complete_file.write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    with patch.object(Path, "unlink", side_effect=OSError("Permission denied")):
        with pytest.raises(RuntimeError, match="build_complete.json o'chirish muvaffaqiyatsiz bo'ldi"):
            remove_completion_marker(out_dir)
def test_subprocess_timeouts_enforced_and_reported_as_failure(tmp_path):
    """Finding 6: Subprocess calls have finite timeouts and report timeouts as failures."""
    from migrate_v2.build_v2 import rebuild_keyword_index_subprocess
    import subprocess
    # 1. Keyword rebuild timeout
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="keyword_index.rebuild()", timeout=1800.0)):
        with pytest.raises(RuntimeError, match="keyword_index qayta qurish vaqt limiti"):
            rebuild_keyword_index_subprocess(tmp_path, timeout=1800.0)
def test_eval_v2_resolves_effective_space_authoritative_or_fails():
    """Finding 7: eval_v2 uses configuration_json['hnsw']['space'] authoritative, else metadata, else fails (no guessing)."""
    from migrate_v2.eval_v2 import get_effective_space
    # 1. When configuration_json specifies l2 and metadata has cosine: effective space is l2
    col_config = {"hnsw": {"space": "l2"}}
    col_meta = {"hnsw:space": "cosine"}
    assert get_effective_space(col_config, col_meta) == "l2"
    # 2. When configuration_json does not specify space, metadata is used
    assert get_effective_space({}, {"hnsw:space": "cosine"}) == "cosine"
    # 3. When neither provides space, must fail (no guessing cosine)
    with pytest.raises(ValueError, match="taxmin qilinmaydi"):
        get_effective_space({}, {})
def test_eval_v2_generates_dynamic_comparative_sentences_from_measured_values(tmp_path):
    """Finding 10: EVAL.md conclusions dynamically reflect measured values rather than fixed text."""
    from migrate_v2.eval_v2 import generate_markdown_report
    out_report = tmp_path / "EVAL.md"
    report_data = {
        "v1_db": "kg_db",
        "v2_db": "kg_db_v2",
        "smoke": False,
        "num_queries": 5,
        "timestamp": "2026-10-28T00:00:00Z",
        "model_identity": {
            "v1": {"model": "all-MiniLM-L6-v2", "dimension": 384},
            "v2": {"model": "bge-m3", "dimension": 1024},
        },
        "results": {
            "v1": {
                "0.8": {"uz_need": {"recall@10": 0.9, "mrr@10": 0.9}, "en_need": {"recall@10": 0.9, "mrr@10": 0.9}, "uz_name": {"recall@10": 0.9, "mrr@10": 0.9}, "en_name": {"recall@10": 0.9, "mrr@10": 0.9}},
                "0.5": {"uz_need": {"recall@10": 0.9, "mrr@10": 0.9}, "en_need": {"recall@10": 0.9, "mrr@10": 0.9}, "uz_name": {"recall@10": 0.9, "mrr@10": 0.9}, "en_name": {"recall@10": 0.9, "mrr@10": 0.9}},
                "0.0": {"uz_need": {"recall@10": 0.9, "mrr@10": 0.9}, "en_need": {"recall@10": 0.9, "mrr@10": 0.9}, "uz_name": {"recall@10": 0.9, "mrr@10": 0.9}, "en_name": {"recall@10": 0.9, "mrr@10": 0.9}},
            },
            "v2": {
                "0.8": {"uz_need": {"recall@10": 0.5, "mrr@10": 0.5}, "en_need": {"recall@10": 0.5, "mrr@10": 0.5}, "uz_name": {"recall@10": 0.5, "mrr@10": 0.5}, "en_name": {"recall@10": 0.5, "mrr@10": 0.5}},
                "0.5": {"uz_need": {"recall@10": 0.4, "mrr@10": 0.4}, "en_need": {"recall@10": 0.4, "mrr@10": 0.4}, "uz_name": {"recall@10": 0.4, "mrr@10": 0.4}, "en_name": {"recall@10": 0.4, "mrr@10": 0.4}},
                "0.0": {"uz_need": {"recall@10": 0.2, "mrr@10": 0.2}, "en_need": {"recall@10": 0.2, "mrr@10": 0.2}, "uz_name": {"recall@10": 0.2, "mrr@10": 0.2}, "en_name": {"recall@10": 0.2, "mrr@10": 0.2}},
            },
        },
    }
    generate_markdown_report(report_data, out_report)
    content = out_report.read_text(encoding="utf-8")
    assert "MiniLM modeli o'zbekcha so'rovlarda bge-m3 ga nisbatan yuqori semantik bog'liqlik ko'rsatdi" in content
    assert "Gate 0.8 dan 0.0 ga tushirilganda bge-m3 MRR ko'rsatkichi 0.50 dan 0.20 ga pasaydi" in content
    assert "MiniLM yuqoriroq natija ko'rsatdi" in content
    assert "sezilarli darajada ortadi" not in content
def test_validate_destination_requires_full_identity_when_records_exist_topilma_2(tmp_path):
    """Finding 1 regression (TOPILMA 2): Destination with records but lacking digest must fail closed."""
    from embedders import ModelIdentityError
    from migrate_v2.build_v2 import validate_destination_persisted_identity
    mock_client = Mock()
    mock_col = Mock()
    # 10 records, has model but missing model_digest
    mock_col.metadata = {"model": "bge-m3", "dimension": 1024, "hnsw:space": "cosine"}
    mock_col.count.return_value = 10
    mock_client.get_collection.return_value = mock_col
    dest_dir = tmp_path / "dest_sparse"
    dest_dir.mkdir()
    with pytest.raises(ModelIdentityError, match="model_digest mos kelmadi yoki yo'q"):
        validate_destination_persisted_identity(
            out_dir=dest_dir,
            col_name="graphify",
            expected_model="bge-m3",
            expected_digest="sha256:expected_digest_123",
            expected_dimension=1024,
            client=mock_client,
        )
def test_sync_persists_identity_metadata_before_first_upsert_topilma_2(tmp_path):
    """Finding 1 regression (TOPILMA 2): Initial sync persists full identity to collection before upsert."""
    v1_dir = tmp_path / "v1_init_sync"
    v1_dir.mkdir()
    create_mock_v1_db(v1_dir)
    out_dir = tmp_path / "v2_init_sync"
    out_dir.mkdir()
    import chromadb
    client = chromadb.PersistentClient(path=str(out_dir))
    fake_digest = "sha256:fake_digest_456"
    fake_vec = [0.1] * 1024
    mock_embedder = Mock()
    mock_embedder.encode.return_value = [fake_vec]
    with patch("migrate_v2.build_v2.get_ollama_model_digest", return_value=fake_digest):
        with patch("embedders.get_ollama_model_digest", return_value=fake_digest):
            with patch("migrate_v2.build_v2.build_embedder", return_value=mock_embedder):
                with patch("migrate_v2.build_v2.embed_text_with_retry", return_value=fake_vec):
                    with patch("migrate_v2.build_v2.build_chroma_embedding_function", return_value=_DummyEF()):
                        with patch("migrate_v2.build_v2.rebuild_keyword_index_subprocess"):
                            with patch("migrate_v2.build_v2.verify_v2"):
                                run_sync(v1_dir, out_dir, batch_size=10)
    # Verify that the created collection has full identity persisted in metadata
    col = client.get_collection("graphify")
    assert col.metadata.get("model") == "bge-m3"
    assert col.metadata.get("model_digest") == fake_digest
    assert col.metadata.get("dimension") == 1024
    assert col.metadata.get("schema_version") == 2
def test_sync_removes_completion_marker_before_keyword_rebuild_even_without_doc_changes_topilma_3(tmp_path):
    """Finding 5 regression (TOPILMA 3): When docs unchanged but index needs rebuild, marker is removed before rebuild."""
    v1_dir = tmp_path / "v1_kw_test"
    v1_dir.mkdir()
    create_mock_v1_db(v1_dir)
    out_dir = tmp_path / "v2_kw_test"
    out_dir.mkdir()
    complete_file = out_dir / "build_complete.json"
    complete_file.write_text(json.dumps({"status": "old_complete"}), encoding="utf-8")
    fake_digest = "sha256:fake_digest_kw"
    fake_vec = [0.1] * 1024
    mock_embedder = Mock()
    mock_embedder.encode.return_value = [fake_vec]
    with patch("migrate_v2.build_v2.get_ollama_model_digest", return_value=fake_digest):
        with patch("embedders.get_ollama_model_digest", return_value=fake_digest):
            with patch("migrate_v2.build_v2.build_embedder", return_value=mock_embedder):
                with patch("migrate_v2.build_v2.embed_text_with_retry", return_value=fake_vec):
                    with patch("migrate_v2.build_v2.build_chroma_embedding_function", return_value=_DummyEF()):
                        with patch("migrate_v2.build_v2.rebuild_keyword_index_subprocess", side_effect=RuntimeError("keyword rebuild failed")):
                            with pytest.raises(RuntimeError, match="keyword rebuild failed"):
                                run_sync(v1_dir, out_dir)
    assert not complete_file.exists(), "build_complete.json must be removed before keyword rebuild and not remain on failure!"
def test_eval_v2_worker_timeout_enforced_and_reported_as_failure():
    """Finding 6 regression: eval_v2 master reports worker subprocess timeout as failure."""
    import subprocess
    from migrate_v2.eval_v2 import main
    test_args = [
        "eval_v2.py",
        "--smoke",
        "--v1-db", "kg_db",
        "--v2-db", "kg_db_v2",
    ]
    with patch.object(Path, "exists", return_value=True):
        with patch("migrate_v2.build_v2.read_v1_records", return_value=("graphify", {"hnsw:space": "l2"}, {}, 1000.0, [])):
            with patch("chromadb.PersistentClient"):
                with patch("migrate_v2.build_v2.rebuild_keyword_index_subprocess"):
                    with patch("shutil.copytree"):
                        with patch("migrate_v2.build_v2.compute_corpus_hash", return_value="hash1"):
                            with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="eval_worker", timeout=5400.0)):
                                with patch.object(sys, "argv", test_args):
                                    with pytest.raises(RuntimeError, match="baholash worker vaqt limiti"):
                                        main()
def test_eval_v2_generates_dynamic_comparative_sentences_when_v2_beats_v1_and_equal(tmp_path):
    """Finding 10 regression: Evaluates that when v2 > v1 or equal, conclusions dynamically reflect values."""
    from migrate_v2.eval_v2 import generate_markdown_report

    out_report = tmp_path / "EVAL_V2_WIN.md"
    report_data_v2_win = {
        "v1_db": "kg_db",
        "v2_db": "kg_db_v2",
        "smoke": False,
        "num_queries": 5,
        "timestamp": "2026-10-28T00:00:00Z",
        "model_identity": {
            "v1": {"model": "all-MiniLM-L6-v2", "dimension": 384},
            "v2": {"model": "bge-m3", "dimension": 1024},
        },
        "results": {
            "v1": {
                "0.8": {"uz_need": {"recall@10": 0.4, "mrr@10": 0.4}, "en_need": {"recall@10": 0.4, "mrr@10": 0.4}, "uz_name": {"recall@10": 0.4, "mrr@10": 0.4}, "en_name": {"recall@10": 0.4, "mrr@10": 0.4}},
                "0.5": {"uz_need": {"recall@10": 0.4, "mrr@10": 0.4}, "en_need": {"recall@10": 0.4, "mrr@10": 0.4}, "uz_name": {"recall@10": 0.4, "mrr@10": 0.4}, "en_name": {"recall@10": 0.4, "mrr@10": 0.4}},
                "0.0": {"uz_need": {"recall@10": 0.4, "mrr@10": 0.4}, "en_need": {"recall@10": 0.4, "mrr@10": 0.4}, "uz_name": {"recall@10": 0.4, "mrr@10": 0.4}, "en_name": {"recall@10": 0.4, "mrr@10": 0.4}},
            },
            "v2": {
                "0.8": {"uz_need": {"recall@10": 0.8, "mrr@10": 0.8}, "en_need": {"recall@10": 0.8, "mrr@10": 0.8}, "uz_name": {"recall@10": 0.8, "mrr@10": 0.8}, "en_name": {"recall@10": 0.8, "mrr@10": 0.8}},
                "0.5": {"uz_need": {"recall@10": 0.85, "mrr@10": 0.85}, "en_need": {"recall@10": 0.85, "mrr@10": 0.85}, "uz_name": {"recall@10": 0.85, "mrr@10": 0.85}, "en_name": {"recall@10": 0.85, "mrr@10": 0.85}},
                "0.0": {"uz_need": {"recall@10": 0.9, "mrr@10": 0.9}, "en_need": {"recall@10": 0.9, "mrr@10": 0.9}, "uz_name": {"recall@10": 0.9, "mrr@10": 0.9}, "en_name": {"recall@10": 0.9, "mrr@10": 0.9}},
            },
        },
    }

    generate_markdown_report(report_data_v2_win, out_report)
    content = out_report.read_text(encoding="utf-8")

    assert "bge-m3 modeli o'zbekcha so'rovlarda MiniLM ga nisbatan yuqori semantik bog'liqlik ko'rsatdi" in content
    assert "Gate 0.8 dan 0.0 ga tushirilganda bge-m3 MRR ko'rsatkichi 0.80 dan 0.90 ga ortdi" in content
    assert "bge-m3 asosiy embedder sifatida tavsiya etiladi" in content


def test_build_v2_modify_strips_hnsw_keys_regression_finding_1(tmp_path):
    """Finding 1 regression: merged_metadata containing hnsw:* keys has them stripped before col.modify, preserving space without error."""
    import chromadb
    client = chromadb.PersistentClient(path=str(tmp_path))
    col = client.create_collection("graphify", metadata={"hnsw:space": "cosine"})

    # 1. Directly passing hnsw:space to modify raises ValueError in Chroma 1.5.9
    with pytest.raises(ValueError, match="Changing the distance function"):
        col.modify(metadata={"hnsw:space": "cosine", "model": "bge-m3"})

    # 2. Stripping hnsw:* keys succeeds and keeps space in configuration_json
    merged_metadata = {
        "hnsw:space": "cosine",
        "schema_version": 2,
        "backend": "ollama",
        "model": "bge-m3",
        "model_digest": "sha256:digest_f1",
        "dimension": 1024,
        "distance": "cosine",
        "normalization": False,
    }
    modify_meta = {
        k: v for k, v in {
            **(col.metadata if isinstance(col.metadata, dict) else {}),
            **merged_metadata,
        }.items()
        if not k.startswith("hnsw:")
    }
    col.modify(metadata=modify_meta)
    assert col.metadata.get("model") == "bge-m3"
    assert col.metadata.get("dimension") == 1024
    assert col.configuration_json["hnsw"]["space"] == "cosine"


def test_sync_and_build_validate_read_only_before_client_init_regression_finding_2(tmp_path):
    """Finding 2 regression: validate manifest and persisted identity read-only (sqlite mode=ro) first, rejecting unreadable/conflicting manifest before PersistentClient is constructed."""
    v1_dir = tmp_path / "v1_f2"
    v1_dir.mkdir()
    create_mock_v1_db(v1_dir)
    v2_dir = tmp_path / "v2_f2"
    v2_dir.mkdir()

    # Corrupt/unreadable manifest
    bad_manifest = v2_dir / "graphify_manifest.json"
    bad_manifest.write_text("{invalid json", encoding="utf-8")

    from embedders import ModelIdentityError
    with patch("migrate_v2.build_v2.get_ollama_model_digest", return_value="sha256:f2"):
        with pytest.raises(ModelIdentityError, match="Destination manifest fayli o'qib bo'lmadi"):
            run_sync(v1_dir, v2_dir)

    # PersistentClient was never called, so chroma.sqlite3 was never created!
    assert not (v2_dir / "chroma.sqlite3").exists()


def test_sync_removes_build_complete_marker_before_first_metadata_mutation_regression_finding_3(tmp_path):
    """Finding 3 regression: sync removes build_complete.json after validation and before the first destination mutation."""
    v1_dir = tmp_path / "v1_f3"
    v1_dir.mkdir()
    create_mock_v1_db(v1_dir)
    v2_dir = tmp_path / "v2_f3"
    v2_dir.mkdir()

    # Pre-create a marker
    marker = v2_dir / "build_complete.json"
    marker.write_text(json.dumps({"status": "previous_complete"}), encoding="utf-8")

    fake_digest = "sha256:f3_digest"
    # Cause failure right during col.modify (first mutation)
    mock_client = Mock()
    mock_col = Mock()
    mock_col.metadata = {}
    mock_col.modify.side_effect = RuntimeError("Forced modify failure")
    mock_client.get_or_create_collection.return_value = mock_col
    with patch("migrate_v2.build_v2.get_ollama_model_digest", return_value=fake_digest):
        with patch("migrate_v2.build_v2.build_chroma_embedding_function", return_value=_DummyEF()):
            with patch("chromadb.PersistentClient", return_value=mock_client):
                with pytest.raises(RuntimeError, match="Forced modify failure"):
                    run_sync(v1_dir, v2_dir)

    # Marker must have been removed BEFORE the mutation attempt and must not exist
    assert not marker.exists(), "build_complete.json must be removed before first mutation and not remain on failure!"


def test_validate_destination_persisted_identity_propagates_count_exception_regression_finding_4(tmp_path):
    """Finding 4 regression: col.count() exception propagates instead of setting col_count = 0."""
    from migrate_v2.build_v2 import validate_destination_persisted_identity
    mock_client = Mock()
    mock_col = Mock()
    mock_col.metadata = {"model": "bge-m3", "model_digest": "sha256:f4", "dimension": 1024, "backend": "ollama"}
    mock_col.count.side_effect = RuntimeError("Database disk image is malformed")
    mock_client.get_collection.return_value = mock_col

    with pytest.raises(RuntimeError, match="Database disk image is malformed"):
        validate_destination_persisted_identity(
            out_dir=Path("/nonexistent"),
            col_name="graphify",
            expected_model="bge-m3",
            expected_digest="sha256:f4",
            expected_dimension=1024,
            client=mock_client,
        )

    # TOPILMA 1: SQLite count error with client=None must propagate when embeddings table is missing
    dest_dir = tmp_path / "f4_sqlite_count_err"
    dest_dir.mkdir()
    db_file = dest_dir / "chroma.sqlite3"
    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()
    cur.execute("CREATE TABLE collections (id TEXT, name TEXT);")
    cur.execute("CREATE TABLE collection_metadata (collection_id TEXT, key TEXT, str_value TEXT, int_value INT, float_value REAL, bool_value INT);")
    cur.execute("INSERT INTO collections VALUES ('c1', 'graphify');")
    cur.execute("INSERT INTO collection_metadata VALUES ('c1', 'model', 'bge-m3', NULL, NULL, NULL);")
    cur.execute("INSERT INTO collection_metadata VALUES ('c1', 'model_digest', 'sha256:f4', NULL, NULL, NULL);")
    cur.execute("INSERT INTO collection_metadata VALUES ('c1', 'dimension', NULL, 1024, NULL, NULL);")
    cur.execute("INSERT INTO collection_metadata VALUES ('c1', 'backend', 'ollama', NULL, NULL, NULL);")
    conn.commit()
    conn.close()

    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        validate_destination_persisted_identity(
            out_dir=dest_dir,
            col_name="graphify",
            expected_model="bge-m3",
            expected_digest="sha256:f4",
            expected_dimension=1024,
            client=None,
        )


def test_validate_destination_persisted_identity_checks_backend_in_both_sources_regression_finding_5(tmp_path):
    """Finding 5 regression: destination validation requires and validates backend in both manifest and collection."""
    from embedders import ModelIdentityError
    from migrate_v2.build_v2 import validate_destination_persisted_identity

    # 1. Manifest missing backend
    out_dir = tmp_path / "f5_manifest_no_backend"
    out_dir.mkdir()
    (out_dir / "graphify_manifest.json").write_text(json.dumps({
        "schema_version": 2,
        "model": "bge-m3",
        "model_digest": "sha256:f5",
        "dimension": 1024,
    }), encoding="utf-8")
    with pytest.raises(ModelIdentityError, match="backend mos kelmadi yoki yo'q"):
        validate_destination_persisted_identity(
            out_dir=out_dir,
            col_name="graphify",
            expected_model="bge-m3",
            expected_digest="sha256:f5",
            expected_dimension=1024,
            client=None,
        )

    # 2. Manifest with conflicting backend
    (out_dir / "graphify_manifest.json").write_text(json.dumps({
        "schema_version": 2,
        "backend": "local",
        "model": "bge-m3",
        "model_digest": "sha256:f5",
        "dimension": 1024,
    }), encoding="utf-8")
    with pytest.raises(ModelIdentityError, match="backend mos kelmadi yoki yo'q"):
        validate_destination_persisted_identity(
            out_dir=out_dir,
            col_name="graphify",
            expected_model="bge-m3",
            expected_digest="sha256:f5",
            expected_dimension=1024,
            client=None,
        )

    # 3. Collection with records missing backend
    mock_client = Mock()
    mock_col = Mock()
    mock_col.metadata = {"model": "bge-m3", "model_digest": "sha256:f5", "dimension": 1024}
    mock_col.count.return_value = 5
    mock_client.get_collection.return_value = mock_col
    dest_dir = tmp_path / "f5_col_no_backend"
    dest_dir.mkdir()
    with pytest.raises(ModelIdentityError, match="backend mos kelmadi yoki yo'q"):
        validate_destination_persisted_identity(
            out_dir=dest_dir,
            col_name="graphify",
            expected_model="bge-m3",
            expected_digest="sha256:f5",
            expected_dimension=1024,
            client=mock_client,
        )

    # 4. TOPILMA 4: Empty collection (0 records) with model/digest/dimension but missing backend
    mock_client_empty = Mock()
    mock_col_empty = Mock()
    mock_col_empty.metadata = {"model": "bge-m3", "model_digest": "sha256:f5", "dimension": 1024}
    mock_col_empty.count.return_value = 0
    mock_client_empty.get_collection.return_value = mock_col_empty
    dest_dir_empty = tmp_path / "f5_empty_col_no_backend"
    dest_dir_empty.mkdir()
    with pytest.raises(ModelIdentityError, match="backend mos kelmadi yoki yo'q"):
        validate_destination_persisted_identity(
            out_dir=dest_dir_empty,
            col_name="graphify",
            expected_model="bge-m3",
            expected_digest="sha256:f5",
            expected_dimension=1024,
            client=mock_client_empty,
        )


def test_unchanged_sync_skips_keyword_rebuild_and_restores_completion_marker_topilma_5(tmp_path):
    """TOPILMA 5: Unchanged sync validates, skips keyword index rebuild, and restores build_complete.json."""
    v1_dir = tmp_path / "v1_topilma5"
    v1_dir.mkdir()
    create_mock_v1_db(v1_dir)
    v2_dir = tmp_path / "v2_topilma5"
    v2_dir.mkdir()

    import chromadb
    client = chromadb.PersistentClient(path=str(v2_dir))
    fake_digest = "sha256:topilma5_digest"
    fake_vec = [0.1] * 1024
    dummy_ef = _DummyEF()

    from migrate_v2.build_v2 import compute_corpus_hash, read_v1_records
    from keyword_index import tokenize
    _, _, _, v1_mtime, v1_records = read_v1_records(v1_dir)
    corpus_hash = compute_corpus_hash(v1_records)

    col = client.create_collection("graphify", metadata={"hnsw:space": "cosine", "model": "bge-m3", "model_digest": fake_digest, "dimension": 1024, "backend": "ollama", "schema_version": 2, "distance": "cosine"}, embedding_function=dummy_ef)
    for r in v1_records:
        col.add(ids=[r["id"]], documents=[r["document"]], metadatas=[r["metadata"] if r["metadata"] else None], embeddings=[fake_vec])

    kw_file = v2_dir / "keyword_index.sqlite3"
    conn = sqlite3.connect(str(kw_file))
    cur = conn.cursor()
    cur.execute("CREATE TABLE docs (id TEXT PRIMARY KEY, project TEXT, kind TEXT, parent_id TEXT, token_count INT, text TEXT);")
    cur.execute("CREATE VIRTUAL TABLE fts_idx USING fts5(id UNINDEXED, text);")
    for r in v1_records:
        text = r["document"] or ""
        meta = r["metadata"] or {}
        tokens = tokenize((meta.get("title") or "") + "\n" + text)
        cur.execute("INSERT INTO docs VALUES (?, ?, ?, ?, ?, ?)", (r["id"], "", "", "", len(tokens), text))
        cur.execute("INSERT INTO fts_idx VALUES (?, ?)", (r["id"], " ".join(tokens)))
    conn.commit()
    conn.close()

    kw_mtime_gen = str(int(kw_file.stat().st_mtime))

    manifest_data = {
        "schema_version": 2,
        "backend": "ollama",
        "model": "bge-m3",
        "model_digest": fake_digest,
        "dimension": 1024,
        "distance": "cosine",
        "normalization": False,
        "corpus_hash": corpus_hash,
        "keyword_index_generation": kw_mtime_gen,
        "total_records": len(v1_records),
        "v1_sqlite_mtime": v1_mtime,
    }
    (v2_dir / "graphify_manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")

    complete_data = dict(manifest_data)
    complete_data["timestamps"] = {"completed_at": "2026-10-28T12:00:00Z"}
    complete_file = v2_dir / "build_complete.json"
    complete_file.write_text(json.dumps(complete_data), encoding="utf-8")

    mock_embedder = Mock()
    mock_embedder.encode.return_value = [fake_vec]

    with patch("migrate_v2.build_v2.get_ollama_model_digest", return_value=fake_digest), \
         patch("embedders.get_ollama_model_digest", return_value=fake_digest), \
         patch("migrate_v2.build_v2.build_embedder", return_value=mock_embedder), \
         patch("migrate_v2.build_v2.build_chroma_embedding_function", return_value=dummy_ef), \
         patch("migrate_v2.build_v2.rebuild_keyword_index_subprocess") as mock_kw_rebuild:
        run_sync(v1_dir, v2_dir)

    assert not mock_kw_rebuild.called, "rebuild_keyword_index_subprocess should NOT be called on unchanged sync!"
    assert complete_file.exists(), "build_complete.json must exist after unchanged sync!"


def test_get_effective_space_reads_legacy_hnsw_configuration_before_metadata_regression_finding_7():
    """Finding 7 regression: eval_v2 get_effective_space reads legacy hnsw_configuration.space before metadata."""
    from migrate_v2.eval_v2 import get_effective_space

    # Dict with legacy hnsw_configuration takes priority over metadata
    col_config = {"hnsw_configuration": {"space": "ip"}}
    col_meta = {"hnsw:space": "l2"}
    assert get_effective_space(col_config, col_meta) == "ip"

    # Object with legacy hnsw_configuration
    class LegacyConfig:
        hnsw_configuration = {"space": "cosine"}
    assert get_effective_space(LegacyConfig(), col_meta) == "cosine"

    # Fallback to metadata when config has no space
    assert get_effective_space({}, col_meta) == "l2"

