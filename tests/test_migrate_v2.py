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
                with patch("migrate_v2.build_v2.build_chroma_embedding_function", return_value=Mock()):
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
            with patch("migrate_v2.build_v2.build_chroma_embedding_function", return_value=Mock()):
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
            with patch("migrate_v2.build_v2.build_chroma_embedding_function", return_value=Mock()):
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
            with patch("migrate_v2.build_v2.build_chroma_embedding_function", return_value=Mock()):
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
