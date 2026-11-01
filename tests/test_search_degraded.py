"""Tests for degraded search mode when vector search encounters exceptions (Finding / Requirement T6)."""

from unittest.mock import patch
import chromadb
import pytest

import keyword_index
import search_core
import yozuv


@pytest.fixture
def temp_search_col(tmp_path, monkeypatch):
    """Fixture providing an isolated ephemeral Chroma collection and keyword index."""
    db_dir = tmp_path / "kg_db"
    db_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(yozuv, "DB_DIR", db_dir)
    monkeypatch.setattr(yozuv, "LOCK_FILE", db_dir / ".yozuv.lock")
    monkeypatch.setattr(yozuv, "PENDING_FILE", db_dir / ".pending_fts.json")
    monkeypatch.setattr(keyword_index, "DB_PATH", db_dir / "keyword_index.sqlite3")
    keyword_index._index_singleton = None

    client = chromadb.EphemeralClient()
    col = client.get_or_create_collection("test_degraded", metadata={"hnsw:space": "cosine"})

    yield col, db_dir

    keyword_index._index_singleton = None
    try:
        client.delete_collection("test_degraded")
    except Exception:
        pass


def test_vector_search_failure_sets_degraded_flag_with_lexical_results(temp_search_col):
    """T6: When vector search throws an exception, hybrid_search proceeds lexical-only,
    preserves status='ok', returns lexical results, and attaches degraded flags.
    """
    col, _ = temp_search_col
    fts = keyword_index.get_index()

    docs = [
        "Python dasturlash tili va uning kutubxonalari haqida to'liq qo'llanma.",
        "JavaScript tili yordamida frontend va brauzer interfeyslarini yaratish.",
    ]
    metas = [
        {"id": "doc_py", "title": "Python darslari"},
        {"id": "doc_js", "title": "JavaScript darslari"},
    ]
    yozuv.upsert(col, [m["id"] for m in metas], docs, metas)

    # Mock collection.query to fail
    with patch.object(col, "query", side_effect=RuntimeError("Vector index corrupted or unavailable")):
        res = search_core.hybrid_search(
            "Python dasturlash tili",
            top_k=5,
            where=None,
            collection=col,
            kw_index=fts,
        )

    # Status contract must be preserved
    assert res["status"] == "ok"
    assert res["degraded"] == "vector_search_failed"
    assert "Vector index corrupted or unavailable" in res["degraded_error"]
    assert len(res["results"]) > 0
    assert any("Python" in r["text"] for r in res["results"])


def test_vector_search_failure_sets_degraded_flag_with_insufficient_evidence(temp_search_col):
    """T6: When vector search fails and lexical search finds no matching documents,
    status='insufficient_evidence' is preserved and degraded flags are present.
    """
    col, _ = temp_search_col
    fts = keyword_index.get_index()

    docs = [
        "Python dasturlash tili va uning kutubxonalari haqida to'liq qo'llanma.",
    ]
    metas = [
        {"id": "doc_py", "title": "Python darslari"},
    ]
    yozuv.upsert(col, [m["id"] for m in metas], docs, metas)

    # Query has zero overlap with the indexed document
    with patch.object(col, "query", side_effect=RuntimeError("Simulated vector engine failure")):
        res = search_core.hybrid_search(
            "Astronomiya va sayyoralar harakati 9999",
            top_k=5,
            where=None,
            collection=col,
            kw_index=fts,
        )

    assert res["status"] == "insufficient_evidence"
    assert res["degraded"] == "vector_search_failed"
    assert "Simulated vector engine failure" in res["degraded_error"]
    assert len(res["results"]) == 0


def test_normal_search_does_not_contain_degraded_flag(temp_search_col):
    """T6: Normal search without vector failures must not contain 'degraded' or 'degraded_error'."""
    col, _ = temp_search_col
    fts = keyword_index.get_index()

    docs = [
        "PostgreSQL ma'lumotlar bazasi va SQL so'rovlar optimizatsiyasi.",
    ]
    metas = [
        {"id": "doc_pg", "title": "PostgreSQL"},
    ]
    yozuv.upsert(col, [m["id"] for m in metas], docs, metas)

    res = search_core.hybrid_search(
        "PostgreSQL ma'lumotlar bazasi",
        top_k=5,
        where=None,
        collection=col,
        kw_index=fts,
    )

    assert res["status"] == "ok"
    assert "degraded" not in res
    assert "degraded_error" not in res
    assert len(res["results"]) > 0


def test_stale_collection_exception_is_re_raised(temp_search_col):
    """T6: Stale collection / not found exceptions must be re-raised (not caught as vector failure)."""
    col, _ = temp_search_col
    fts = keyword_index.get_index()
    yozuv.upsert(col, ["doc_stale"], ["sample doc"], [{"id": "doc_stale"}])

    with patch.object(col, "query", side_effect=RuntimeError("collection does not exist")):
        with pytest.raises(RuntimeError, match="collection does not exist"):
            search_core.hybrid_search(
                "query",
                top_k=5,
                where=None,
                collection=col,
                kw_index=fts,
            )
