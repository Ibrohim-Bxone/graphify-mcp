import json
import os
from unittest.mock import Mock, patch
import numpy as np
import pytest

import similarity
import search_core
import keyword_index
import reranker


class MockChromaCollection:
    def __init__(self, records, space=None, config_space=None, col_id="col-mock", count_val=None, seq_id=0):
        self.records = records  # id -> (text, meta, vec)
        self.id = col_id
        self._count = len(records) if count_val is None else count_val
        self.seq_id = seq_id
        self.metadata = {}
        if space is not None:
            self.metadata["hnsw:space"] = space
        if config_space is not None:
            self.configuration_json = {"hnsw": {"space": config_space}}
        else:
            self.configuration_json = {}
        self.get_calls = 0

    def count(self):
        return self._count

    def get(self, ids=None, limit=None, include=None):
        self.get_calls += 1
        include = include or []
        if ids is not None:
            matched_ids = [i for i in ids if i in self.records]
        else:
            matched_ids = list(self.records.keys())[:limit]

        res = {"ids": matched_ids}
        if "embeddings" in include:
            res["embeddings"] = [self.records[i][2] for i in matched_ids]
        if "documents" in include:
            res["documents"] = [self.records[i][0] for i in matched_ids]
        if "metadatas" in include:
            res["metadatas"] = [self.records[i][1] for i in matched_ids]
        return res

    def query(self, query_texts=None, query_embeddings=None, n_results=10, where=None, include=None):
        keys = list(self.records.keys())[:n_results]
        res = {
            "ids": [keys],
            "documents": [[self.records[k][0] for k in keys]],
            "metadatas": [[self.records[k][1] for k in keys]],
            "distances": [[0.05 + 0.1 * i for i in range(len(keys))]],
        }
        if include and "embeddings" in include:
            res["embeddings"] = [[self.records[k][2] for k in keys]]
        return res


@pytest.fixture
def kw_index(tmp_path, monkeypatch):
    monkeypatch.setattr(keyword_index, "DB_PATH", tmp_path / "kw_test.sqlite3")
    idx = keyword_index.KeywordIndex()
    yield idx
    idx.conn.close()


def test_space_helper_never_guesses():
    # 1. configuration_json takes precedence
    col1 = MockChromaCollection({}, space="l2", config_space="cosine")
    assert similarity.collection_space(col1) == "cosine"

    # 2. metadata fallback if no configuration_json space
    col2 = MockChromaCollection({}, space="ip")
    assert similarity.collection_space(col2) == "ip"

    # 3. Missing space raises ValueError
    col3 = MockChromaCollection({})
    with pytest.raises(ValueError, match="hnsw:space topilmadi, l2 deb taxmin qilinmaydi"):
        similarity.collection_space(col3)

    # 4. hybrid_search with missing space raises ValueError without guessing l2
    dummy_kw = Mock()
    with pytest.raises(ValueError, match="hnsw:space topilmadi"):
        search_core.hybrid_search("query", top_k=5, where=None, collection=col3, kw_index=dummy_kw)


def test_unit_length_cache_hit():
    similarity._unit_length_cache.clear()

    # Vektorlar birlik uzunlikda
    records = {
        "doc1": ("text1", {"id": "doc1"}, np.array([1.0, 0.0])),
        "doc2": ("text2", {"id": "doc2"}, np.array([0.0, 1.0])),
    }
    col = MockChromaCollection(records, space="l2", col_id="col-unit-cache-test", count_val=2)

    # 1-chaqiruv: cache miss -> get_calls == 1
    assert similarity._check_unit_length(col) is True
    assert col.get_calls == 1

    # 2-chaqiruv: cache hit -> get_calls still 1
    assert similarity._check_unit_length(col) is True
    assert col.get_calls == 1

    # Count o'zgarsa (yangi hujjat qo'shildi) -> cache miss -> get_calls == 2
    col._count = 3
    assert similarity._check_unit_length(col) is True
    assert col.get_calls == 2


def test_fusion_weights_change_order(kw_index, monkeypatch):
    # Disable rerank to test pure RRF fusion weights
    monkeypatch.setenv("GRAPHIFY_RERANK", "0")

    # doc_dense has good vector similarity, rank 1 in kw
    # doc_lexical has good lexical score, rank 1 in vector
    records = {
        "doc_dense": ("python database storage architecture", {"id": "doc_dense", "title": "Dense"}, [1.0, 0.0]),
        "doc_lexical": ("python database query optimizer index database python", {"id": "doc_lexical", "title": "Lexical"}, [0.0, 1.0]),
    }
    col = MockChromaCollection(records, space="cosine")

    # Insert into kw_index: doc_lexical has higher TF for python database
    kw_index.upsert(
        ["doc_dense", "doc_lexical"],
        [records["doc_dense"][0], records["doc_lexical"][0]],
        [records["doc_dense"][1], records["doc_lexical"][1]],
    )

    # Mock query to return doc_dense first in vector
    def mock_query(**kwargs):
        return {
            "ids": [["doc_dense", "doc_lexical"]],
            "documents": [[records["doc_dense"][0], records["doc_lexical"][0]]],
            "metadatas": [[records["doc_dense"][1], records["doc_lexical"][1]]],
            "distances": [[0.05, 0.40]],
        }
    col.query = mock_query

    # Default weights: dense=1.0, lexical=0.2 -> doc_dense wins
    monkeypatch.setenv("GRAPHIFY_DENSE_WEIGHT", "1.0")
    monkeypatch.setenv("GRAPHIFY_LEXICAL_WEIGHT", "0.2")
    res1 = search_core.hybrid_search("python database", top_k=2, where=None, collection=col, kw_index=kw_index)
    assert res1["status"] == "ok"
    assert res1["results"][0]["id"] == "doc_dense"

    # Invert weights: dense=0.1, lexical=2.0 -> doc_lexical wins
    monkeypatch.setenv("GRAPHIFY_DENSE_WEIGHT", "0.1")
    monkeypatch.setenv("GRAPHIFY_LEXICAL_WEIGHT", "2.0")
    res2 = search_core.hybrid_search("python database", top_k=2, where=None, collection=col, kw_index=kw_index)
    assert res2["status"] == "ok"
    assert res2["results"][0]["id"] == "doc_lexical"


@pytest.mark.reranker
def test_rerank_reorders_top_n_without_dropping_items(kw_index, monkeypatch):
    reranker.reset_reranker()
    records = {
        "doc_a": ("alpha search engine database", {"id": "doc_a", "title": "Doc A"}, [1.0, 0.0]),
        "doc_b": ("beta search engine database", {"id": "doc_b", "title": "Doc B"}, [0.9, 0.1]),
        "doc_c": ("gamma search engine database", {"id": "doc_c", "title": "Doc C"}, [0.8, 0.2]),
    }
    col = MockChromaCollection(records, space="cosine")
    kw_index.upsert(
        list(records.keys()),
        [v[0] for v in records.values()],
        [v[1] for v in records.values()],
    )

    # Initial RRF order will be doc_a, doc_b, doc_c
    # Mock reranker to assign doc_c the highest score (0.99) and doc_a the lowest (0.01)
    mock_reranker = Mock()
    mock_reranker.last_error = None
    # Pairs order in to_rerank: [doc_a, doc_b, doc_c]
    mock_reranker.score_pairs.return_value = [0.01, 0.50, 0.99]

    with patch("reranker.get_reranker_with_reason", return_value=(mock_reranker, None)):
        res = search_core.hybrid_search("search engine database", top_k=3, where=None, collection=col, kw_index=kw_index)

    assert res["status"] == "ok"
    assert res["rerank"] == "on"
    assert res["rerank_reason"] is None
    # All 3 items preserved (nothing dropped)
    assert len(res["results"]) == 3
    # Reordered: doc_c should be promoted to #1
    assert res["results"][0]["id"] == "doc_c"
    # Each item carries fused_score and rerank_score
    for item in res["results"]:
        assert "fused_score" in item
        assert "rerank_score" in item
        assert isinstance(item["fused_score"], float)
        assert isinstance(item["rerank_score"], float)


def test_rerank_off_env_setting(kw_index, monkeypatch):
    reranker.reset_reranker()
    records = {
        "doc1": ("search query document test", {"id": "doc1", "title": "Title"}, [1.0, 0.0]),
    }
    col = MockChromaCollection(records, space="cosine")
    kw_index.upsert(["doc1"], [records["doc1"][0]], [records["doc1"][1]])

    monkeypatch.setenv("GRAPHIFY_RERANK", "0")
    res = search_core.hybrid_search("search query", top_k=1, where=None, collection=col, kw_index=kw_index)
    assert res["status"] == "ok"
    assert res["rerank"] == "off"
    assert res["rerank_reason"] == "GRAPHIFY_RERANK=0"
    item = res["results"][0]
    assert "fused_score" in item
    assert "rerank_score" in item
    assert item["rerank_score"] is None


@pytest.mark.reranker
def test_fallback_missing_model_reports_unavailable(kw_index, monkeypatch):
    reranker.reset_reranker()
    records = {
        "doc1": ("search query document test", {"id": "doc1", "title": "Title"}, [1.0, 0.0]),
    }
    col = MockChromaCollection(records, space="cosine")
    kw_index.upsert(["doc1"], [records["doc1"][0]], [records["doc1"][1]])

    with patch("reranker.get_reranker_with_reason", return_value=(None, "model missing: not found in cache")):
        res = search_core.hybrid_search("search query", top_k=1, where=None, collection=col, kw_index=kw_index)

    assert res["status"] == "ok"
    assert res["rerank"] == "unavailable"
    assert "model missing" in res["rerank_reason"]
    assert len(res["results"]) == 1
    assert res["results"][0]["rerank_score"] is None


@pytest.mark.reranker
def test_fallback_budget_exceeded_reports_unavailable(kw_index, monkeypatch):
    reranker.reset_reranker()
    records = {
        "doc1": ("search query document test", {"id": "doc1", "title": "Title"}, [1.0, 0.0]),
    }
    col = MockChromaCollection(records, space="cosine")
    kw_index.upsert(["doc1"], [records["doc1"][0]], [records["doc1"][1]])

    mock_reranker = Mock()
    mock_reranker.score_pairs.return_value = None
    mock_reranker.last_error = "budget exceeded"

    with patch("reranker.get_reranker_with_reason", return_value=(mock_reranker, None)):
        res = search_core.hybrid_search("search query", top_k=1, where=None, collection=col, kw_index=kw_index)

    assert res["status"] == "ok"
    assert res["rerank"] == "unavailable"
    assert res["rerank_reason"] == "budget exceeded"
    assert len(res["results"]) == 1


def test_search_service_retrieve_aggregates_rerank_fallback_across_variants(monkeypatch):
    import search_service
    import search_policy

    monkeypatch.setattr(search_policy, "query_variants", lambda q: ["variant1", "variant2"])

    def mock_hybrid_search(query, **kwargs):
        if query == "variant1":
            return {
                "results": [{"id": "d1", "text": "text1", "meta": {"id": "d1"}}],
                "status": "ok",
                "related_candidates": [],
                "rerank": "on",
                "rerank_reason": None,
            }
        else:
            return {
                "results": [{"id": "d2", "text": "text2", "meta": {"id": "d2"}}],
                "status": "ok",
                "related_candidates": [],
                "rerank": "unavailable",
                "rerank_reason": "budget exceeded",
            }

    monkeypatch.setattr(search_core, "hybrid_search", mock_hybrid_search)
    dummy_col = Mock()
    dummy_kw = Mock()
    res = search_service.retrieve("test query", top_k=2, collection=dummy_col, kw_index=dummy_kw)

    assert res["status"] == "ok"
    assert res["rerank"] == "unavailable"
    assert "budget exceeded" in res["rerank_reason"]


def test_finding4_unit_length_cache_mutation_signal_and_skip_when_missing():
    """Finding 4 regression: cache key includes mutation signal; invalidates on mutation without size change; skips cache when no signal."""
    similarity._unit_length_cache.clear()

    # 1. Collection with mutation signal (seq_id)
    records = {
        "doc1": ("text1", {"id": "doc1"}, np.array([1.0, 0.0])),
        "doc2": ("text2", {"id": "doc2"}, np.array([0.0, 1.0])),
    }
    col = MockChromaCollection(records, space="l2", col_id="col-mutation-test", count_val=2, seq_id=0)

    # First call: cache miss -> get_calls = 1
    assert similarity._check_unit_length(col) is True
    assert col.get_calls == 1

    # Second call (same count, same seq_id): cache hit -> get_calls = 1
    assert similarity._check_unit_length(col) is True
    assert col.get_calls == 1

    # Embeddings change without count changing: seq_id increments
    col.records["doc2"] = ("text2", {"id": "doc2"}, np.array([2.0, 2.0]))
    col.seq_id = 1
    # Cache key changed -> cache miss -> get_calls = 2, detects non-unit vector!
    assert similarity._check_unit_length(col) is False
    assert col.get_calls == 2

    # 2. Collection without reliable mutation signal (seq_id is None)
    col_no_sig = MockChromaCollection(records, space="l2", col_id="col-no-signal", count_val=2, seq_id=None)
    assert similarity._check_unit_length(col_no_sig) is False
    assert col_no_sig.get_calls == 1

    # Second call without signal: skips cache -> get_calls = 2
    assert similarity._check_unit_length(col_no_sig) is False
    assert col_no_sig.get_calls == 2

    # 3. Collection with count=None and mutation_signal=None: also skips cache
    col_no_count = MockChromaCollection(records, space="l2", col_id="col-no-count-no-sig", count_val=None, seq_id=None)
    col_no_count.count = None
    assert similarity._check_unit_length(col_no_count) is False
    assert col_no_count.get_calls == 1
    assert similarity._check_unit_length(col_no_count) is False
    assert col_no_count.get_calls == 2


def test_finding5_count_fn_failure_escapes_handled_and_cache_bypassed():
    """Finding 5 regression: count_fn() exception is caught, does not escape, and bypasses cache."""
    similarity._unit_length_cache.clear()
    records = {
        "doc1": ("text1", {"id": "doc1"}, np.array([1.0, 0.0])),
    }
    col = MockChromaCollection(records, space="l2", col_id="col-err-test", count_val=1, seq_id=0)

    def failing_count():
        raise RuntimeError("database count failed")

    col.count = failing_count

    # Exception from count() is caught; unit check succeeds and bypasses cache
    assert similarity._check_unit_length(col) is True
    assert col.get_calls == 1

    # Next call also queries because cache was bypassed
    assert similarity._check_unit_length(col) is True
    assert col.get_calls == 2


def test_finding7_autouse_conftest_blocks_model_loading_and_sets_rerank_off():
    """Finding 7 regression: unmocked tests have GRAPHIFY_RERANK=0 and model loading blocked without opt-in."""
    import reranker as reranker_mod

    assert os.environ.get("GRAPHIFY_RERANK") == "0"
    with pytest.raises(RuntimeError, match="Model loading is blocked in tests without explicit opt-in"):
        reranker_mod._load_reranker()
