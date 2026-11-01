"""Tests for _ChromaEmbeddingFunctionAdapter, real Chroma PersistentClient regression, and degraded warning banner (A1, A2, A4)."""

from pathlib import Path
from unittest.mock import Mock, patch
import chromadb
from chromadb.api import types
import keyword_index
import numpy as np
import pytest
import search_core
import server
from embedders import EmbedderBase, _ChromaEmbeddingFunctionAdapter


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------
class DeterministicFakeEmbedder(EmbedderBase):
    """Deterministic fake embedder producing distinct 4-dim vectors for testing."""

    def __init__(self, normalize: bool = True):
        self.normalize = bool(normalize)

    def encode(self, texts: list) -> np.ndarray:
        out = []
        for t in texts:
            t_str = str(t).lower()
            if "alpha" in t_str or "first" in t_str:
                v = np.array([3.0, 0.0, 0.0, 4.0], dtype=np.float32)  # norm = 5.0
            elif "beta" in t_str or "second" in t_str:
                v = np.array([0.0, 6.0, 8.0, 0.0], dtype=np.float32)  # norm = 10.0
            elif "gamma" in t_str or "third" in t_str:
                v = np.array([0.0, 0.0, 0.0, 2.0], dtype=np.float32)  # norm = 2.0
            else:
                v = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)  # norm = 2.0

            if self.normalize:
                norm = float(np.linalg.norm(v))
                if norm > 0:
                    v = v / norm
            out.append(v)
        return np.array(out, dtype=np.float32)


# ---------------------------------------------------------------------------
# A1: _ChromaEmbeddingFunctionAdapter Chroma 1.5.x Protocol Tests
# ---------------------------------------------------------------------------
def test_a1_adapter_implements_chroma_embedding_function_protocol():
    """A1: _ChromaEmbeddingFunctionAdapter satisfies Chroma 1.5.x EmbeddingFunction protocol."""
    embedder = DeterministicFakeEmbedder(normalize=True)
    adapter = _ChromaEmbeddingFunctionAdapter(embedder, "fake")

    # Required methods and protocol attributes
    assert callable(adapter)
    assert hasattr(adapter, "embed_query") and callable(adapter.embed_query)
    assert hasattr(adapter, "embed_documents") and callable(adapter.embed_documents)
    assert hasattr(adapter, "name") and callable(adapter.name)
    assert hasattr(adapter, "is_legacy") and callable(adapter.is_legacy)
    assert hasattr(adapter, "default_space") and callable(adapter.default_space)
    assert hasattr(adapter, "supported_spaces") and callable(adapter.supported_spaces)

    assert adapter.name() == "graphify-fake"
    assert adapter.is_legacy() is True
    assert adapter.default_space() == "cosine"
    assert "cosine" in adapter.supported_spaces()

    # embed_query and embed_documents return same vectors as __call__
    texts = ["alpha document", "beta document"]
    res_call = adapter(texts)
    res_query = adapter.embed_query(texts)
    res_docs = adapter.embed_documents(texts)

    assert res_call == res_query == res_docs
    assert len(res_query) == 2
    assert len(res_query[0]) == 4


def test_a1_adapter_respects_normalize_flag():
    """A1: embed_query and embed_documents respect normalize flag on the underlying embedder."""
    # Normalized embedder produces unit vectors
    norm_embedder = DeterministicFakeEmbedder(normalize=True)
    norm_adapter = _ChromaEmbeddingFunctionAdapter(norm_embedder, "fake_norm")

    vec_query = norm_adapter.embed_query(["alpha document"])
    vec_docs = norm_adapter.embed_documents(["alpha document"])
    assert np.isclose(float(np.linalg.norm(vec_query[0])), 1.0, atol=1e-5)
    assert np.isclose(float(np.linalg.norm(vec_docs[0])), 1.0, atol=1e-5)

    # Unnormalized embedder preserves original magnitude
    raw_embedder = DeterministicFakeEmbedder(normalize=False)
    raw_adapter = _ChromaEmbeddingFunctionAdapter(raw_embedder, "fake_raw")

    raw_query = raw_adapter.embed_query(["alpha document"])
    raw_docs = raw_adapter.embed_documents(["alpha document"])
    assert np.isclose(float(np.linalg.norm(raw_query[0])), 5.0, atol=1e-5)
    assert np.isclose(float(np.linalg.norm(raw_docs[0])), 5.0, atol=1e-5)


def test_a1_adapter_handles_single_string_or_list_input():
    """A1: adapter accepts both single string and list of strings without error."""
    embedder = DeterministicFakeEmbedder(normalize=True)
    adapter = _ChromaEmbeddingFunctionAdapter(embedder, "fake")

    from_str = adapter.embed_query("alpha text")
    from_list = adapter.embed_query(["alpha text"])
    assert from_str == from_list
    assert len(from_str) == 1


# ---------------------------------------------------------------------------
# A2: Real Chroma PersistentClient Regression Test
# ---------------------------------------------------------------------------
def test_a2_real_chroma_persistent_client_query_and_hybrid_search(tmp_path, monkeypatch):
    """A2: PersistentClient with cosine collection and adapter queries without error and returns nearest doc;

    search_core.hybrid_search produces non-degraded result. Real Ollama is never called.
    """
    db_path = tmp_path / "real_chroma_db"
    kw_path = tmp_path / "kw_test.sqlite3"
    client = chromadb.PersistentClient(path=str(db_path))

    fake_embedder = DeterministicFakeEmbedder(normalize=True)
    adapter = _ChromaEmbeddingFunctionAdapter(fake_embedder, "fake")

    meta = {
        "schema_version": 2,
        "backend": "fake",
        "model": "fake-model",
        "model_digest": "sha256:fake_digest_abc",
        "dimension": 4,
        "distance": "cosine",
        "normalization": True,
        "hnsw:space": "cosine",
    }

    col = client.create_collection(
        name="graphify",
        metadata=meta,
        embedding_function=adapter,
    )

    docs = [
        "Python alpha asynchronous architecture",
        "Frontend beta user interface components",
        "Database gamma SQL query index tuning",
    ]
    metas = [
        {"id": "doc_alpha", "title": "Doc Alpha", "project": "test", "kind": "doc"},
        {"id": "doc_beta", "title": "Doc Beta", "project": "test", "kind": "doc"},
        {"id": "doc_gamma", "title": "Doc Gamma", "project": "test", "kind": "doc"},
    ]

    # Populate collection
    col.add(ids=["doc_alpha", "doc_beta", "doc_gamma"], documents=docs, metadatas=metas)

    # Initialize keyword index for hybrid search
    monkeypatch.setattr(keyword_index, "DB_PATH", kw_path)
    keyword_index._index_singleton = None
    fts = keyword_index.KeywordIndex()
    fts.upsert(
        ["doc_alpha", "doc_beta", "doc_gamma"],
        docs,
        metas,
    )

    # Mock Ollama calls to fail if invoked
    with patch("requests.post", side_effect=AssertionError("Real Ollama was called!")), \
         patch("urllib.request.urlopen", side_effect=AssertionError("Real Ollama urlopen was called!")):
        # 1. Direct collection.query(query_texts=[...]) must not throw embed_query AttributeError
        res_q = col.query(query_texts=["Python alpha asynchronous"], n_results=1)
        assert res_q["ids"][0][0] == "doc_alpha"

        # 2. search_core.hybrid_search must succeed WITHOUT degraded key
        res_hybrid = search_core.hybrid_search(
            "Python alpha asynchronous",
            top_k=3,
            where=None,
            collection=col,
            kw_index=fts,
        )

        assert res_hybrid["status"] == "ok"
        assert "degraded" not in res_hybrid
        assert "degraded_error" not in res_hybrid
        assert len(res_hybrid["results"]) > 0
        assert res_hybrid["results"][0]["id"] == "doc_alpha"

    fts.conn.close()
    if hasattr(client, "close"):
        client.close()


# ---------------------------------------------------------------------------
# A4: Degraded Search Banner in server.search_knowledge
# ---------------------------------------------------------------------------
def test_a4_search_knowledge_prepends_banner_when_degraded(monkeypatch):
    """A4: When search_core.hybrid_search returns degraded, search_knowledge prepends the warning banner."""
    fake_degraded_result = {
        "status": "ok",
        "results": [
            {
                "id": "doc_lex",
                "text": "Leksik qidiruv orqali topilgan muhim hujjat matni.",
                "similarity": 0.0,
                "meta": {
                    "id": "doc_lex",
                    "title": "Leksik Hujjat",
                    "project": "test",
                    "source": "manual.md",
                    "date": "2026-02-01",
                    "kind": "doc",
                },
            }
        ],
        "related_candidates": [],
        "degraded": "vector_search_failed",
        "degraded_error": "Connection refused to vector service",
    }

    monkeypatch.setattr(search_core, "hybrid_search", lambda *a, **kw: fake_degraded_result)
    monkeypatch.setattr(server._col, "count", lambda: 1)
    monkeypatch.setattr(server, "_record_search_savings", lambda *a, **kw: None)
    monkeypatch.setattr(keyword_index, "get_index", lambda: None)

    res_text = server.search_knowledge("muhim hujjat", project="test")

    expected_banner = "⚠️ Vektor qidiruv ishlamadi (Connection refused to vector service), natija faqat kalit so'z bo'yicha."
    assert res_text.startswith(expected_banner)
    assert "Leksik Hujjat" in res_text
    assert "Leksik qidiruv orqali topilgan muhim hujjat matni." in res_text


def test_a4_search_knowledge_normal_has_no_warning_banner(monkeypatch):
    """A4: Normal search without degradation must NOT include warning banner."""
    fake_normal_result = {
        "status": "ok",
        "results": [
            {
                "id": "doc_norm",
                "text": "Oddiy qidiruv natijasi.",
                "similarity": 0.85,
                "meta": {
                    "id": "doc_norm",
                    "title": "Oddiy Hujjat",
                    "project": "test",
                    "source": "normal.md",
                    "date": "2026-02-01",
                    "kind": "doc",
                },
            }
        ],
        "related_candidates": [],
    }

    monkeypatch.setattr(search_core, "hybrid_search", lambda *a, **kw: fake_normal_result)
    monkeypatch.setattr(server._col, "count", lambda: 1)
    monkeypatch.setattr(server, "_record_search_savings", lambda *a, **kw: None)
    monkeypatch.setattr(keyword_index, "get_index", lambda: None)

    res_text = server.search_knowledge("oddiy", project="test")

    assert "⚠️ Vektor qidiruv ishlamadi" not in res_text
    assert "Oddiy Hujjat" in res_text
