"""Storage routing regressions; every Chroma client and embedder is synthetic."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock
from contextlib import nullcontext

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "db.py"


@pytest.fixture
def load_db(monkeypatch, tmp_path):
    chroma = ModuleType("chromadb")
    client = Mock()
    collection = client.get_or_create_collection.return_value
    chroma.PersistentClient = Mock(return_value=client)
    chroma.HttpClient = Mock(return_value=client)
    chroma.errors = SimpleNamespace(ChromaError=type("ChromaError", (Exception,), {}))
    shared = Mock()
    api = ModuleType("chromadb.api")
    api_client = ModuleType("chromadb.api.client")
    api_client.SharedSystemClient = shared
    embedders = ModuleType("embedders")
    embedding = object()
    embedders.build_chroma_embedding_function = Mock(return_value=embedding)
    for name, value in {"chromadb": chroma, "chromadb.api": api,
                        "chromadb.api.client": api_client, "embedders": embedders}.items():
        monkeypatch.setitem(sys.modules, name, value)
    for key in ("GRAPHIFY_STORAGE_MODE", "GRAPHIFY_DB_PATH", "GRAPHIFY_CHROMA_HOST", "GRAPHIFY_CHROMA_PORT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GRAPHIFY_DB_PATH", str(tmp_path / "synthetic-db"))
    source = compile(SOURCE.read_text(encoding="utf-8"), str(SOURCE), "exec")

    def load():
        module = ModuleType("isolated_graphify_db")
        module.__file__ = str(SOURCE)
        exec(source, module.__dict__)
        return module

    return SimpleNamespace(load=load, chroma=chroma, client=client, collection=collection,
                           shared=shared, embedding=embedding)


def test_local_default_and_path_override(load_db, monkeypatch):
    db = load_db.load()
    assert db.STORAGE_MODE == "local"
    assert Path(db.DB_PATH).name == "synthetic-db"
    load_db.chroma.PersistentClient.assert_called_once_with(path=db.DB_PATH)
    load_db.chroma.HttpClient.assert_not_called()
    load_db.client.get_or_create_collection.assert_called_once_with(
        "graphify", metadata={"hnsw:space": "cosine"}, embedding_function=load_db.embedding)
    monkeypatch.delenv("GRAPHIFY_DB_PATH")
    # Assert the original default path without stat'ing the real database.
    with monkeypatch.context() as patch:
        patch.setattr(Path, "exists", lambda _: False)
        default = load_db.load()
    assert default.DB_PATH == str(SOURCE.parent / "kg_db")


def test_local_mtime_still_reopens_and_stops_old_client(load_db, monkeypatch):
    db = load_db.load()
    monkeypatch.setattr(db, "get_db_mtime", lambda: 42.0)
    monkeypatch.setattr(db.time, "monotonic", lambda: db._last_check_time + 3)
    db.check_cache_freshness()
    assert load_db.chroma.PersistentClient.call_count == 2
    load_db.client._system.stop.assert_called_once()
    assert load_db.shared.clear_system_cache.call_count == 2
    assert db._last_mtime == 42.0


@pytest.mark.parametrize("host,expected", [(None, "127.0.0.1"), ("localhost", "127.0.0.1"), ("127.0.0.2", "127.0.0.2")])
def test_http_wiring_and_startup_only_settings(load_db, monkeypatch, host, expected):
    monkeypatch.setenv("GRAPHIFY_STORAGE_MODE", "http")
    if host is not None:
        monkeypatch.setenv("GRAPHIFY_CHROMA_HOST", host)
        monkeypatch.setenv("GRAPHIFY_CHROMA_PORT", "9123")
    db = load_db.load()
    kwargs = {"host": expected, "port": 8993 if host is None else 9123, "ssl": False}
    load_db.chroma.HttpClient.assert_called_once_with(**kwargs)
    load_db.client.get_or_create_collection.assert_called_once_with(
        "graphify", metadata={"hnsw:space": "cosine"}, embedding_function=load_db.embedding)
    monkeypatch.setenv("GRAPHIFY_STORAGE_MODE", "local")
    monkeypatch.setenv("GRAPHIFY_CHROMA_PORT", "9999")
    assert db.reopen() is load_db.collection
    load_db.chroma.HttpClient.assert_called_with(**kwargs)
    load_db.chroma.PersistentClient.assert_not_called()
    load_db.shared.clear_system_cache.assert_not_called()
    load_db.client._system.stop.assert_not_called()


@pytest.mark.parametrize("failure_at", ["connect", "collection"])
def test_http_failure_never_opens_local_database(load_db, monkeypatch, failure_at):
    monkeypatch.setenv("GRAPHIFY_STORAGE_MODE", "http")
    target = load_db.chroma.HttpClient if failure_at == "connect" else load_db.client.get_or_create_collection
    target.side_effect = OSError("server unavailable")
    with pytest.raises(OSError, match="server unavailable"):
        load_db.load()
    load_db.chroma.PersistentClient.assert_not_called()
    load_db.shared.clear_system_cache.assert_not_called()


def test_http_helpers_and_proxy_writes_never_touch_local_disk(load_db, monkeypatch):
    monkeypatch.setenv("GRAPHIFY_STORAGE_MODE", "http")
    db = load_db.load()
    forbidden = Mock(side_effect=AssertionError("local disk access"))
    monkeypatch.setattr(db, "Path", forbidden)
    monkeypatch.setattr(db.sqlite3, "connect", forbidden)
    assert db.get_db_mtime() == 0.0
    assert db.get_segment_seq_ids() == (None, None)
    assert db.get_sync_threshold() == 1000
    target = Mock()
    assert db.flush_index(target) is False
    assert not target.mock_calls
    db.notify_write()
    db.check_cache_freshness()
    assert db.get_collection() is load_db.collection
    assert db.get_client() is load_db.client
    db.col.add(ids=["synthetic"], embeddings=[[1.0]])
    load_db.collection.add.assert_called_once_with(ids=["synthetic"], embeddings=[[1.0]])
    db._col = None
    db.check_cache_freshness()
    load_db.chroma.PersistentClient.assert_not_called()
    load_db.shared.clear_system_cache.assert_not_called()
    load_db.client._system.stop.assert_not_called()


def test_http_preserves_embedder_compatibility_fallback(load_db, monkeypatch):
    monkeypatch.setenv("GRAPHIFY_STORAGE_MODE", "http")
    load_db.client.get_or_create_collection.side_effect = [ValueError("embedder mismatch"), load_db.collection]
    db = load_db.load()
    assert db.get_collection() is load_db.collection
    load_db.client.get_or_create_collection.assert_called_with("graphify", metadata={"hnsw:space": "cosine"})
    load_db.chroma.PersistentClient.assert_not_called()


@pytest.mark.parametrize("mode", ["", "remote", "HTTP"])
def test_unknown_mode_rejected_before_any_client(load_db, monkeypatch, mode):
    monkeypatch.setenv("GRAPHIFY_STORAGE_MODE", mode)
    with pytest.raises(ValueError, match="GRAPHIFY_STORAGE_MODE"):
        load_db.load()
    load_db.chroma.PersistentClient.assert_not_called()
    load_db.chroma.HttpClient.assert_not_called()


@pytest.mark.parametrize("port", ["", "0", "65536", "-1", "8993.0", "abc", " 8993", "８９９３"])
def test_bad_http_port_rejected(load_db, monkeypatch, port):
    monkeypatch.setenv("GRAPHIFY_STORAGE_MODE", "http")
    monkeypatch.setenv("GRAPHIFY_CHROMA_PORT", port)
    with pytest.raises(ValueError, match="GRAPHIFY_CHROMA_PORT"):
        load_db.load()
    load_db.chroma.PersistentClient.assert_not_called()
    load_db.chroma.HttpClient.assert_not_called()


@pytest.mark.parametrize("host", ["", "example.com", "0.0.0.0", "192.168.1.1", "http://127.0.0.1", "::1%eth0"])
def test_non_loopback_host_rejected(load_db, monkeypatch, host):
    monkeypatch.setenv("GRAPHIFY_STORAGE_MODE", "http")
    monkeypatch.setenv("GRAPHIFY_CHROMA_HOST", host)
    with pytest.raises(ValueError, match="GRAPHIFY_CHROMA_HOST"):
        load_db.load()
    load_db.chroma.PersistentClient.assert_not_called()
    load_db.chroma.HttpClient.assert_not_called()


def import_source(name):
    path = SOURCE.parent / (name + ".py")
    module = ModuleType("isolated_" + name)
    module.__file__ = str(path)
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    return module


@pytest.mark.parametrize("override", [False, True])
def test_fts_lock_and_pending_paths_match_selected_database(load_db, monkeypatch, override):
    if not override:
        monkeypatch.delenv("GRAPHIFY_DB_PATH")
    # Neither module opens its database at import time.
    keyword = import_source("keyword_index")
    writer = import_source("yozuv")
    expected = Path(load_db.load().DB_PATH) if override else SOURCE.parent / "kg_db"
    assert keyword.DB_PATH == expected / "keyword_index.sqlite3"
    assert writer.DB_DIR == expected
    assert writer.LOCK_FILE == expected / ".yozuv.lock"
    assert writer.PENDING_FILE == expected / ".pending_fts.json"


def test_keyword_rebuild_reads_through_selected_db_client(load_db, monkeypatch):
    fake_db = ModuleType("db")
    fake_db.get_collection = Mock(return_value=load_db.collection)
    monkeypatch.setitem(sys.modules, "db", fake_db)
    fake_writer = ModuleType("yozuv")
    fake_writer.lock = lambda: nullcontext()
    monkeypatch.setitem(sys.modules, "yozuv", fake_writer)
    monkeypatch.setattr(sys, "path", sys.path[:])
    keyword = import_source("keyword_index")
    index = MagicMock()
    monkeypatch.setattr(keyword, "KeywordIndex", Mock(return_value=index))
    load_db.collection.get.return_value = {"ids": ["one"], "documents": ["text"], "metadatas": [{"title": "test"}]}
    keyword.rebuild()
    fake_db.get_collection.assert_called_once_with()
    index.upsert.assert_called_once_with(["one"], ["text"], [{"title": "test"}])
    load_db.chroma.PersistentClient.assert_not_called()


def test_indexer_reads_through_selected_db_client(load_db, monkeypatch):
    fake_db = ModuleType("db")
    fake_db.get_collection = Mock(return_value=load_db.collection)
    fake_db.flush_index = Mock()
    monkeypatch.setitem(sys.modules, "db", fake_db)
    chunking = ModuleType("chunking")
    chunking.chunk_by_tokens = Mock()
    monkeypatch.setitem(sys.modules, "chunking", chunking)
    indexer = import_source("indexer")
    monkeypatch.setattr(indexer, "collect_files", lambda _: [])
    monkeypatch.setattr(sys, "argv", ["indexer.py", "synthetic"])
    indexer.main()
    fake_db.get_collection.assert_called_once_with()
    load_db.collection.count.assert_called_once_with()
    load_db.chroma.PersistentClient.assert_not_called()
