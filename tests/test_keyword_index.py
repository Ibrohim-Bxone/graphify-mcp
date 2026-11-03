import pytest
import keyword_index as ki

DOCS = [
    ("d1", "mushuk va it birga o'ynaydi", {}),
    ("d2", "faqat mushuk bor bu yerda", {}),
    ("d3", "faqat it bor bu yerda", {}),
    ("d4", "baliq va qush haqida gap", {}),
]


def _make_index(tmp_path, monkeypatch, filename, force_fts5):
    monkeypatch.setattr(ki, "DB_PATH", tmp_path / filename)
    idx = ki.KeywordIndex.__new__(ki.KeywordIndex)
    idx.conn = ki._get_conn()
    idx.use_fts5 = force_fts5
    idx._init_db()
    return idx


def test_fts5_and_python_bm25_return_same_doc_set(tmp_path, monkeypatch):
    """FTS5 MATCH va sof Python BM25 bir xil OR mantiqda ishlashi kerak:
    ikkalasi ham so'rovdagi kamida bitta so'z uchraydigan hujjatlarni topsin
    (AND emas). Aks holda ikki yo'l boshqa-boshqa hujjat to'plamini qaytaradi."""
    monkeypatch.setattr(ki, "DB_PATH", tmp_path / "probe.sqlite3")
    probe_conn = ki._get_conn()
    fts5_available = ki._check_fts5(probe_conn)
    probe_conn.close()
    if not fts5_available:
        pytest.skip("Bu muhitdagi sqlite3 FTS5'ni qo'llab-quvvatlamaydi")

    ids = [d[0] for d in DOCS]
    texts = [d[1] for d in DOCS]
    metas = [d[2] for d in DOCS]

    fts_idx = _make_index(tmp_path, monkeypatch, "fts.sqlite3", True)
    fts_idx.upsert(ids, texts, metas)
    fts_results = {r["id"] for r in fts_idx.search("mushuk it", limit=50)}

    py_idx = _make_index(tmp_path, monkeypatch, "py.sqlite3", False)
    py_idx.upsert(ids, texts, metas)
    py_results = {r["id"] for r in py_idx.search("mushuk it", limit=50)}

    # d1: ikkala so'z ham bor, d2: faqat "mushuk", d3: faqat "it" -> OR bo'lgani
    # uchun uchalasi ham topilishi kerak. d4 hech biriga mos kelmaydi.
    expected = {"d1", "d2", "d3"}
    assert fts_results == expected
    assert py_results == expected


def test_search_on_empty_index_warns_and_returns_empty(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ki, "DB_PATH", tmp_path / "empty.sqlite3")
    idx = ki.KeywordIndex()
    assert idx.search("hech narsa yo'q") == []
    assert "bo'sh" in capsys.readouterr().err
