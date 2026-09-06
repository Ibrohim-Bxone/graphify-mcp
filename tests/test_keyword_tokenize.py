import keyword_index


def test_tokenize_suffix_variants_share_root():
    root_tokens = [
        keyword_index.tokenize("kitob"),
        keyword_index.tokenize("kitoblar"),
        keyword_index.tokenize("kitobning"),
        keyword_index.tokenize("kitobdan"),
        keyword_index.tokenize("kitobga"),
    ]
    assert all(tokens == ["kitob"] for tokens in root_tokens)


def test_tokenize_short_words_not_stripped():
    assert keyword_index.tokenize("bor") == ["bor"]
    assert keyword_index.tokenize("bir") == ["bir"]
    assert keyword_index.tokenize("van") == ["van"]


def test_tokenize_unsuffixed_words_unchanged():
    assert keyword_index.tokenize("Dify") == ["dify"]
    assert keyword_index.tokenize("customer") == ["customer"]


def test_keyword_index_roundtrip_with_suffixed_query(tmp_path, monkeypatch):
    monkeypatch.setattr(keyword_index, "DB_PATH", tmp_path / "keyword_index.sqlite3")
    idx = keyword_index.KeywordIndex()
    idx.upsert(
        ids=["doc1"],
        texts=["Narxlar bo'yicha savol"],
        metadatas=[{"project": "test", "kind": "note"}],
    )
    results = idx.search("narx")
    assert any(r["id"] == "doc1" for r in results)
