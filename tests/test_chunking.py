import pytest
from chunking import chunk_by_tokens, MAX_TOKENS, PREFIX_BUDGET, count_tokens

def test_chunking_empty():
    assert chunk_by_tokens("") == []
    assert chunk_by_tokens("   ") == []

def test_chunking_short():
    res = chunk_by_tokens("Short text", parent_id="test_id")
    assert len(res) == 1
    assert res[0]["text"] == "Short text"
    assert res[0]["parent_id"] == "test_id"
    assert res[0]["chunk_index"] == 0
    assert res[0]["chunk_total"] == 1
    assert res[0]["token_count"] == count_tokens("Short text")
    assert res[0]["token_count"] <= MAX_TOKENS - PREFIX_BUDGET - 2

def test_chunking_long():
    text = "Word " * 500
    res = chunk_by_tokens(text)
    assert len(res) > 1
    for r in res:
        assert r["token_count"] <= MAX_TOKENS - PREFIX_BUDGET - 2
        
def test_chunking_overlap():
    text = "Word " * 500
    res = chunk_by_tokens(text, overlap_tokens=40)
    assert len(res) > 1
    # Check if there is some overlap between consecutive chunks
    # Since they are words, the end of chunk i should be in the beginning of chunk i+1
    for i in range(len(res) - 1):
        c1 = res[i]["text"]
        c2 = res[i+1]["text"]
        
        words1 = set(c1.split())
        words2 = set(c2.split())
        assert len(words1.intersection(words2)) > 0, "No overlap found!"

def test_chunking_last_info_preserved():
    # ~2000 tokenli matn oxiriga noyob ibora qo'yilsin
    text = "Filler word " * 2000 + "NOYOB_IBORA_123"
    res = chunk_by_tokens(text)
    assert len(res) > 1
    found = False
    for r in res:
        assert r["token_count"] <= MAX_TOKENS - PREFIX_BUDGET - 2
        if "NOYOB_IBORA_123" in r["text"]:
            found = True
    assert found, "Oxiridagi noyob ibora yo'qolib qoldi!"
    

