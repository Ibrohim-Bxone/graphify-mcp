import os
import json
import pytest
import chromadb
from pathlib import Path
import yozuv
import keyword_index

@pytest.fixture
def temp_yozuv(tmp_path, monkeypatch):
    db_dir = tmp_path / "kg_db"
    db_dir.mkdir()
    
    # Patch yozuv files
    monkeypatch.setattr(yozuv, "DB_DIR", db_dir)
    monkeypatch.setattr(yozuv, "LOCK_FILE", db_dir / ".yozuv.lock")
    monkeypatch.setattr(yozuv, "PENDING_FILE", db_dir / ".pending_fts.json")
    
    # Patch fts db
    monkeypatch.setattr(keyword_index, "DB_PATH", db_dir / "keyword_index.sqlite3")
    keyword_index._index_singleton = None # Reset singleton
    
    # Create temp Chroma
    client = chromadb.EphemeralClient()
    col = client.get_or_create_collection("test_col", metadata={"hnsw:space": "cosine"})
    
    yield col, db_dir
    
    keyword_index._index_singleton = None
    try:
        client.delete_collection("test_col")
    except Exception:
        pass

def test_yozuv_lock(temp_yozuv):
    col, db_dir = temp_yozuv
    
    with yozuv.lock():
        assert yozuv.LOCK_FILE.exists()
        
    # Lock is released but file exists
    assert yozuv.LOCK_FILE.exists()

def test_yozuv_upsert_delete(temp_yozuv):
    col, db_dir = temp_yozuv
    
    docs = ["Doc 1", "Doc 2"]
    metas = [{"id": "1", "title": "t1"}, {"id": "2", "title": "t2"}]
    
    yozuv.upsert(col, ["1", "2"], docs, metas)
    
    # Check chroma
    res = col.get()
    assert len(res["ids"]) == 2
    
    # Check FTS
    fts = keyword_index.get_index()
    fts_res = fts.search("Doc")
    assert len(fts_res) == 2
    
    yozuv.delete(col, ["1"])
    
    # Check chroma
    res = col.get()
    assert res["ids"] == ["2"]
    
    # Check FTS
    fts_res = fts.search("Doc")
    assert len(fts_res) == 1
    assert fts_res[0]["id"] == "2"

def test_yozuv_reconcile(temp_yozuv):
    col, db_dir = temp_yozuv
    
    # Manually add to chroma only (simulate FTS failure)
    col.add(ids=["3"], documents=["Missing in FTS"], metadatas=[{"id": "3", "title": "orphan"}])
    
    # Manually add to FTS only
    fts = keyword_index.get_index()
    fts.upsert(["4"], ["Missing in Chroma"], [{"id": "4", "title": "ghost"}])
    
    assert col.count() == 1
    fts_res1 = fts.search("Missing")
    assert len(fts_res1) == 1
    assert fts_res1[0]["id"] == "4"
    
    # Run reconcile
    yozuv.reconcile(col, dry_run=False)
    
    # Now FTS should have 3 and NOT have 4
    fts_res2 = fts.search("Missing")
    assert len(fts_res2) == 1
    assert fts_res2[0]["id"] == "3"

def test_yozuv_reconcile_dry_run(temp_yozuv):
    col, db_dir = temp_yozuv
    
    # Manually add to chroma only (simulate FTS failure)
    col.add(ids=["3"], documents=["Missing in FTS"], metadatas=[{"id": "3", "title": "orphan"}])
    
    # Manually add to FTS only
    fts = keyword_index.get_index()
    fts.upsert(["4"], ["Missing in Chroma"], [{"id": "4", "title": "ghost"}])
    
    # Run reconcile in dry-run mode
    diff_count = yozuv.reconcile(col, dry_run=True)
    assert diff_count == 2
    
    # Check that FTS is NOT modified
    fts_res = fts.search("Missing")
    assert len(fts_res) == 1
    assert fts_res[0]["id"] == "4"

