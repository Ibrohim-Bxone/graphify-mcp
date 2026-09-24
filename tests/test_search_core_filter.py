import pytest
from unittest.mock import patch, MagicMock

from test_yozuv import temp_yozuv

def test_search_core_filter(temp_yozuv):
    import search_core
    import keyword_index
    import yozuv
    
    col, db_dir = temp_yozuv
    
    # Insert id1 and id2 to both
    docs = ["Test doc one", "Test doc two"]
    metas = [{"id": "id1", "title": "t1"}, {"id": "id2", "title": "t2"}]
    yozuv.upsert(col, ["id1", "id2"], docs, metas)
    
    # Manually insert id3 to FTS only
    fts = keyword_index.get_index()
    fts.upsert(["id3"], ["Test doc three"], [{"id": "id3", "title": "ghost"}])
    
    res = search_core.hybrid_search(
        query="Test",
        collection=col,
        kw_index=fts,
        top_k=2,
        where=None
    )
    
    if isinstance(res, dict) and "results" in res:
        res = res["results"]
        
    for r in res:
        assert r["id"] != "id3"
        
    assert len(res) == 2
