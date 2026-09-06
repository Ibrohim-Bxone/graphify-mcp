import pytest
import os
import sys
import subprocess
from pathlib import Path
import json
import chromadb

def test_indeks_tekshir_detects_errors(tmp_path, monkeypatch):
    # Mock environment variables for the test
    monkeypatch.setenv("GRAPHIFY_DIR", str(tmp_path / "Graphify"))
    monkeypatch.setenv("ARCHIVE_DIR", str(tmp_path / "Promtlarim" / "instagram"))
    
    graphify_dir = tmp_path / "Graphify"
    archive_dir = tmp_path / "Promtlarim" / "instagram"
    
    graphify_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    # Create synthetic archive files
    def make_record(sc, usable, items_count):
        items = [{"kind": "prompt", "subtype": "test", "name_en": f"I{i}", "content": "c"} for i in range(items_count)]
        return {
            "shortcode": sc,
            "usable": usable,
            "items": items
        }
    
    def render(record):
        sc = record["shortcode"]
        lines = [
            "---",
            f"shortcode: {sc}",
            f"usable: {str(record.get('usable')).lower()}",
            "---",
            "<!-- graphify-record\n```json\n" + json.dumps(record) + "\n```\n-->"
        ]
        return "\n".join(lines)
        
    (archive_dir / "sc1_not_indexed.md").write_text(render(make_record("sc1", True, 1)))
    (archive_dir / "sc2_false_indexed.md").write_text(render(make_record("sc2", False, 1)))
    (archive_dir / "sc3_orphan.md").write_text(render(make_record("sc3", True, 1))) # We will add item:0 and item:1 (orphan) to DB
    # We will also add 'sc4_missing_archive' to DB without archive file
    (archive_dir / "sc5_duplicate.md").write_text(render(make_record("sc5", True, 1)))
    
    # Build DB
    client = chromadb.PersistentClient(path=str(graphify_dir / "kg_db"))
    col = client.create_collection("graphify")
    
    # Add sc2 (should not be here because usable=false)
    col.add(ids=["ig:sc2:item:0"], documents=["d"], metadatas=[{"shortcode": "sc2"}])
    
    # Add sc3 with an extra item
    col.add(ids=["ig:sc3:item:0", "ig:sc3:item:1"], documents=["d", "d"], metadatas=[{"shortcode": "sc3"}, {"shortcode": "sc3"}])
    
    # Add sc4 (no archive)
    col.add(ids=["ig:sc4:item:0"], documents=["d"], metadatas=[{"shortcode": "sc4"}])
    
    # Add sc5 as ig and as file duplicate
    sc5_path = str((archive_dir / "sc5_duplicate.md").resolve())
    col.add(ids=["ig:sc5:item:0", "file_1234"], documents=["d", "d"], 
            metadatas=[{"shortcode": "sc5"}, {"source": sc5_path}])
            
    # Add a long chunk
    col.add(ids=["file_5678"], documents=["long"], metadatas=[{"token_count": 300, "source": "other.md"}])
    
    import runpy
    sys.argv = ["indeks_tekshir.py", "--json"]
    
    # We must patch sys.path so it can find 'Instagram new Ideas/src'
    # Actually, the script uses __file__.parent.parent... it relies on the real file path.
    # We will just run the script using subprocess with our mocked ENV vars.

    import os
    env = os.environ.copy()
    env["GRAPHIFY_DIR"] = str(graphify_dir)
    env["ARCHIVE_DIR"] = str(archive_dir)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent.parent) # Root
    
    script_path = Path(__file__).resolve().parent.parent / "indeks_tekshir.py"
    
    res = subprocess.run([sys.executable, str(script_path), "--json"], env=env, capture_output=True, text=True)
    
    # Now verify output
    out = json.loads(res.stdout)
    
    assert "sc1" in out["usable_true_no_index"]
    assert "sc2" in out["usable_false_has_index"]
    assert "ig:sc3:item:1" in out["orphan_items"]
    assert "sc4" in out["missing_archive_has_index"]
    assert "file_1234" in out["duplicate_file_chunks"]
    assert "file_5678" in out["long_chunks"]
