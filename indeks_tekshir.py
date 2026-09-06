import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

os.environ.setdefault("GRAPHIFY_DIR", str(ROOT / "Graphify"))
os.environ.setdefault("ARCHIVE_DIR", str(ROOT / "Promtlarim" / "instagram"))

sys.path.insert(0, str(ROOT / "Instagram new Ideas"))
from src import archive, kb
import chromadb

def audit_index(db_path: Path):
    client = chromadb.PersistentClient(path=str(db_path))
    col = client.get_collection("graphify")

    archive_records = {}
    for p, rec in archive.all_records():
        if "shortcode" in rec:
            archive_records[rec["shortcode"]] = rec
    
    res = col.get(include=["metadatas"])
    
    index_chunks = {}
    if res and res.get("ids"):
        for i, cid in enumerate(res["ids"]):
            meta = res["metadatas"][i] if res.get("metadatas") else {}
            index_chunks[cid] = meta

    errors = {
        "usable_true_no_index": [],
        "usable_false_has_index": [],
        "orphan_items": [],
        "missing_archive_has_index": [],
        "duplicate_file_chunks": [],
        "long_chunks": []
    }

    ig_chunks_by_sc = {}
    for cid, meta in index_chunks.items():
        if cid.startswith("ig:"):
            parts = cid.split(":")
            if len(parts) >= 2:
                sc = parts[1]
                ig_chunks_by_sc.setdefault(sc, []).append((cid, meta))
        if meta.get("token_count", 0) > 256 or meta.get("est_tokens", 0) > 256:
            errors["long_chunks"].append(cid)

    for sc, rec in archive_records.items():
        usable = rec.get("usable", False)
        has_ig = sc in ig_chunks_by_sc
        
        if usable and not has_ig:
            errors["usable_true_no_index"].append(sc)
        elif not usable and has_ig:
            errors["usable_false_has_index"].append(sc)
            
        if has_ig:
            items = rec.get("items", [])
            for cid, meta in ig_chunks_by_sc[sc]:
                if ":item:" in cid:
                    parts = cid.split(":item:")
                    if len(parts) == 2:
                        try:
                            idx_str = parts[1].split(":")[0]
                            idx = int(idx_str)
                            if idx >= len(items):
                                errors["orphan_items"].append(cid)
                        except ValueError:
                            pass

    for sc in ig_chunks_by_sc:
        if sc not in archive_records:
            errors["missing_archive_has_index"].append(sc)

    archive_file_paths = set(str(p.resolve()) for p, _ in archive.all_records())
    
    for cid, meta in index_chunks.items():
        if cid.startswith("file_"):
            source = meta.get("source", "")
            if source:
                try:
                    if str(Path(source).resolve()) in archive_file_paths:
                        errors["duplicate_file_chunks"].append(cid)
                except Exception:
                    pass
                    
    return errors

def main():
    parser = argparse.ArgumentParser(description="Audit and repair index")
    parser.add_argument("--tuzat", action="store_true", help="Repair the index")
    parser.add_argument("--json", action="store_true", help="Output in JSON")
    args = parser.parse_args()

    db_path = Path(os.environ["GRAPHIFY_DIR"]) / "kg_db"
    errors = audit_index(db_path)

    if args.json:
        print(json.dumps(errors, indent=2))
    else:
        print("=== Indeks Audit Hisoboti ===")
        print(f"1. Usable=true, lekin indekslanmagan: {len(errors['usable_true_no_index'])}")
        print(f"2. Usable=false, lekin indeksda mavjud: {len(errors['usable_false_has_index'])}")
        print(f"3. Yetim bo'laklar (orphan items): {len(errors['orphan_items'])}")
        print(f"4. Arxivda yo'q, lekin indeksda bor shortcode'lar: {len(errors['missing_archive_has_index'])}")
        print(f"5. Dublikat arxiv fayl-chunk'lari: {len(errors['duplicate_file_chunks'])}")
        print(f"6. 256 tokendan uzun chunk'lar: {len(errors['long_chunks'])}")

    if args.tuzat:
        client = chromadb.PersistentClient(path=str(db_path))
        col = client.get_collection("graphify")
        print("\nTa'mirlash boshlandi...")
        
        archive_records = {}
        for p, rec in archive.all_records():
            if "shortcode" in rec:
                archive_records[rec["shortcode"]] = rec
                
        to_upsert_sc = set(errors["usable_true_no_index"])
        to_upsert_sc.update(errors["usable_false_has_index"])
        
        for cid in errors["orphan_items"]:
            sc = cid.split(":")[1]
            to_upsert_sc.add(sc)
            
        for sc in errors["missing_archive_has_index"]:
            kb.delete_shortcode(sc)
            print(f"O'chirildi (arxivda yo'q): {sc}")
            
        if errors["duplicate_file_chunks"]:
            col.delete(ids=errors["duplicate_file_chunks"])
            print(f"O'chirildi (dublikat fayl-chunk): {len(errors['duplicate_file_chunks'])} ta")
            
        for sc in to_upsert_sc:
            rec = archive_records.get(sc)
            if rec:
                kb.upsert_record(rec)
                print(f"Qayta ishlandi (upsert): {sc}")

        print("Ta'mirlash tugadi.")

if __name__ == "__main__":
    main()
