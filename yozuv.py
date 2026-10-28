import os
import sys
import json
import time
import argparse
from pathlib import Path
from contextlib import contextmanager

DB_DIR = Path(os.environ.get("GRAPHIFY_DB_PATH", str(Path(__file__).parent / "kg_db")))
LOCK_FILE = DB_DIR / ".yozuv.lock"
PENDING_FILE = DB_DIR / ".pending_fts.json"

@contextmanager
def lock():
    DB_DIR.mkdir(parents=True, exist_ok=True)
    f = open(LOCK_FILE, 'w', encoding='utf-8')
    try:
        if sys.platform == "win32":
            import msvcrt
            start_time = time.monotonic()
            while True:
                try:
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.monotonic() - start_time > 30:
                        raise RuntimeError("Graphify bazasini qulflashda taymaut (boshqa jarayon bazani uzoq ushlab turibdi).")
                    time.sleep(0.1)
        else:
            import fcntl
            start_time = time.monotonic()
            while True:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError:
                    if time.monotonic() - start_time > 30:
                        raise RuntimeError("Graphify bazasini qulflashda taymaut (boshqa jarayon bazani uzoq ushlab turibdi).")
                    time.sleep(0.1)
        yield
    finally:
        if sys.platform == "win32":
            import msvcrt
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        else:
            import fcntl
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        f.close()

def _add_pending(ids):
    try:
        pending = set()
        if PENDING_FILE.exists():
            with open(PENDING_FILE, "r", encoding="utf-8") as f:
                pending = set(json.load(f))
        pending.update(ids)
        with open(PENDING_FILE, "w", encoding="utf-8") as f:
            json.dump(list(pending), f)
    except Exception as e:
        print(f"[yozuv] pending faylga yozishda xato: {e}", file=sys.stderr)

def upsert(col, ids, documents, metadatas, embeddings=None):
    if not ids:
        return
    with lock():
        kwargs = {"ids": ids, "documents": documents, "metadatas": metadatas}
        if embeddings is not None:
            kwargs["embeddings"] = embeddings
        col.upsert(**kwargs)
        
        try:
            import keyword_index
            keyword_index.get_index().upsert(ids=ids, texts=documents, metadatas=metadatas)
        except Exception as e:
            print(f"[yozuv] FTS ga yozishda xato: {e}", file=sys.stderr)
            _add_pending(ids)

def delete(col, ids=None, where=None):
    if ids is None and where is None:
        return
    
    with lock():
        # Avval o'chiriladigan id larni olamiz
        del_ids = ids if ids else []
        if where:
            # try/except ni olib tashlaymiz, xato chaqiruvchiga uzatiladi
            res = col.get(where=where, include=[])
            if res and res.get("ids"):
                del_ids = list(set(del_ids + res["ids"]))
        
        if not del_ids:
            return
            
        col.delete(ids=del_ids)
        
        try:
            import keyword_index
            keyword_index.get_index().delete(ids=del_ids)
        except Exception as e:
            print(f"[yozuv] FTS dan o'chirishda xato: {e}", file=sys.stderr)
            _add_pending(del_ids)

def reconcile(col, dry_run=False):
    with lock():
        import keyword_index
        kw = keyword_index.get_index()
        
        # 1. Barcha Chroma idlarini olish
        chroma_ids = set()
        chroma_res = col.get(include=[])
        if chroma_res and chroma_res.get("ids"):
            chroma_ids = set(chroma_res["ids"])
            
        # 2. Barcha FTS idlarini olish
        fts_ids = set()
        with kw.conn:
            rows = kw.conn.execute("SELECT id FROM docs").fetchall()
            for row in rows:
                fts_ids.add(row["id"])
                
        # 3. Farqlarni topish
        missing_in_fts = set(chroma_ids - fts_ids)
        excess_in_fts = list(fts_ids - chroma_ids)
        
        # Pending id'larni ham yangilash ro'yxatiga qo'shish
        pending_ids = set()
        if PENDING_FILE.exists():
            try:
                with open(PENDING_FILE, "r", encoding="utf-8") as f:
                    pending_ids = set(json.load(f))
            except Exception:
                pass
                
        # Faqat Chroma'da mavjud pending id'larni yangilaymiz
        pending_to_update = pending_ids.intersection(chroma_ids)
        missing_in_fts.update(pending_to_update)
        missing_in_fts = list(missing_in_fts)
        
        total_missing = len(missing_in_fts)
        total_excess = len(excess_in_fts)
        diff_count = total_missing + total_excess
        
        if dry_run:
            return diff_count
            
        # FTS dan ortiqchalarni o'chirish
        batch_size = 500
        for i in range(0, total_excess, batch_size):
            batch = excess_in_fts[i:i+batch_size]
            kw.delete(ids=batch)
            
        # FTS ga yetishmaganini qo'shish
        for i in range(0, total_missing, batch_size):
            batch = missing_in_fts[i:i+batch_size]
            res = col.get(ids=batch, include=["documents", "metadatas"])
            if res and res.get("ids"):
                kw.upsert(ids=res["ids"], texts=res["documents"], metadatas=res["metadatas"])
                
        # pending ni tozalash (faqat muvaffaqiyatli sinxronizatsiyadan so'ng)
        if PENDING_FILE.exists():
            try:
                PENDING_FILE.unlink()
            except OSError:
                pass
                
        return diff_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--moslash", action="store_true", help="Chroma va FTS orasidagi farqlarni tuzatish")
    parser.add_argument("--quruq", action="store_true", help="Faqat tekshirish, o'zgartirmaslik")
    args = parser.parse_args()
    
    if args.moslash:
        import db
        count = reconcile(db.get_collection(), dry_run=args.quruq)
        if args.quruq:
            print(f"Topilgan farqlar soni: {count}")
        else:
            print(f"Tuzatilgan farqlar soni: {count}")
