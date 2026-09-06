import argparse
import hashlib
import os
import sys
import json
import multiprocessing
import time
import random
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

os.environ.setdefault("GRAPHIFY_DIR", str(ROOT / "Graphify"))
os.environ.setdefault("ARCHIVE_DIR", str(ROOT / "Promtlarim" / "instagram"))

sys.path.insert(0, str(ROOT / "Instagram new Ideas"))
from src import archive, kb

sys.path.insert(0, str(ROOT / "Graphify"))
import chromadb
from chunking import chunk_by_tokens, count_tokens
import re


def init_worker():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['ORT_NUM_THREADS'] = '1'
    global worker_embed_fn
    from embedders import build_chroma_embedding_function
    worker_embed_fn = build_chroma_embedding_function()
    if worker_embed_fn is None:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        worker_embed_fn = DefaultEmbeddingFunction()

def embed_batch(batch):
    global worker_embed_fn
    return worker_embed_fn(batch)

def normalize_source(src) -> str:
    if not src:
        return ""
    s = str(src).strip()
    if s == "save_memory" or s.startswith("http://") or s.startswith("https://") or s.startswith("instagram-file:"):
        return s.lower()
    try:
        p = Path(s)
        if not p.is_absolute():
            p = ROOT / p
        return str(p.resolve()).lower()
    except Exception:
        return s.lower()

def has_shortcode_frontmatter(text: str) -> bool:
    if not text.startswith("---"):
        return False
    end = text.find("\n---", 3)
    if end == -1:
        return False
    frontmatter = text[3:end]
    return bool(re.search(r'^shortcode:', frontmatter, re.MULTILINE))

def collect_files(base_paths):
    EXTS = {".md", ".txt", ".rst"}
    for p in base_paths:
        p = Path(p)
        if p.is_file() and p.suffix.lower() in EXTS:
            yield p
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and f.suffix.lower() in EXTS and ".venv" not in f.parts and "kg_db" not in f.parts and "kg_db-yangi" not in f.parts:
                    yield f

def check_backup(graphify_dir):
    zaxira = graphify_dir / "kg_db-zaxira-S-9e795f"
    if zaxira.exists():
        return True
    return False

def take_backup(graphify_dir, kg_db_path):
    zaxira = graphify_dir / "kg_db-zaxira-S-9e795f"
    print(f"Zaxira ({zaxira.name}) topilmadi, olinmoqda...")
    # shutil.copytree ishlatamiz, chunki shutil.move butun boshli zaxirani xavf ostiga qo'yadi.
    import shutil
    shutil.copytree(str(kg_db_path), str(zaxira))
    print("Zaxira olindi.")

def get_fingerprint(db_path: Path, chunk_count: int) -> str:
    start = time.time()
    
    sqlite_path = db_path / "chroma.sqlite3"
    sqlite_hash = "no-sqlite"
    if sqlite_path.exists():
        h = hashlib.sha256()
        with open(sqlite_path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
        sqlite_hash = h.hexdigest()
        
    final_fp = f"{chunk_count}-{sqlite_hash}"
    duration = time.time() - start
    print(f"Barmoq izi hisoblandi (vaqt: {duration:.2f} s)")
    return final_fp

def check_wal_and_checkpoint(db_path: Path):
    sqlite_path = db_path / "chroma.sqlite3"
    if not sqlite_path.exists():
        return
    import sqlite3
    conn = sqlite3.connect(str(sqlite_path))
    try:
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode;")
        mode = cursor.fetchone()[0].lower()
        if mode == "wal":
            cursor.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            res = cursor.fetchone()
            if res is None or res != (0, 0, 0):
                print(f"Xato: {db_path.name} fayli boshqa jarayon tomonidan band yoki checkpoint to'liq emas (res={res}).")
                sys.exit(1)
    except Exception as e:
        print(f"Xato: sqlite ulanishda muammo {db_path.name} - {e}")
        sys.exit(1)
    finally:
        conn.close()

def almashtir(graphify_dir):
    kg_db = graphify_dir / "kg_db"
    kg_db_yangi = graphify_dir / "kg_db-yangi"
    
    if not kg_db_yangi.exists():
        print("Xato: kg_db-yangi topilmadi. Avval indeksni qayta quring.")
        sys.exit(1)
        
    tekshiruv_file = kg_db_yangi / "_tekshiruv.json"
    if not tekshiruv_file.exists():
        print("Xato: _tekshiruv.json topilmadi. Avval indeksni qayta quring.")
        sys.exit(1)
        
    try:
        report = json.loads(tekshiruv_file.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Xato: _tekshiruv.json o'qib bo'lmadi: {e}")
        sys.exit(1)
        
    for key in ["ok", "sana", "xatolar", "yangi_db_barmoq_izi"]:
        if key not in report:
            print(f"Xato: _tekshiruv.json da kutilgan kalit '{key}' yo'q.")
            sys.exit(1)
            
    if report.get("ok") is not True:
        print("Xato: Tekshiruv xatolar bilan tugagan. Almashtirish taqiqlanadi.")
        sys.exit(1)
        
    if not check_backup(graphify_dir):
        take_backup(graphify_dir, kg_db)

    print("!!! DIQQAT !!!")
    print("1. MCP server (Claude Code), bot, dashboard, HTTP server — HAMMASI to'xtatiladi")
    print("2. python indeks-qayta-qur.py --almashtir")
    print("3. Claude Code va bot qayta ishga tushiriladi")
    print("\nOgohlantirish: SQLite checkpoint Chroma'ning HNSW indeksini flush qilmaydi, va muvaffaqiyatli checkpoint barcha jarayonlar to'xtaganini ISBOTLAMAYDI — bu operatorning javobgarligi.")

    check_wal_and_checkpoint(kg_db)
    check_wal_and_checkpoint(kg_db_yangi)
    
    client_new = chromadb.PersistentClient(path=str(kg_db_yangi))
    col_new = client_new.get_collection("graphify")
    total_new = col_new.count()
    current_fp = get_fingerprint(kg_db_yangi, total_new)
    
    if current_fp != report.get("yangi_db_barmoq_izi"):
        print("Xato: kg_db-yangi barmoq izi mos emas (tekshiruvdan keyin o'zgargan).")
        sys.exit(1)
        
    sana = datetime.now().strftime("%Y%m%d-%H%M%S")
    eski_nom = f"kg_db-eski-{sana}"
    kg_db_eski = graphify_dir / eski_nom
    
    print(f"\nJarayon:")
    print(f"1. {kg_db.name} -> {kg_db_eski.name}")
    try:
        os.rename(str(kg_db), str(kg_db_eski))
    except Exception as e:
        print(f"Xato: {e}")
        sys.exit(1)
        
    print(f"2. {kg_db_yangi.name} -> {kg_db.name}")
    try:
        os.rename(str(kg_db_yangi), str(kg_db))
    except Exception as e:
        print(f"Xato: {e}. Orqaga qaytarilmoqda...")
        os.rename(str(kg_db_eski), str(kg_db))
        sys.exit(1)
        
    print("Muvaffaqiyatli almashtirildi!")

def qaytar(graphify_dir, eski_nomi):
    kg_db = graphify_dir / "kg_db"
    kg_db_eski = graphify_dir / eski_nomi
    
    if not kg_db_eski.exists():
        print(f"Xato: {eski_nomi} topilmadi.")
        sys.exit(1)
        
    print("!!! DIQQAT !!!")
    print("Qaytarishdan keyin MCP server va bot qayta ishga tushirilishi shart!")
    
    sana = datetime.now().strftime("%Y%m%d-%H%M%S")
    kg_db_buzuq = graphify_dir / f"kg_db-buzuq-{sana}"
    
    print(f"\nJarayon:")
    print(f"1. {kg_db.name} -> {kg_db_buzuq.name}")
    try:
        if kg_db.exists():
            os.rename(str(kg_db), str(kg_db_buzuq))
    except Exception as e:
        print(f"Xato: {e}")
        sys.exit(1)
        
    print(f"2. {kg_db_eski.name} -> {kg_db.name}")
    try:
        os.rename(str(kg_db_eski), str(kg_db))
    except Exception as e:
        print(f"Xato: {e}. Orqaga qaytarilmoqda...")
        os.rename(str(kg_db_buzuq), str(kg_db))
        sys.exit(1)
        
    print("Muvaffaqiyatli qaytarildi!")

def build_new_db(jarayon_soni=None):
    graphify_dir = Path(os.environ["GRAPHIFY_DIR"])
    old_db_path = graphify_dir / "kg_db"
    new_db_path = graphify_dir / "kg_db-yangi"
    
    if new_db_path.exists():
        print("Mavjud kg_db-yangi o'chirilmoqda (yarim eski ma'lumot qolmasligi uchun)...")
        import shutil
        shutil.rmtree(str(new_db_path), ignore_errors=True)
        
    try:
        from embedders import build_chroma_embedding_function
        embed_fn = build_chroma_embedding_function()
    except Exception as e:
        print(f"Embedder yuklanmadi: {e}")
        sys.exit(1)
        
    client_old = chromadb.PersistentClient(path=str(old_db_path))
    col_old = client_old.get_collection("graphify", embedding_function=embed_fn)
    
    client_new = chromadb.PersistentClient(path=str(new_db_path))
    col_new = client_new.get_or_create_collection("graphify", metadata={"hnsw:space": "cosine"}, embedding_function=embed_fn)
    
    print("\n--- 0. Eski indeksdan source -> project xaritasini qurish ---")
    source_project_counts = defaultdict(Counter)
    old_project_counts = Counter()
    old_source_projects = set()
    batch_size = 2000
    offset = 0
    total_old_records = 0
    
    while True:
        res_batch = col_old.get(limit=batch_size, offset=offset, include=["metadatas"])
        batch_ids = res_batch.get("ids", [])
        if not batch_ids:
            break
        total_old_records += len(batch_ids)
        for meta in res_batch.get("metadatas", []):
            if not meta:
                continue
            src = meta.get("source")
            prj = meta.get("project")
            if prj:
                old_project_counts[prj] += 1
            if src and prj:
                norm_src = normalize_source(src)
                source_project_counts[norm_src][prj] += 1
                old_source_projects.add((norm_src, prj))
        offset += len(batch_ids)
        
    source_to_project = {}
    conflict_count = 0
    for norm_src, prj_counter in source_project_counts.items():
        most_common_prj, _ = prj_counter.most_common(1)[0]
        source_to_project[norm_src] = most_common_prj
        if len(prj_counter) > 1:
            conflict_count += 1
            
    print(f"Eski indeksdan {total_old_records} ta yozuv o'qildi.")
    print(f"Xaritada unikal fayl yo'llari: {len(source_to_project)} ta.")
    print(f"Eski indeksdagi loyihalar soni: {len(old_project_counts)} ta.")
    if conflict_count > 0:
        print(f"Ogohlantirish: {conflict_count} ta fayl uchun bir nechta loyiha topildi (eng ko'p uchragani tanlandi).")
    
    print("\n--- 1. save_memory yozuvlarini ko'chirish ---")
    res_mem = col_old.get(where={"source": "save_memory"}, include=["documents", "metadatas", "embeddings"])
    old_mem_count = 0
    if res_mem and res_mem.get("ids"):
        old_mem_count = len(res_mem["ids"])
        col_new.add(
            ids=res_mem["ids"],
            documents=res_mem["documents"],
            metadatas=res_mem["metadatas"],
            embeddings=res_mem["embeddings"]
        )
    print(f"{old_mem_count} ta save_memory ko'chirildi.")
    
    print("\n--- Yangi yozuvlarni yig'ish ---")
    all_new_ids = []
    all_new_docs = []
    all_new_metas = []

    print("\n--- 2. Arxiv fayllarini indekslash ---")
    archive_shortcodes_expected = set()
    for p, rec in archive.all_records():
        if rec.get("usable"):
            archive_shortcodes_expected.add(rec.get("shortcode"))
            rows = kb.build_documents(rec)
            if rows:
                for r in rows:
                    all_new_ids.append(r[0])
                    all_new_docs.append(r[1])
                    all_new_metas.append(r[2])

    print(f"{len(archive_shortcodes_expected)} ta usable=true post yig'ildi.")
    
    print("\n--- 3. Oddiy hujjatlarni indekslash ---")
    search_paths = [ROOT / "Graphify", ROOT / "Promtlarim"]
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    doc_files = 0
    fallback_files_count = 0
    fallback_files_list = []
    
    for f in collect_files(search_paths):
        try:
            f_text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
            
        if has_shortcode_frontmatter(f_text):
            continue
            
        parent_id = str(f.resolve())
        chunks = chunk_by_tokens(f_text, parent_id=parent_id)
        
        norm_key = normalize_source(f)
        project_name = source_to_project.get(norm_key)
        if not project_name:
            project_name = "shared"
            fallback_files_count += 1
            fallback_files_list.append(str(f))
            
        added_file = False
        for c in chunks:
            chunk_text = c["text"]
            i = c["chunk_index"]
            cid = "file_" + hashlib.sha1(f"{parent_id}#{i}".encode()).hexdigest()[:16]
            all_new_ids.append(cid)
            all_new_docs.append(chunk_text)
            
            all_new_metas.append({
                "id": cid,
                "title": f"{f.name} (part {i + 1})",
                "kind": "doc",
                "project": project_name,
                "source": str(f),
                "date": today,
                "tags": "",
                "est_tokens": max(1, len(chunk_text) // 4),
                "parent_id": parent_id,
                "chunk_index": i,
                "chunk_total": c["chunk_total"],
                "token_count": c["token_count"]
            })
            added_file = True
            
        if added_file:
            doc_files += 1

    print(f"{doc_files} ta hujjat yig'ildi.")
    if fallback_files_count > 0:
        print(f"Xaritada topilmagan yangi fayllar soni: {fallback_files_count} ta.")

    print(f"\n--- 4. Embedding hisoblash va DB ga yozish (Parallel) ---")
    
    total_chunks = len(all_new_docs)
    if total_chunks > 0:
        batch_size = 256
        batches = [all_new_docs[i:i+batch_size] for i in range(0, total_chunks, batch_size)]
        
        num_workers = min(24, os.cpu_count() - 4) if os.cpu_count() else 4
        if jarayon_soni:
            num_workers = jarayon_soni
        print(f"Ishchi jarayonlar soni: {num_workers}")
        
        all_new_embs = []
        start_time = time.time()
        
        with multiprocessing.Pool(num_workers, initializer=init_worker) as pool:
            processed = 0
            for i, res in enumerate(pool.imap(embed_batch, batches)):
                all_new_embs.extend(res)
                processed += len(res)
                
                # Progress every 5 batches (~1280 chunks)
                if (i + 1) % 5 == 0 or processed == total_chunks:
                    elapsed = time.time() - start_time
                    speed = processed / elapsed if elapsed > 0 else 0
                    remain_chunks = total_chunks - processed
                    eta = remain_chunks / speed if speed > 0 else 0
                    print(f"[{processed}/{total_chunks}] {elapsed:.1f}s, {speed:.1f} chunk/s, qolgan vaqt: {eta:.1f}s")

        print("\nChroma'ga yozilmoqda...")
        # Chroma write in single process
        batch_write = 5000
        for i in range(0, total_chunks, batch_write):
            col_new.add(
                ids=all_new_ids[i:i+batch_write],
                documents=all_new_docs[i:i+batch_write],
                metadatas=all_new_metas[i:i+batch_write],
                embeddings=all_new_embs[i:i+batch_write]
            )
        print(f"Barcha {total_chunks} ta chunk DB ga yozildi.")

    
    print("\n--- TEKSHIRUV ---")
    new_project_counts = Counter()
    new_source_projects = set()
    new_mem_count = 0
    db_shortcodes = set()
    transcript_ocr_count = 0
    total_new_chunks = 0
    all_doc_ids = []
    
    batch_size = 2000
    offset = 0
    while True:
        res_batch = col_new.get(limit=batch_size, offset=offset, include=["metadatas"])
        batch_ids = res_batch.get("ids", [])
        if not batch_ids:
            break
        total_new_chunks += len(batch_ids)
        for cid, m in zip(batch_ids, res_batch.get("metadatas", [])):
            if not m:
                continue
            src = m.get("source", "")
            prj = m.get("project")
            
            if src == "save_memory":
                new_mem_count += 1
            else:
                all_doc_ids.append(cid)
                
            if src and prj:
                new_source_projects.add((normalize_source(src), prj))
                
            if src.startswith("instagram-file:"):
                parts = src.split(":")
                if len(parts) >= 2:
                    db_shortcodes.add(parts[1])
                if ":transcript:" in src or ":ocr:" in src or ":caption:" in src:
                    transcript_ocr_count += 1
                    
            if prj:
                new_project_counts[prj] += 1
            else:
                new_project_counts["(noma'lum)"] += 1
        offset += len(batch_ids)
        
    print("\n--- Loyihalar taqsimoti jadvali ---")
    print(f"{'Loyiha':<30} | {'Eski':>8} | {'Yangi':>8} | {'Farq':>8}")
    print("-" * 62)
    
    all_projects = sorted(set(old_project_counts.keys()) | set(new_project_counts.keys()))
    for prj in all_projects:
        c_old = old_project_counts.get(prj, 0)
        c_new = new_project_counts.get(prj, 0)
        diff = c_new - c_old
        diff_str = f"+{diff}" if diff > 0 else str(diff)
        print(f"{prj:<30} | {c_old:>8} | {c_new:>8} | {diff_str:>8}")
    print("-" * 62)
    
    errors = []
    
    if new_mem_count != old_mem_count:
        errors.append(f"save_memory soni mos emas: eski={old_mem_count}, yangi={new_mem_count}")
        
    missing_sc = archive_shortcodes_expected - db_shortcodes
    if missing_sc:
        errors.append(f"Quyidagi usable=true postlar indekslanmadi ({len(missing_sc)} ta): {list(missing_sc)[:5]}...")
        
    missing_sp = old_source_projects - new_source_projects
    if missing_sp:
        errors.append(f"Yangi bazada {len(missing_sp)} ta (manba, loyiha) juftligi yo'qoldi. Misol: {list(missing_sp)[:3]}")
        
    # Token length checking with true tokenizer
    sample_size = min(1000, len(all_doc_ids))
    sampled_ids = random.sample(all_doc_ids, sample_size) if sample_size > 0 else []
    long_chunks = 0
    if sampled_ids:
        docs_res = col_new.get(ids=sampled_ids, include=["documents"])
        for doc in docs_res.get("documents", []):
            if count_tokens(doc) > 256:
                long_chunks += 1
                
    if long_chunks > 0:
        errors.append(f"O'lchangan {sample_size} ta namunadan {long_chunks} ta chunk 256 tokendan uzun.")
        
    # Memory chunk token length warning
    long_memory_chunks = 0
    if res_mem and res_mem.get("documents"):
        for doc in res_mem.get("documents", []):
            if count_tokens(doc) > 256:
                long_memory_chunks += 1

    try:
        from indeks_tekshir import audit_index
        audit_errors = audit_index(new_db_path)
        for k, v in audit_errors.items():
            if k == "long_chunks":
                continue  # we did our own accurate token length test
            if v and isinstance(v, list) and len(v) > 0:
                errors.append(f"Audit ({k}): {len(v)} ta xato")
    except Exception as e:
        errors.append(f"indeks_tekshir.py ishga tushirishda xato: {e}")
        
    if conflict_count > 0:
        print(f"\nOgohlantirish: {conflict_count} ta fayl uchun bir nechta loyiha topildi.")
    if fallback_files_count > 0:
        print(f"Eslatma: {fallback_files_count} ta yangi fayl eski indeksda topilmadi va 'shared' ga biriktirildi.")
        (new_db_path / "fallback_files.txt").write_text("\n".join(fallback_files_list), encoding="utf-8")
        
    print("\n--- Smoke test ishga tushirilmoqda ---")
    smoke_res = subprocess.run([sys.executable, str(graphify_dir / "indeks-smoke.py")], capture_output=True, text=True)
    if smoke_res.returncode != 0:
        errors.append(f"Smoke test yiqildi:\n{smoke_res.stdout}\n{smoke_res.stderr}")
        
    # Tekshiruv natijasini faylga yozish
    fingerprint = get_fingerprint(new_db_path, total_new_chunks)
    report = {
        "ok": not bool(errors),
        "sana": datetime.now(timezone.utc).isoformat(),
        "xatolar": errors,
        "yangi_db_barmoq_izi": fingerprint,
        "uzun_memory": long_memory_chunks,
        "transcript_ocr_count": transcript_ocr_count
    }
    
    (new_db_path / "_tekshiruv.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if errors:
        print("\nXATOLIKLAR TOPILDI:")
        for err in errors:
            print(f"- {err}")
        print("Almashtirish TAQIQLANADI.")
        sys.exit(1)
    else:
        print("\nTekshiruv MUVAFFAQIYATLI o'tdi!")
        print(f"Jami chunk: {total_new_chunks}")
        print(f"Xotira (save_memory) yozuvlari: {new_mem_count} ta")
        print(f"Uzun memory yozuvlari (ogohlantirish): {long_memory_chunks} ta")
        print(f"O'lchangan {sample_size} namunada uzun chunklar yo'q.")
        print(f"Transcript/OCR chunklari: {transcript_ocr_count} ta")
        print("Indeksni almashtirishga tayyor. Ishga tushiring: python indeks-qayta-qur.py --almashtir")

def main():
    parser = argparse.ArgumentParser(description="Indeksni qayta qurish va almashtirish")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--almashtir", action="store_true", help="Yangi indeksni joriy qilib, eskini zaxiralash")
    group.add_argument("--qaytar", metavar="ESKI_NOM", help="Zaxiralangan indeksga qaytarish (masalan: kg_db-eski-20260905-120000)")
    group.add_argument("--yangidan", action="store_true", help="Mavjud kg_db-yangi ni tozalab, noldan qayta qurish")
    parser.add_argument("--jarayon", type=int, default=None, help="Ishchi jarayonlar soni (multiprocessing uchun)")
    
    args = parser.parse_args()
    
    graphify_dir = Path(os.environ["GRAPHIFY_DIR"])
    kg_db = graphify_dir / "kg_db"
    
    if args.almashtir:
        almashtir(graphify_dir)
    elif args.qaytar:
        qaytar(graphify_dir, args.qaytar)
    else:
        if not kg_db.exists():
            print("Xato: kg_db topilmadi! Avtomatik bo'sh baza yaratish taqiqlanadi.")
            sys.exit(1)
        build_new_db(args.jarayon)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
