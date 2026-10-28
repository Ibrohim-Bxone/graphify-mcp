#!/usr/bin/env python3
"""Build and synchronize Graphify v2 database (bge-m3 embeddings side-by-side).

Reads v1 strictly via SQLite in read-only mode (file:...chroma.sqlite3?mode=ro),
embeds records with Ollama bge-m3, populates kg_db_v2 ChromaDB and keyword index,
and verifies consistency against manifest.
"""

import argparse
import hashlib
import json
import os
import random
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

# Add project root to sys.path so embedders and other modules can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embedders import build_embedder

BGE_M3_CONFIG = {
    "embedder_type": "ollama",
    "config": {
        "ollama": {
            "base_url": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            "model": "bge-m3",
        }
    },
}
EXPECTED_DIMENSION = 1024


def compute_hashes(doc: str, meta: dict) -> tuple[str, str]:
    """Compute sha256 of document and sha256 of sorted metadata JSON."""
    doc_hash = hashlib.sha256((doc or "").encode("utf-8")).hexdigest()
    meta_json = json.dumps(meta or {}, sort_keys=True, ensure_ascii=False)
    meta_hash = hashlib.sha256(meta_json.encode("utf-8")).hexdigest()
    return doc_hash, meta_hash


def read_v1_records(v1_dir: Path, limit: int | None = None) -> tuple[str, dict, float, list[dict]]:
    """Read collection metadata and records from v1 via read-only SQLite.

    Never opens v1 via chromadb to prevent unintended writes or lock contention.
    """
    v1_sqlite = v1_dir / "chroma.sqlite3"
    if not v1_sqlite.exists():
        raise FileNotFoundError(f"v1 SQLite fayli topilmadi: {v1_sqlite}")

    v1_mtime = v1_sqlite.stat().st_mtime
    uri = f"file:{v1_sqlite.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=30.0)
    try:
        cur = conn.cursor()

        # 1. Collection name and settings
        cur.execute("SELECT id, name FROM collections LIMIT 1")
        col_row = cur.fetchone()
        if not col_row:
            raise RuntimeError("v1 bazasida kolleksiya topilmadi")
        col_id, col_name = col_row

        cur.execute(
            "SELECT key, str_value, int_value, float_value, bool_value "
            "FROM collection_metadata WHERE collection_id = ?",
            (col_id,),
        )
        col_meta = {}
        for k, s, i, f, b in cur.fetchall():
            if b is not None:
                col_meta[k] = bool(b)
            elif i is not None:
                col_meta[k] = i
            elif f is not None:
                col_meta[k] = f
            elif s is not None:
                col_meta[k] = s

        # 2. Fetch document texts and IDs
        query = (
            "SELECT e.id, e.embedding_id, f.c0 "
            "FROM embeddings e "
            "JOIN embedding_fulltext_search_content f ON e.id = f.id "
            "ORDER BY e.id"
        )
        if limit is not None and limit > 0:
            query += f" LIMIT {limit}"
        cur.execute(query)
        doc_rows = cur.fetchall()

        if not doc_rows:
            return col_name, col_meta, v1_mtime, []

        row_ids = [r[0] for r in doc_rows]
        min_id, max_id = min(row_ids), max(row_ids)

        # 3. Fetch metadata for these row IDs
        cur.execute(
            "SELECT id, key, string_value, int_value, float_value, bool_value "
            "FROM embedding_metadata "
            "WHERE id >= ? AND id <= ?",
            (min_id, max_id),
        )
        meta_rows = cur.fetchall()

        meta_by_id: dict[int, dict] = {}
        for mid, key, s_val, i_val, f_val, b_val in meta_rows:
            if key.startswith("chroma:"):
                continue
            val = (
                bool(b_val)
                if b_val is not None
                else i_val
                if i_val is not None
                else f_val
                if f_val is not None
                else s_val
            )
            meta_by_id.setdefault(mid, {})[key] = val

        records = []
        for rid, emb_id, doc in doc_rows:
            records.append({
                "id": emb_id,
                "document": doc,
                "metadata": meta_by_id.get(rid, {}),
            })
        return col_name, col_meta, v1_mtime, records
    finally:
        conn.close()


def embed_text_with_retry(embedder, text: str, max_retries: int = 5, initial_backoff: float = 1.0) -> list[float]:
    """Embed single text using Ollama with exponential backoff on failure."""
    delay = initial_backoff
    last_err = None
    for attempt in range(max_retries):
        try:
            vec = embedder.encode([text])[0]
            if hasattr(vec, "tolist"):
                vec = vec.tolist()
            else:
                vec = list(vec)
            if len(vec) != EXPECTED_DIMENSION:
                raise ValueError(
                    f"O'lcham mos emas: kutilgan={EXPECTED_DIMENSION}, olindi={len(vec)}"
                )
            return vec
        except ValueError:
            raise
        except Exception as e:
            last_err = e
            if attempt == max_retries - 1:
                break
            time.sleep(delay)
            delay *= 2.0
    raise RuntimeError(f"Ollama embedding xatoligi ({max_retries} urinishdan so'ng): {last_err}") from last_err


def rebuild_keyword_index_subprocess(out_dir: Path) -> None:
    """Run keyword_index.rebuild() in an isolated subprocess with GRAPHIFY_DB_PATH set to out_dir."""
    env = dict(os.environ)
    env["GRAPHIFY_DB_PATH"] = str(out_dir.resolve())
    cmd = [
        sys.executable,
        "-c",
        "import sys; sys.path.insert(0, '.'); import keyword_index; keyword_index.rebuild()",
    ]
    print(f"[keyword_index] Qayta qurish boshlandi: GRAPHIFY_DB_PATH={out_dir}...", flush=True)
    res = subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[keyword_index] Xatolik yuz berdi:\nSTDERR:\n{res.stderr}\nSTDOUT:\n{res.stdout}", file=sys.stderr)
        res.check_returncode()
    print("[keyword_index] Muvaffaqiyatli qayta qurildi.", flush=True)


def verify_v2(col, records: list[dict], manifest_path: Path, expected_count: int) -> None:
    """Verify record counts, manifest IDs, and sample documents/metadata."""
    v2_count = col.count()
    if v2_count != expected_count:
        raise AssertionError(f"Kolleksiya soni mos kelmadi: v2={v2_count}, kutilgan={expected_count}")

    # Verify all manifest IDs are present
    manifest_ids = set()
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                manifest_ids.add(item["id"])

    existing_v2_ids = set(col.get(include=[])["ids"])
    missing_in_v2 = manifest_ids - existing_v2_ids
    if missing_in_v2:
        raise AssertionError(f"Manifestdagi {len(missing_in_v2)} ta ID v2 kolleksiyasida topilmadi: {list(missing_in_v2)[:5]}")

    # Random 20 checks
    sample_size = min(20, len(records))
    samples = random.sample(records, sample_size) if sample_size > 0 else []
    for s in samples:
        res = col.get(ids=[s["id"]], include=["documents", "metadatas"])
        if not res["ids"]:
            raise AssertionError(f"Namunaviy ID v2 da topilmadi: {s['id']}")
        v2_doc = res["documents"][0]
        v2_meta = res["metadatas"][0] or {}
        if v2_doc != s["document"]:
            raise AssertionError(f"Hujjat matni mos kelmadi (ID={s['id']}): v1={s['document'][:50]} vs v2={v2_doc[:50]}")
        if v2_meta != s["metadata"]:
            raise AssertionError(f"Metadata mos kelmadi (ID={s['id']}): v1={s['metadata']} vs v2={v2_meta}")

    print(
        f"[verify] Tekshiruv muvaffaqiyatli: {v2_count} ta yozuv, barcha manifest ID mavjud, "
        f"{sample_size} ta tasodifiy namuna 100% mos.",
        flush=True,
    )


def write_manifest_and_info(out_dir: Path, records: list[dict], v1_mtime: float, started_at: datetime) -> None:
    """Write manifest.jsonl and build_info.json."""
    manifest_path = out_dir / "manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for r in records:
            d_hash, m_hash = compute_hashes(r["document"], r["metadata"])
            entry = {
                "id": r["id"],
                "doc_sha256": d_hash,
                "meta_sha256": m_hash,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Source counts
    sources = Counter()
    for r in records:
        m = r.get("metadata", {})
        origin = m.get("origin") or m.get("source") or "unknown"
        if "instagram.com" in str(origin):
            origin = "instagram"
        sources[origin] += 1

    completed_at = datetime.now(timezone.utc)
    build_info = {
        "source_counts": dict(sources),
        "total_records": len(records),
        "model": "bge-m3",
        "dimension": EXPECTED_DIMENSION,
        "timestamps": {
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "duration_seconds": round((completed_at - started_at).total_seconds(), 2),
        },
        "v1_sqlite_mtime": v1_mtime,
    }
    with open(out_dir / "build_info.json", "w", encoding="utf-8") as f:
        json.dump(build_info, f, indent=2, ensure_ascii=False)


def run_sync(v1_dir: Path, out_dir: Path, batch_size: int = 50, max_workers: int = 8) -> None:
    """Compare current v1 against manifest, upsert new/changed ids, delete removed ids."""
    manifest_path = out_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest topilmadi ({manifest_path}). Avval to'liq build yarating.")

    started_at = datetime.now(timezone.utc)
    print(f"[sync] Manifest yuklanmoqda: {manifest_path}...", flush=True)
    manifest = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                manifest[item["id"]] = item

    col_name, col_meta, v1_mtime, v1_records = read_v1_records(v1_dir)
    v1_map = {r["id"]: r for r in v1_records}

    to_upsert = []
    unchanged = 0
    for r in v1_records:
        rid = r["id"]
        d_hash, m_hash = compute_hashes(r["document"], r["metadata"])
        if rid not in manifest:
            to_upsert.append(r)
        elif manifest[rid]["doc_sha256"] != d_hash or manifest[rid]["meta_sha256"] != m_hash:
            to_upsert.append(r)
        else:
            unchanged += 1

    to_delete = [mid for mid in manifest if mid not in v1_map]

    print(
        f"[sync] Reja: {len(to_upsert)} ta yangilash/qo'shish, {len(to_delete)} ta o'chirish, "
        f"{unchanged} ta o'zgarmagan.",
        flush=True,
    )

    if not to_upsert and not to_delete:
        print("[sync] Baza allaqachon to'liq sinxron holatda.", flush=True)
        return

    import chromadb

    client = chromadb.PersistentClient(path=str(out_dir))
    col = client.get_or_create_collection(
        name=col_name,
        metadata={"hnsw:space": col_meta.get("hnsw:space", "l2")},
    )

    if to_delete:
        print(f"[sync] {len(to_delete)} ta yozuv o'chirilmoqda...", flush=True)
        col.delete(ids=to_delete)

    if to_upsert:
        embedder = build_embedder(BGE_M3_CONFIG)
        print(f"[sync] {len(to_upsert)} ta yozuv bge-m3 bilan embed qilinmoqda...", flush=True)
        for i in range(0, len(to_upsert), batch_size):
            chunk = to_upsert[i : i + batch_size]
            docs = [c["document"] for c in chunk]
            ids = [c["id"] for c in chunk]
            metas = [c["metadata"] for c in chunk]

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                embeddings = list(
                    executor.map(lambda text: embed_text_with_retry(embedder, text), docs)
                )

            col.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
            print(f"[sync] {min(i + batch_size, len(to_upsert))}/{len(to_upsert)} yangilandi...", flush=True)

    # Rewrite manifest & info
    write_manifest_and_info(out_dir, v1_records, v1_mtime, started_at)

    # Rebuild keyword index
    rebuild_keyword_index_subprocess(out_dir)

    # Verify
    verify_v2(col, v1_records, manifest_path, len(v1_records))
    print("[sync] Sinxronizatsiya yakunlandi.", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Build or sync Graphify v2 database")
    parser.add_argument("--v1-db", default="kg_db", help="Path to source v1 database (default: kg_db)")
    parser.add_argument("--out", default=None, help="Output directory for v2 database")
    parser.add_argument("--smoke", action="store_true", help="Smoke run: first 300 records into kg_db_v2_smoke/")
    parser.add_argument("--sync", action="store_true", help="Sync incremental changes from v1")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for embedding and DB writes")
    parser.add_argument("--max-workers", type=int, default=8, help="Worker threads for parallel Ollama calls")
    args = parser.parse_args()

    v1_dir = Path(args.v1_db)
    if args.out:
        out_dir = Path(args.out)
    elif args.smoke:
        out_dir = Path("kg_db_v2_smoke")
    else:
        out_dir = Path("kg_db_v2")

    out_dir.mkdir(parents=True, exist_ok=True)

    if args.sync:
        run_sync(v1_dir, out_dir, batch_size=args.batch_size, max_workers=args.max_workers)
        return

    started_at = datetime.now(timezone.utc)
    limit = 300 if args.smoke else None
    print(
        f"[build_v2] Boshlanmoqda: v1={v1_dir}, out={out_dir}, smoke={args.smoke} (limit={limit})...",
        flush=True,
    )

    # 1. Read v1 via read-only SQLite
    col_name, col_meta, v1_mtime, records = read_v1_records(v1_dir, limit=limit)
    print(
        f"[build_v2] v1 o'qildi: kolleksiya='{col_name}', meta={col_meta}, yozuvlar={len(records)}.",
        flush=True,
    )
    if not records:
        print("[build_v2] Yozuvlar topilmadi, to'xtatildi.", flush=True)
        return

    # 2. Setup v2 ChromaDB
    import chromadb

    client = chromadb.PersistentClient(path=str(out_dir))
    space = col_meta.get("hnsw:space", "l2")
    col = client.get_or_create_collection(name=col_name, metadata={"hnsw:space": space})

    # Resumable: find already processed IDs in v2
    existing_ids = set(col.get(include=[])["ids"])
    pending = [r for r in records if r["id"] not in existing_ids]
    print(
        f"[build_v2] Mavjud yozuvlar: {len(existing_ids)}, qolgan kutilayotgan: {len(pending)}.",
        flush=True,
    )

    # 3. Embed and upsert pending records
    if pending:
        embedder = build_embedder(BGE_M3_CONFIG)
        # Check model dimension on first item
        test_vec = embed_text_with_retry(embedder, pending[0]["document"])
        if len(test_vec) != EXPECTED_DIMENSION:
            raise ValueError(f"Model o'lchami noto'g'ri: {len(test_vec)} != {EXPECTED_DIMENSION}")

        total_pending = len(pending)
        for i in range(0, total_pending, args.batch_size):
            chunk = pending[i : i + args.batch_size]
            docs = [c["document"] for c in chunk]
            ids = [c["id"] for c in chunk]
            metas = [c["metadata"] for c in chunk]

            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                embeddings = list(
                    executor.map(lambda text: embed_text_with_retry(embedder, text), docs)
                )

            col.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
            processed = min(i + args.batch_size, total_pending)
            print(
                f"[build_v2] [{processed}/{total_pending}] {processed / total_pending * 100:.1f}% embed qilindi va saqlandi.",
                flush=True,
            )

    # 4. Write manifest and build_info
    print("[build_v2] Manifest va build_info.json yozilmoqda...", flush=True)
    write_manifest_and_info(out_dir, records, v1_mtime, started_at)

    # 5. Rebuild keyword index
    rebuild_keyword_index_subprocess(out_dir)

    # 6. Verify
    verify_v2(col, records, out_dir / "manifest.jsonl", len(records))
    print("[build_v2] Jarayon muvaffaqiyatli yakunlandi!", flush=True)


if __name__ == "__main__":
    main()
