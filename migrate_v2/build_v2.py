#!/usr/bin/env python3
"""Build and synchronize Graphify v2 database (bge-m3 embeddings side-by-side).

Reads v1 strictly via SQLite in read-only mode in one snapshot transaction,
collection-scoped by segments, copies full metadata/configuration,
embeds records with Ollama bge-m3, populates kg_db_v2 ChromaDB and keyword index,
verifies ALL records and keyword index, and writes manifest + atomic build_complete.json.
"""

import argparse
import hashlib
import json
import os
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

from embedders import build_embedder, get_ollama_model_digest, build_chroma_embedding_function, ModelIdentityError

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


def compute_corpus_hash(records: list[dict]) -> str:
    """Compute deterministic SHA-256 corpus hash across all record IDs and contents."""
    hasher = hashlib.sha256()
    for r in sorted(records, key=lambda x: str(x["id"])):
        d_hash, m_hash = compute_hashes(r["document"], r["metadata"])
        hasher.update(f"{r['id']}:{d_hash}:{m_hash}\n".encode("utf-8"))
    return hasher.hexdigest()


def read_v1_records(
    v1_dir: Path, limit: int | None = None, collection_name: str = "graphify"
) -> tuple[str, dict, dict, float, list[dict]]:
    """Read collection metadata, configuration, and records from v1 via read-only SQLite in one snapshot transaction.

    Never opens v1 via chromadb to prevent unintended writes or lock contention.
    Includes documentless records (LEFT JOIN) and scopes strictly to collection segments.
    """
    v1_sqlite = v1_dir / "chroma.sqlite3"
    if not v1_sqlite.exists():
        raise FileNotFoundError(f"v1 SQLite fayli topilmadi: {v1_sqlite}")

    v1_mtime = v1_sqlite.stat().st_mtime
    uri = f"file:{v1_sqlite.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=30.0)
    conn.isolation_level = None
    try:
        cur = conn.cursor()
        cur.execute("BEGIN")

        # 1. Collection name, config and settings
        cur.execute(
            "SELECT id, name, config_json_str FROM collections WHERE name = ? LIMIT 1",
            (collection_name,),
        )
        col_row = cur.fetchone()
        if not col_row:
            raise RuntimeError(f"v1 bazasida '{collection_name}' kolleksiyasi topilmadi")
        col_id, col_name, config_json_str = col_row
        col_config = json.loads(config_json_str) if config_json_str else {}

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

        # 2. Fetch document texts and IDs scoped by collection segments (including documentless records)
        query = (
            "SELECT e.id, e.embedding_id, f.c0 "
            "FROM embeddings e "
            "LEFT JOIN embedding_fulltext_search_content f ON e.id = f.id "
            "WHERE e.segment_id IN ("
            "    SELECT s.id FROM segments s "
            "    JOIN collections c ON s.collection = c.id "
            "    WHERE c.name = ?"
            ") "
            "ORDER BY e.id"
        )
        if limit is not None and limit > 0:
            query += f" LIMIT {limit}"
        cur.execute(query, (col_name,))
        doc_rows = cur.fetchall()

        if not doc_rows:
            return col_name, col_meta, col_config, v1_mtime, []

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
                "document": doc if doc is not None else "",
                "metadata": meta_by_id.get(rid, {}),
            })
        return col_name, col_meta, col_config, v1_mtime, records
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


def remove_completion_marker(out_dir: Path) -> None:
    """Remove build_complete.json before mutating destination; abort if removal fails."""
    complete_file = out_dir / "build_complete.json"
    if complete_file.exists():
        try:
            complete_file.unlink()
        except Exception as e:
            raise RuntimeError(f"build_complete.json o'chirish muvaffaqiyatsiz bo'ldi: {e}") from e


def validate_destination_persisted_identity(
    out_dir: Path,
    col_name: str,
    expected_model: str,
    expected_digest: str,
    expected_dimension: int,
    client=None,
) -> None:
    """Validate that any persisted manifest and collection identity in out_dir matches expected model identity.

    Fails closed if:
    - manifest exists but is unreadable or malformed JSON
    - manifest identity mismatches expected model, digest, or dimension
    - collection exists and has identity metadata that mismatches expected model, digest, or dimension
    - manifest and collection metadata conflict with each other
    """
    manifest_meta = {}
    manifest_file = out_dir / "graphify_manifest.json"
    if manifest_file.exists():
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest_meta = json.load(f)
        except Exception as e:
            raise ModelIdentityError(f"Sync rad etildi: Destination manifest fayli o'qib bo'lmadi yoki yaroqsiz: {e}") from e

        if not isinstance(manifest_meta, dict):
            raise ModelIdentityError("Sync rad etildi: Destination manifest fayli yaroqsiz (JSON lug'at emas)")

        m_model = manifest_meta.get("model")
        m_digest = manifest_meta.get("model_digest")
        m_dim = manifest_meta.get("dimension")

        if not m_model or m_model != expected_model:
            raise ModelIdentityError(
                f"Sync rad etildi: Destination manifest model mos kelmadi: mavjud={m_model}, kutilgan={expected_model}"
            )
        if not m_digest or m_digest != expected_digest:
            raise ModelIdentityError(
                f"Sync rad etildi: Destination manifest model_digest mos kelmadi: mavjud={m_digest}, kutilgan={expected_digest}"
            )
        if m_dim is None or int(m_dim) != expected_dimension:
            raise ModelIdentityError(
                f"Sync rad etildi: Destination manifest dimension mos kelmadi: mavjud={m_dim}, kutilgan={expected_dimension}"
            )

    col_meta = {}
    col_count = 0
    if client is not None:
        try:
            col = client.get_collection(col_name)
            if col:
                if isinstance(getattr(col, "metadata", None), dict):
                    col_meta = col.metadata
                try:
                    cnt = col.count()
                    col_count = int(cnt) if isinstance(cnt, (int, float)) else 0
                except Exception:
                    col_count = 0
        except Exception as e:
            from db import _is_absent_collection_error
            if not _is_absent_collection_error(e):
                raise

    if not col_meta:
        sqlite_file = out_dir / "chroma.sqlite3"
        if sqlite_file.exists():
            try:
                conn = sqlite3.connect(f"file:{sqlite_file.as_posix()}?mode=ro", uri=True, timeout=5.0)
                try:
                    cur = conn.cursor()
                    cur.execute("""
                        SELECT m.key, m.str_value, m.int_value, m.float_value, m.bool_value
                        FROM collection_metadata m
                        JOIN collections c ON m.collection_id = c.id
                        WHERE c.name = ?
                    """, (col_name,))
                    rows = cur.fetchall()
                    if rows:
                        s_meta = {}
                        for k, s, i, f, b in rows:
                            val = bool(b) if b is not None else i if i is not None else f if f is not None else s
                            s_meta[k] = val
                        col_meta = s_meta
                finally:
                    conn.close()
            except sqlite3.OperationalError as e:
                if "no such table" not in str(e).lower():
                    raise
            except Exception:
                raise

    if isinstance(col_count, (int, float)) and col_count > 0:
        c_model = col_meta.get("model") or col_meta.get("graphify:model")
        c_digest = col_meta.get("model_digest") or col_meta.get("graphify:model_digest")
        c_dim = col_meta.get("dimension") or col_meta.get("graphify:dimension")

        if not c_model or c_model != expected_model:
            raise ModelIdentityError(
                f"Sync rad etildi: Destination kolleksiyada yozuvlar mavjud, ammo model mos kelmadi yoki yo'q: mavjud={c_model}, kutilgan={expected_model}"
            )
        if not c_digest or c_digest != expected_digest:
            raise ModelIdentityError(
                f"Sync rad etildi: Destination kolleksiyada yozuvlar mavjud, ammo model_digest mos kelmadi yoki yo'q: mavjud={c_digest}, kutilgan={expected_digest}"
            )
        if c_dim is None or int(c_dim) != expected_dimension:
            raise ModelIdentityError(
                f"Sync rad etildi: Destination kolleksiyada yozuvlar mavjud, ammo dimension mos kelmadi yoki yo'q: mavjud={c_dim}, kutilgan={expected_dimension}"
            )
    elif col_meta:
        c_model = col_meta.get("model") or col_meta.get("graphify:model")
        c_digest = col_meta.get("model_digest") or col_meta.get("graphify:model_digest")
        c_dim = col_meta.get("dimension") or col_meta.get("graphify:dimension")

        if c_model and c_model != expected_model:
            raise ModelIdentityError(
                f"Sync rad etildi: Destination kolleksiya model mos kelmadi: mavjud={c_model}, kutilgan={expected_model}"
            )
        if c_digest and c_digest != expected_digest:
            raise ModelIdentityError(
                f"Sync rad etildi: Destination kolleksiya model_digest mos kelmadi: mavjud={c_digest}, kutilgan={expected_digest}"
            )
        if c_dim is not None and int(c_dim) != expected_dimension:
            raise ModelIdentityError(
                f"Sync rad etildi: Destination kolleksiya dimension mos kelmadi: mavjud={c_dim}, kutilgan={expected_dimension}"
            )

    if manifest_meta and col_meta:
        from db import _check_identity_conflict
        _check_identity_conflict(manifest_meta, "manifest", col_meta, "kolleksiya")


def rebuild_keyword_index_subprocess(out_dir: Path, timeout: float = None) -> None:
    """Run keyword_index.rebuild() in an isolated subprocess with GRAPHIFY_DB_PATH set to out_dir."""
    if timeout is None:
        timeout = float(os.environ.get("GRAPHIFY_KEYWORD_REBUILD_TIMEOUT", 1800.0))
    env = dict(os.environ)
    env["GRAPHIFY_DB_PATH"] = str(out_dir.resolve())
    env["GRAPHIFY_STORAGE_MODE"] = "local"
    cmd = [
        sys.executable,
        "-c",
        "import sys; sys.path.insert(0, '.'); import keyword_index; keyword_index.rebuild()",
    ]
    print(f"[keyword_index] Qayta qurish boshlandi: GRAPHIFY_DB_PATH={out_dir}...", flush=True)
    try:
        res = subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        print(f"[keyword_index] Qayta qurish vaqt limiti ({timeout}s) tugadi", file=sys.stderr)
        raise RuntimeError(f"keyword_index qayta qurish vaqt limiti ({timeout}s) tugadi") from e

    if res.returncode != 0:
        print(f"[keyword_index] Xatolik yuz berdi:\nSTDERR:\n{res.stderr}\nSTDOUT:\n{res.stdout}", file=sys.stderr)
        res.check_returncode()
    print("[keyword_index] Muvaffaqiyatli qayta qurildi.", flush=True)


def verify_v2(col, records: list[dict], out_dir: Path) -> None:
    """Verify ALL record counts, IDs, documents, metadatas, and keyword index content."""
    v2_count = col.count()
    expected_count = len(records)
    if v2_count != expected_count:
        raise AssertionError(f"Kolleksiya soni mos kelmadi: v2={v2_count}, kutilgan={expected_count}")

    v1_map = {r["id"]: r for r in records}
    dest_data = col.get(include=["documents", "metadatas"])
    dest_ids = set(dest_data["ids"])

    missing_in_v2 = set(v1_map.keys()) - dest_ids
    if missing_in_v2:
        raise AssertionError(f"v2 da {len(missing_in_v2)} ta ID topilmadi: {list(missing_in_v2)[:5]}")
    extra_in_v2 = dest_ids - set(v1_map.keys())
    if extra_in_v2:
        raise AssertionError(f"v2 da ortiqcha {len(extra_in_v2)} ta ID mavjud: {list(extra_in_v2)[:5]}")

    # Verify document text and metadata for all records
    for did, doc, meta in zip(dest_data["ids"], dest_data["documents"], dest_data["metadatas"]):
        expected = v1_map[did]
        if (doc or "") != (expected["document"] or ""):
            raise AssertionError(f"Hujjat matni mos kelmadi (ID={did})")
        if (meta or {}) != (expected["metadata"] or {}):
            raise AssertionError(f"Metadata mos kelmadi (ID={did})")

    # Verify keyword index content
    kw_sqlite = out_dir / "keyword_index.sqlite3"
    if not kw_sqlite.exists():
        raise AssertionError(f"Keyword index fayli topilmadi: {kw_sqlite}")

    from keyword_index import tokenize
    kw_conn = sqlite3.connect(str(kw_sqlite), timeout=10.0)
    try:
        kw_cur = kw_conn.cursor()
        kw_cur.execute("SELECT id, project, kind, parent_id, token_count, text FROM docs")
        doc_rows = kw_cur.fetchall()
        if len(doc_rows) != len(records):
            raise AssertionError(f"Keyword index 'docs' qatorlar soni mos kelmadi: {len(doc_rows)} != {len(records)}")

        doc_by_id = {r[0]: r for r in doc_rows}
        for r in records:
            rid = r["id"]
            if rid not in doc_by_id:
                raise AssertionError(f"Keyword index 'docs' da ID topilmadi: {rid}")
            _, proj, kind, parent, t_cnt, txt = doc_by_id[rid]
            exp_txt = r["document"] or ""
            if (txt or "") != exp_txt:
                raise AssertionError(f"Keyword index 'docs' matni mos kelmadi: ID={rid}")
            meta = r["metadata"] or {}
            exp_proj = meta.get("project", "")
            exp_kind = meta.get("kind", "")
            exp_parent = meta.get("parent_id", "")
            if not exp_parent and "shortcode" in meta:
                exp_parent = meta["shortcode"]
            if not exp_parent and "source" in meta:
                exp_parent = meta["source"]
            if proj != exp_proj:
                raise AssertionError(f"Keyword index 'docs' project mos kelmadi: ID={rid} ({proj} != {exp_proj})")
            if kind != exp_kind:
                raise AssertionError(f"Keyword index 'docs' kind mos kelmadi: ID={rid} ({kind} != {exp_kind})")
            if parent != exp_parent:
                raise AssertionError(f"Keyword index 'docs' parent_id mos kelmadi: ID={rid} ({parent} != {exp_parent})")
            exp_tokens = tokenize((meta.get("title") or "") + "\n" + exp_txt)
            if t_cnt != len(exp_tokens):
                raise AssertionError(f"Keyword index 'docs' token_count mos kelmadi: ID={rid} ({t_cnt} != {len(exp_tokens)})")

        kw_cur.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='fts_idx'")
        if kw_cur.fetchone()[0] > 0:
            kw_cur.execute("SELECT id, text FROM fts_idx")
            fts_rows = kw_cur.fetchall()
            if len(fts_rows) != len(records):
                raise AssertionError(f"Keyword index 'fts_idx' qatorlar soni mos kelmadi: {len(fts_rows)} != {len(records)}")
            fts_by_id = {r[0]: r[1] for r in fts_rows}
            for r in records:
                rid = r["id"]
                if rid not in fts_by_id:
                    raise AssertionError(f"Keyword index 'fts_idx' da ID topilmadi: {rid}")
                meta = r["metadata"] or {}
                exp_tokens = tokenize((meta.get("title") or "") + "\n" + (r["document"] or ""))
                exp_fts = " ".join(exp_tokens)
                if fts_by_id[rid] != exp_fts:
                    raise AssertionError(f"Keyword index 'fts_idx' matni mos kelmadi: ID={rid}")
    finally:
        kw_conn.close()

    print(
        f"[verify] Tekshiruv muvaffaqiyatli: barcha {v2_count} ta yozuv 100% mos, "
        f"keyword index ({len(doc_rows)} ta hujjat) tasdiqlandi.",
        flush=True,
    )


def write_manifest_and_identity(
    out_dir: Path,
    records: list[dict],
    v1_mtime: float,
    started_at: datetime,
    model_digest: str,
    distance: str,
    corpus_hash: str,
    kw_gen: str,
) -> dict:
    """Write manifest.jsonl and graphify_manifest.json with complete model identity."""
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

    identity = {
        "schema_version": 2,
        "backend": "ollama",
        "model": "bge-m3",
        "model_digest": model_digest,
        "dimension": EXPECTED_DIMENSION,
        "distance": distance,
        "normalization": False,
        "corpus_hash": corpus_hash,
        "keyword_index_generation": kw_gen,
        "total_records": len(records),
        "v1_sqlite_mtime": v1_mtime,
    }
    with open(out_dir / "graphify_manifest.json", "w", encoding="utf-8") as f:
        json.dump(identity, f, indent=2, ensure_ascii=False)

    return identity


def write_build_complete_atomic(out_dir: Path, identity: dict, started_at) -> None:
    """Atomically write build_complete.json only after verification and keyword rebuild succeed."""
    completed_at = datetime.now(timezone.utc)
    started_dt = (
        started_at
        if isinstance(started_at, datetime)
        else datetime.fromtimestamp(float(started_at), tz=timezone.utc)
    )
    complete_info = dict(identity)
    complete_info["timestamps"] = {
        "started_at": started_dt.isoformat(),
        "completed_at": completed_at.isoformat(),
        "duration_seconds": round((completed_at - started_dt).total_seconds(), 2),
    }
    tmp_path = out_dir / "build_complete.json.tmp"
    final_path = out_dir / "build_complete.json"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(complete_info, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, final_path)


def run_sync(v1_dir: Path, out_dir: Path, batch_size: int = 50, max_workers: int = 8) -> None:
    """Compare destination document+metadata hashes, re-embed/upsert changed ids, delete removed ids.

    Verifies ALL records and keyword index content; writes build_complete.json atomically
    only after keyword rebuild succeeds. An unchanged retry does not return early with a broken index.
    """
    if out_dir.resolve() == v1_dir.resolve():
        raise ValueError(f"Chiqish yo'li v1 manbasi bilan bir xil bo'lishi mumkin emas: {out_dir}")

    started_at = datetime.now(timezone.utc)
    print(f"[sync] v1 o'qilmoqda: {v1_dir}...", flush=True)
    col_name, col_meta, col_config, v1_mtime, v1_records = read_v1_records(v1_dir)
    v1_map = {r["id"]: r for r in v1_records}
    corpus_hash = compute_corpus_hash(v1_records)

    # Strictly validate model identity before ANY mutation
    model_digest = get_ollama_model_digest(
        BGE_M3_CONFIG["config"]["ollama"]["base_url"],
        BGE_M3_CONFIG["config"]["ollama"]["model"],
    )

    import chromadb

    client = chromadb.PersistentClient(path=str(out_dir))
    validate_destination_persisted_identity(
        out_dir=out_dir,
        col_name=col_name,
        expected_model="bge-m3",
        expected_digest=model_digest,
        expected_dimension=EXPECTED_DIMENSION,
        client=client,
    )

    space = col_meta.get("hnsw:space", "l2")
    sync_identity = {
        "schema_version": 2,
        "backend": "ollama",
        "model": "bge-m3",
        "model_digest": model_digest,
        "dimension": EXPECTED_DIMENSION,
        "distance": space,
    }
    embed_fn = build_chroma_embedding_function(identity=sync_identity)
    col = client.get_or_create_collection(
        name=col_name,
        metadata={"hnsw:space": space, **sync_identity},
        embedding_function=embed_fn,
    )
    # Persist identity metadata before any upsert (Finding 1 & TOPILMA 2)
    col.modify(metadata={**{k: v for k, v in (col.metadata if isinstance(col.metadata, dict) else {}).items() if not k.startswith("hnsw:")}, **sync_identity})

    # Fetch destination records to compare hashes
    dest_data = col.get(include=["documents", "metadatas"])
    dest_map = {}
    for did, doc, meta in zip(dest_data["ids"], dest_data["documents"], dest_data["metadatas"]):
        dest_map[did] = (doc, meta)

    to_upsert = []
    unchanged = 0
    for r in v1_records:
        rid = r["id"]
        d_hash, m_hash = compute_hashes(r["document"], r["metadata"])
        if rid not in dest_map:
            to_upsert.append(r)
        else:
            dest_doc, dest_meta = dest_map[rid]
            cur_d_hash, cur_m_hash = compute_hashes(dest_doc, dest_meta)
            if cur_d_hash != d_hash or cur_m_hash != m_hash:
                to_upsert.append(r)
            else:
                unchanged += 1

    to_delete = [did for did in dest_map if did not in v1_map]

    print(
        f"[sync] Reja: {len(to_upsert)} ta yangilash/qo'shish, {len(to_delete)} ta o'chirish, "
        f"{unchanged} ta o'zgarmagan.",
        flush=True,
    )

    manifest_file = out_dir / "graphify_manifest.json"
    complete_file = out_dir / "build_complete.json"
    kw_sqlite = out_dir / "keyword_index.sqlite3"
    needs_rebuild = bool(to_upsert or to_delete or not complete_file.exists() or not kw_sqlite.exists() or not manifest_file.exists())

    if not needs_rebuild:
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                m_check = json.load(f)
            with open(complete_file, "r", encoding="utf-8") as f:
                c_check = json.load(f)
            kw_mtime_gen = str(int(kw_sqlite.stat().st_mtime))
            if (
                m_check.get("corpus_hash") != corpus_hash
                or m_check.get("total_records") != len(v1_records)
                or m_check.get("model_digest") != model_digest
                or m_check.get("keyword_index_generation") != kw_mtime_gen
                or c_check.get("corpus_hash") != corpus_hash
                or c_check.get("keyword_index_generation") != kw_mtime_gen
            ):
                needs_rebuild = True
        except Exception:
            needs_rebuild = True

    if not to_upsert and not to_delete and not needs_rebuild:
        try:
            verify_v2(col, v1_records, out_dir)
            print("[sync] Baza allaqachon to'liq sinxron holatda va tasdiqlangan.", flush=True)
            return
        except Exception:
            needs_rebuild = True

    # Invalidate completion status before mutations, keyword rebuild, or manifest update (Finding 5 & TOPILMA 3)
    remove_completion_marker(out_dir)

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
            metas = [c["metadata"] if c.get("metadata") else None for c in chunk]

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                embeddings = list(
                    executor.map(lambda text: embed_text_with_retry(embedder, text), docs)
                )

            col.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
            print(f"[sync] {min(i + batch_size, len(to_upsert))}/{len(to_upsert)} yangilandi...", flush=True)

    # Rebuild keyword index
    rebuild_keyword_index_subprocess(out_dir)

    # Verify ALL records and keyword index content
    verify_v2(col, v1_records, out_dir)

    kw_gen = str(int(kw_sqlite.stat().st_mtime)) if kw_sqlite.exists() else "gen1"

    # Write manifest and identity
    identity = write_manifest_and_identity(
        out_dir=out_dir,
        records=v1_records,
        v1_mtime=v1_mtime,
        started_at=started_at,
        model_digest=model_digest,
        distance=space,
        corpus_hash=corpus_hash,
        kw_gen=kw_gen,
    )

    # Update collection metadata in ChromaDB (Finding 5)
    # modify() replaces the whole metadata and rejects hnsw:* keys; the space stays in configuration_json
    col.modify(metadata={**{k: v for k, v in (col.metadata if isinstance(col.metadata, dict) else {}).items() if not k.startswith("hnsw:")}, **identity})

    # Write build_complete.json atomically
    write_build_complete_atomic(out_dir, identity, started_at)
    print("[sync] Sinxronizatsiya muvaffaqiyatli yakunlandi.", flush=True)


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

    # Finding 1: reject --out that resolves to source
    if out_dir.resolve() == v1_dir.resolve():
        raise ValueError(f"Chiqish yo'li v1 manbasi bilan bir xil bo'lishi mumkin emas: out={out_dir}, v1={v1_dir}")

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

    # 1. Read v1 via read-only SQLite in snapshot transaction
    col_name, col_meta, col_config, v1_mtime, records = read_v1_records(v1_dir, limit=limit)
    print(
        f"[build_v2] v1 o'qildi: kolleksiya='{col_name}', meta={col_meta}, yozuvlar={len(records)}.",
        flush=True,
    )
    if not records:
        print("[build_v2] Yozuvlar topilmadi, to'xtatildi.", flush=True)
        return

    corpus_hash = compute_corpus_hash(records)

    # 2. Setup v2 ChromaDB copying full collection metadata and configuration + identity
    import chromadb

    client = chromadb.PersistentClient(path=str(out_dir))
    space = col_meta.get("hnsw:space", "l2")

    # Fetch model digest without falling back to unknown
    model_digest = get_ollama_model_digest(
        BGE_M3_CONFIG["config"]["ollama"]["base_url"],
        BGE_M3_CONFIG["config"]["ollama"]["model"],
    )

    # Validate persisted collection AND manifest identity before any destination mutation
    validate_destination_persisted_identity(
        out_dir=out_dir,
        col_name=col_name,
        expected_model="bge-m3",
        expected_digest=model_digest,
        expected_dimension=EXPECTED_DIMENSION,
        client=client,
    )

    merged_metadata = dict(col_meta)
    merged_metadata.update({
        "hnsw:space": space,
        "schema_version": 2,
        "backend": "ollama",
        "model": "bge-m3",
        "model_digest": model_digest,
        "dimension": EXPECTED_DIMENSION,
        "distance": space,
        "normalization": False,
        "corpus_hash": corpus_hash,
    })

    embed_fn = build_chroma_embedding_function(identity=merged_metadata)
    col = client.get_or_create_collection(
        name=col_name,
        metadata=merged_metadata,
        configuration=col_config,
        embedding_function=embed_fn,
    )
    # Persist identity metadata before any upsert (Finding 1 & TOPILMA 2)
    col.modify(metadata={**{k: v for k, v in (col.metadata if isinstance(col.metadata, dict) else {}).items() if not k.startswith("hnsw:")}, **merged_metadata})

    # Resumable: compare destination document+metadata hashes, do not skip changed ids!
    dest_data = col.get(include=["documents", "metadatas"])
    dest_map = {}
    for did, doc, meta in zip(dest_data["ids"], dest_data["documents"], dest_data["metadatas"]):
        dest_map[did] = (doc, meta)

    # Delete removed destination records on resume/build
    records_map = {r["id"]: r for r in records}
    to_delete = [did for did in dest_map if did not in records_map]

    pending = []
    for r in records:
        rid = r["id"]
        d_hash, m_hash = compute_hashes(r["document"], r["metadata"])
        if rid not in dest_map:
            pending.append(r)
        else:
            dest_doc, dest_meta = dest_map[rid]
            cur_d_hash, cur_m_hash = compute_hashes(dest_doc, dest_meta)
            if cur_d_hash != d_hash or cur_m_hash != m_hash:
                pending.append(r)

    # Remove completion marker before first mutation, keyword rebuild, or manifest update (Finding 5 & TOPILMA 3)
    remove_completion_marker(out_dir)

    if to_delete:
        print(f"[build_v2] Manbadan o'chirilgan {len(to_delete)} ta yozuv v2 dan o'chirilmoqda...", flush=True)
        col.delete(ids=to_delete)
        for did in to_delete:
            dest_map.pop(did, None)

    print(
        f"[build_v2] Mavjud to'g'ri yozuvlar: {len(dest_map) - (len(pending) - (len(records) - len(dest_map)))}, "
        f"kutilayotgan/o'zgargan: {len(pending)}.",
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
            metas = [c["metadata"] if c.get("metadata") else None for c in chunk]

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

    # 4. Rebuild keyword index
    rebuild_keyword_index_subprocess(out_dir)

    # 5. Verify ALL records and keyword index content
    verify_v2(col, records, out_dir)

    kw_sqlite = out_dir / "keyword_index.sqlite3"
    kw_gen = str(int(kw_sqlite.stat().st_mtime)) if kw_sqlite.exists() else "gen1"

    # 6. Write manifest and identity
    print("[build_v2] Manifest va graphify_manifest.json yozilmoqda...", flush=True)
    identity = write_manifest_and_identity(
        out_dir=out_dir,
        records=records,
        v1_mtime=v1_mtime,
        started_at=started_at,
        model_digest=model_digest,
        distance=space,
        corpus_hash=corpus_hash,
        kw_gen=kw_gen,
    )

    # Update collection metadata in ChromaDB (Finding 5)
    # modify() replaces the whole metadata and rejects hnsw:* keys; the space stays in configuration_json
    col.modify(metadata={**{k: v for k, v in (col.metadata if isinstance(col.metadata, dict) else {}).items() if not k.startswith("hnsw:")}, **identity})

    # 7. Write build_complete.json atomically
    write_build_complete_atomic(out_dir, identity, started_at)
    print("[build_v2] Jarayon muvaffaqiyatli yakunlandi!", flush=True)


if __name__ == "__main__":
    main()
