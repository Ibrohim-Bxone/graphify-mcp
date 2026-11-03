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
            "base_url": os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
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
    expected_backend: str = "ollama",
    expected_schema_version: int = 2,
    expected_distance: str = "l2",
    expected_normalization: bool = False,
) -> None:
    """Validate that any persisted manifest and collection identity in out_dir matches expected model identity.

    Fails closed if:
    - manifest exists but is unreadable or malformed JSON
    - manifest identity mismatches expected model, digest, dimension, backend, distance, normalization, or schema_version
    - collection exists and has identity metadata that mismatches expected values
    - manifest and collection metadata conflict with each other
    - plain and prefixed graphify:* keys conflict within the same source
    - collection count is bool, non-int, or negative
    """
    from identity_utils import _parse_int, _parse_bool, _check_source_internal_conflict

    keys_to_extract = [
        ("backend", str),
        ("model", str),
        ("dimension", _parse_int),
        ("model_digest", str),
        ("schema_version", _parse_int),
        ("distance", str),
        ("normalization", _parse_bool),
    ]

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

        _check_source_internal_conflict(manifest_meta, "Destination manifest")

        resolved_m = {}
        for key, cast_type in keys_to_extract:
            val_plain = manifest_meta.get(key)
            val_pref = manifest_meta.get(f"graphify:{key}")
            if val_plain is not None and val_pref is not None:
                differ = False
                try:
                    c1 = cast_type(val_plain)
                    c2 = cast_type(val_pref)
                    differ = (c1 != c2)
                except Exception:
                    differ = True
                if differ:
                    raise ModelIdentityError(
                        f"Sync rad etildi: Destination manifestda '{key}' va 'graphify:{key}' qiymatlari mos kelmadi: {val_plain} != {val_pref}"
                    )
                resolved_m[key] = val_plain
            else:
                resolved_m[key] = val_plain if val_plain is not None else val_pref

        m_backend = resolved_m.get("backend")
        m_model = resolved_m.get("model")
        m_digest = resolved_m.get("model_digest")
        m_dim = resolved_m.get("dimension")
        m_schema = resolved_m.get("schema_version")
        m_dist = resolved_m.get("distance")
        m_norm = resolved_m.get("normalization")

        if not m_model or m_model != expected_model:
            raise ModelIdentityError(
                f"Sync rad etildi: Destination manifest model mos kelmadi: mavjud={m_model}, kutilgan={expected_model}"
            )
        if not m_digest or m_digest != expected_digest:
            raise ModelIdentityError(
                f"Sync rad etildi: Destination manifest model_digest mos kelmadi: mavjud={m_digest}, kutilgan={expected_digest}"
            )
        try:
            m_dim_int = _parse_int(m_dim) if m_dim is not None else None
        except (ValueError, TypeError, OverflowError):
            m_dim_int = None
        if m_dim_int != expected_dimension:
            raise ModelIdentityError(
                f"Sync rad etildi: Destination manifest dimension mos kelmadi: mavjud={m_dim}, kutilgan={expected_dimension}"
            )
        if not m_backend or m_backend != expected_backend:
            raise ModelIdentityError(
                f"Sync rad etildi: Destination manifest backend mos kelmadi yoki yo'q: mavjud={m_backend}, kutilgan={expected_backend}"
            )
        if m_schema is not None:
            try:
                m_schema_int = _parse_int(m_schema)
            except (ValueError, TypeError, OverflowError):
                m_schema_int = None
            if m_schema_int != expected_schema_version:
                raise ModelIdentityError(
                    f"Sync rad etildi: Destination manifest schema_version mos kelmadi: mavjud={m_schema}, kutilgan={expected_schema_version}"
                )
        if m_dist is not None and expected_distance is not None:
            if not isinstance(m_dist, str) or m_dist.strip().lower() != expected_distance.strip().lower():
                raise ModelIdentityError(
                    f"Sync rad etildi: Destination manifest distance mos kelmadi: mavjud={m_dist}, kutilgan={expected_distance}"
                )
        if m_norm is not None:
            try:
                m_norm_bool = _parse_bool(m_norm)
            except (ValueError, TypeError):
                raise ModelIdentityError(
                    f"Sync rad etildi: Destination manifest normalization noto'g'ri formatda: {m_norm}"
                )
            if m_norm_bool != expected_normalization:
                raise ModelIdentityError(
                    f"Sync rad etildi: Destination manifest normalization mos kelmadi: mavjud={m_norm}, kutilgan={expected_normalization}"
                )

    col_meta = {}
    col_count = 0
    if client is not None:
        try:
            col = client.get_collection(col_name)
            if col:
                if isinstance(getattr(col, "metadata", None), dict):
                    col_meta = col.metadata
                cnt = col.count()
                if isinstance(cnt, bool) or not isinstance(cnt, int) or cnt < 0:
                    raise ModelIdentityError(
                        f"Sync rad etildi: Destination kolleksiya count yaroqsiz (bool, butun son emas yoki manfiy): {cnt!r}"
                    )
                col_count = cnt
        except ModelIdentityError:
            raise
        except Exception as e:
            from identity_utils import _is_absent_collection_error
            if not _is_absent_collection_error(e):
                raise

    if not col_meta and client is None:
        sqlite_file = out_dir / "chroma.sqlite3"
        if sqlite_file.exists():
            try:
                conn = sqlite3.connect(f"file:{sqlite_file.as_posix()}?mode=ro", uri=True, timeout=5.0)
                try:
                    cur = conn.cursor()
                    try:
                        cur.execute("SELECT id FROM collections WHERE name = ?", (col_name,))
                        c_row = cur.fetchone()
                    except sqlite3.OperationalError as e:
                        if "no such table" in str(e).lower():
                            c_row = None
                        else:
                            raise

                    if c_row is not None:
                        col_id = c_row[0]
                        cur.execute("""
                            SELECT m.key, m.str_value, m.int_value, m.float_value, m.bool_value
                            FROM collection_metadata m
                            WHERE m.collection_id = ?
                        """, (col_id,))
                        rows = cur.fetchall()
                        if rows:
                            s_meta = {}
                            for k, s, i, f, b in rows:
                                val = bool(b) if b is not None else i if i is not None else f if f is not None else s
                                s_meta[k] = val
                            col_meta = s_meta

                        cur.execute("""
                            SELECT count(*)
                            FROM embeddings e
                            JOIN segments s ON e.segment_id = s.id
                            WHERE s.collection = ?
                        """, (col_id,))
                        cnt_row = cur.fetchone()
                        if cnt_row:
                            cnt = cnt_row[0]
                            if isinstance(cnt, bool) or not isinstance(cnt, int) or cnt < 0:
                                raise ModelIdentityError(
                                    f"Sync rad etildi: Destination kolleksiya count yaroqsiz (bool, butun son emas yoki manfiy): {cnt!r}"
                                )
                            col_count = cnt
                finally:
                    conn.close()
            except sqlite3.OperationalError:
                raise
            except Exception:
                raise

    resolved_c = {}
    if col_meta:
        _check_source_internal_conflict(col_meta, "Destination kolleksiya")
        for key, cast_type in keys_to_extract:
            val_plain = col_meta.get(key)
            val_pref = col_meta.get(f"graphify:{key}")
            if val_plain is not None and val_pref is not None:
                differ = False
                try:
                    c1 = cast_type(val_plain)
                    c2 = cast_type(val_pref)
                    differ = (c1 != c2)
                except Exception:
                    differ = True
                if differ:
                    raise ModelIdentityError(
                        f"Sync rad etildi: Destination kolleksiyada '{key}' va 'graphify:{key}' qiymatlari mos kelmadi: {val_plain} != {val_pref}"
                    )
                resolved_c[key] = val_plain
            else:
                resolved_c[key] = val_plain if val_plain is not None else val_pref

    c_backend = resolved_c.get("backend")
    c_model = resolved_c.get("model")
    c_digest = resolved_c.get("model_digest")
    c_dim = resolved_c.get("dimension")
    c_schema = resolved_c.get("schema_version")
    c_dist = resolved_c.get("distance")
    c_norm = resolved_c.get("normalization")

    if col_count > 0:
        if not c_model or c_model != expected_model:
            raise ModelIdentityError(
                f"Sync rad etildi: Destination kolleksiyada yozuvlar mavjud, ammo model mos kelmadi yoki yo'q: mavjud={c_model}, kutilgan={expected_model}"
            )
        if not c_digest or c_digest != expected_digest:
            raise ModelIdentityError(
                f"Sync rad etildi: Destination kolleksiyada yozuvlar mavjud, ammo model_digest mos kelmadi yoki yo'q: mavjud={c_digest}, kutilgan={expected_digest}"
            )
        try:
            c_dim_int = _parse_int(c_dim) if c_dim is not None else None
        except (ValueError, TypeError, OverflowError):
            c_dim_int = None
        if c_dim_int != expected_dimension:
            raise ModelIdentityError(
                f"Sync rad etildi: Destination kolleksiyada yozuvlar mavjud, ammo dimension mos kelmadi yoki yo'q: mavjud={c_dim}, kutilgan={expected_dimension}"
            )
        if not c_backend or c_backend != expected_backend:
            raise ModelIdentityError(
                f"Sync rad etildi: Destination kolleksiyada yozuvlar mavjud, ammo backend mos kelmadi yoki yo'q: mavjud={c_backend}, kutilgan={expected_backend}"
            )
        if c_schema is not None:
            try:
                c_schema_int = _parse_int(c_schema)
            except (ValueError, TypeError, OverflowError):
                c_schema_int = None
            if c_schema_int != expected_schema_version:
                raise ModelIdentityError(
                    f"Sync rad etildi: Destination kolleksiya schema_version mos kelmadi: mavjud={c_schema}, kutilgan={expected_schema_version}"
                )
        if c_dist is not None and expected_distance is not None:
            if not isinstance(c_dist, str) or c_dist.strip().lower() != expected_distance.strip().lower():
                raise ModelIdentityError(
                    f"Sync rad etildi: Destination kolleksiya distance mos kelmadi: mavjud={c_dist}, kutilgan={expected_distance}"
                )
        if c_norm is not None:
            try:
                c_norm_bool = _parse_bool(c_norm)
            except (ValueError, TypeError):
                raise ModelIdentityError(
                    f"Sync rad etildi: Destination kolleksiya normalization noto'g'ri formatda: {c_norm}"
                )
            if c_norm_bool != expected_normalization:
                raise ModelIdentityError(
                    f"Sync rad etildi: Destination kolleksiya normalization mos kelmadi: mavjud={c_norm}, kutilgan={expected_normalization}"
                )
    elif col_meta:
        if c_model and c_model != expected_model:
            raise ModelIdentityError(
                f"Sync rad etildi: Destination kolleksiya model mos kelmadi: mavjud={c_model}, kutilgan={expected_model}"
            )
        if c_digest and c_digest != expected_digest:
            raise ModelIdentityError(
                f"Sync rad etildi: Destination kolleksiya model_digest mos kelmadi: mavjud={c_digest}, kutilgan={expected_digest}"
            )
        try:
            c_dim_int = _parse_int(c_dim) if c_dim is not None else None
        except (ValueError, TypeError, OverflowError):
            c_dim_int = None
        if c_dim is not None and c_dim_int != expected_dimension:
            raise ModelIdentityError(
                f"Sync rad etildi: Destination kolleksiya dimension mos kelmadi: mavjud={c_dim}, kutilgan={expected_dimension}"
            )

        has_identity = bool(
            c_model
            or c_digest
            or c_dim is not None
            or c_backend
            or c_schema is not None
        )
        if has_identity:
            if not c_backend or c_backend != expected_backend:
                raise ModelIdentityError(
                    f"Sync rad etildi: Destination kolleksiya backend mos kelmadi yoki yo'q: mavjud={c_backend}, kutilgan={expected_backend}"
                )
        if c_backend and c_backend != expected_backend:
            raise ModelIdentityError(
                f"Sync rad etildi: Destination kolleksiya backend mos kelmadi: mavjud={c_backend}, kutilgan={expected_backend}"
            )
        if c_schema is not None:
            try:
                c_schema_int = _parse_int(c_schema)
            except (ValueError, TypeError, OverflowError):
                c_schema_int = None
            if c_schema_int != expected_schema_version:
                raise ModelIdentityError(
                    f"Sync rad etildi: Destination kolleksiya schema_version mos kelmadi: mavjud={c_schema}, kutilgan={expected_schema_version}"
                )
        if c_dist is not None and expected_distance is not None:
            if not isinstance(c_dist, str) or c_dist.strip().lower() != expected_distance.strip().lower():
                raise ModelIdentityError(
                    f"Sync rad etildi: Destination kolleksiya distance mos kelmadi: mavjud={c_dist}, kutilgan={expected_distance}"
                )
        if c_norm is not None:
            try:
                c_norm_bool = _parse_bool(c_norm)
            except (ValueError, TypeError):
                raise ModelIdentityError(
                    f"Sync rad etildi: Destination kolleksiya normalization noto'g'ri formatda: {c_norm}"
                )
            if c_norm_bool != expected_normalization:
                raise ModelIdentityError(
                    f"Sync rad etildi: Destination kolleksiya normalization mos kelmadi: mavjud={c_norm}, kutilgan={expected_normalization}"
                )

    if manifest_meta and col_meta:
        from identity_utils import _check_identity_conflict
        _check_identity_conflict(manifest_meta, "manifest", col_meta, "kolleksiya")


def rebuild_keyword_index_subprocess(out_dir: Path, timeout: float = None, no_embedder: bool = True) -> None:
    """Run keyword_index.rebuild() in an isolated subprocess with GRAPHIFY_DB_PATH set to out_dir."""
    if timeout is None:
        timeout = float(os.environ.get("GRAPHIFY_KEYWORD_REBUILD_TIMEOUT", 1800.0))
    env = dict(os.environ)
    env["GRAPHIFY_DB_PATH"] = str(out_dir.resolve())
    env["GRAPHIFY_STORAGE_MODE"] = "local"
    if no_embedder:
        env["GRAPHIFY_KEYWORD_NO_EMBEDDER"] = "1"
        cmd = [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, '.'); import keyword_index; keyword_index.rebuild(no_embedder=True)",
        ]
    else:
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
    normalization: bool = False,
    base_identity: dict | None = None,
    backend: str | None = None,
    model: str | None = None,
    dimension: int | None = None,
    schema_version: int | None = None,
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

    if base_identity:
        identity = dict(base_identity)
    else:
        identity = {
            "schema_version": schema_version if schema_version is not None else 2,
            "backend": backend or "ollama",
            "model": model or "bge-m3",
            "model_digest": model_digest,
            "dimension": dimension if dimension is not None else EXPECTED_DIMENSION,
        }
    if schema_version is not None:
        identity["schema_version"] = schema_version
    if backend is not None:
        identity["backend"] = backend
    if model is not None:
        identity["model"] = model
    if dimension is not None:
        identity["dimension"] = dimension
    if model_digest is not None:
        identity["model_digest"] = model_digest

    identity["distance"] = distance
    identity["normalization"] = bool(normalization)
    identity["corpus_hash"] = corpus_hash
    identity["keyword_index_generation"] = kw_gen
    identity["total_records"] = len(records)
    identity["v1_sqlite_mtime"] = v1_mtime

    for k in [
        "schema_version",
        "backend",
        "model",
        "model_digest",
        "dimension",
        "distance",
        "normalization",
        "corpus_hash",
        "keyword_index_generation",
        "total_records",
        "v1_sqlite_mtime",
    ]:
        if f"graphify:{k}" in identity and k in identity:
            identity[f"graphify:{k}"] = identity[k]

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


def _validate_completion_marker_identity(marker_data: dict, current_identity: dict) -> bool:
    """Validate completion marker identity fields against current build. Return True if valid, False if mismatch or missing."""
    if not isinstance(marker_data, dict):
        return False
    from identity_utils import _parse_int, _parse_bool, _check_source_internal_conflict, _check_identity_conflict
    from embedders import ModelIdentityError
    try:
        _check_source_internal_conflict(marker_data, "completion_marker")
        _check_identity_conflict(marker_data, "completion_marker", current_identity, "current_identity")
    except ModelIdentityError:
        return False
    keys = ["model", "model_digest", "dimension", "backend", "distance", "normalization", "schema_version"]
    for k in keys:
        if k not in marker_data or marker_data[k] is None:
            return False
    if marker_data.get("model") != current_identity.get("model"):
        return False
    if marker_data.get("model_digest") != current_identity.get("model_digest"):
        return False
    try:
        if _parse_int(marker_data.get("dimension")) != _parse_int(current_identity.get("dimension")):
            return False
    except (ValueError, TypeError, OverflowError):
        return False
    if str(marker_data.get("backend")).lower() != str(current_identity.get("backend")).lower():
        return False
    if str(marker_data.get("distance")).lower() != str(current_identity.get("distance")).lower():
        return False
    try:
        if _parse_bool(marker_data.get("normalization")) != _parse_bool(current_identity.get("normalization", False)):
            return False
    except (ValueError, TypeError):
        return False
    try:
        if _parse_int(marker_data.get("schema_version")) != _parse_int(current_identity.get("schema_version")):
            return False
    except (ValueError, TypeError, OverflowError):
        return False
    return True


def _align_chroma_collection_configuration(db_path: Path, collection_name: str = "graphify") -> None:
    """Aligns Chroma collection embedding configuration with identity to prevent Chroma 1.5.9 conflict with 'default'."""
    sqlite_file = db_path / "chroma.sqlite3"
    if not sqlite_file.exists():
        return
    try:
        conn = sqlite3.connect(str(sqlite_file), timeout=10.0)
        try:
            cur = conn.cursor()
            cur.execute("SELECT id, schema_str FROM collections WHERE name = ?", (collection_name,))
            row = cur.fetchone()
            if row and row[1] and '"name":"default"' in row[1].replace(" ", ""):
                cid, s_str = row
                s_obj = json.loads(s_str)
                def fix_ef(obj):
                    if isinstance(obj, dict):
                        if obj.get("embedding_function") == {"type": "known", "name": "default", "config": {}}:
                            obj["embedding_function"] = {"type": "legacy"}
                        for v in obj.values():
                            fix_ef(v)
                    elif isinstance(obj, list):
                        for v in obj:
                            fix_ef(v)
                fix_ef(s_obj)
                cur.execute("UPDATE collections SET schema_str = ? WHERE id = ?", (json.dumps(s_obj), cid))
                conn.commit()
        finally:
            conn.close()
    except Exception:
        pass


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

    dest_manifest_file = out_dir / "graphify_manifest.json"
    dest_norm = False
    dest_dist = col_meta.get("hnsw:space", "l2")
    if dest_manifest_file.exists():
        try:
            with open(dest_manifest_file, "r", encoding="utf-8") as f:
                dest_m = json.load(f)
            if isinstance(dest_m, dict):
                from identity_utils import _parse_bool
                if "normalization" in dest_m:
                    dest_norm = _parse_bool(dest_m["normalization"])
                elif "graphify:normalization" in dest_m:
                    dest_norm = _parse_bool(dest_m["graphify:normalization"])
                if "distance" in dest_m:
                    dest_dist = dest_m["distance"]
                elif "graphify:distance" in dest_m:
                    dest_dist = dest_m["graphify:distance"]
        except Exception:
            pass
    elif (out_dir / "chroma.sqlite3").exists():
        try:
            conn = sqlite3.connect(f"file:{(out_dir / 'chroma.sqlite3').as_posix()}?mode=ro", uri=True, timeout=5.0)
            try:
                cur = conn.cursor()
                cur.execute("SELECT id FROM collections WHERE name = ?", (col_name,))
                c_row = cur.fetchone()
                if c_row:
                    cur.execute("SELECT key, bool_value, str_value FROM collection_metadata WHERE collection_id = ?", (c_row[0],))
                    for k, b, s in cur.fetchall():
                        if k in ("normalization", "graphify:normalization"):
                            from identity_utils import _parse_bool
                            val = bool(b) if b is not None else s
                            dest_norm = _parse_bool(val)
                        elif k in ("distance", "graphify:distance"):
                            dest_dist = s
            finally:
                conn.close()
        except Exception:
            pass

    # Strictly validate model identity before ANY mutation (read-only SQLite mode=ro first)
    validate_destination_persisted_identity(
        out_dir=out_dir,
        col_name=col_name,
        expected_model="bge-m3",
        expected_digest=model_digest,
        expected_dimension=EXPECTED_DIMENSION,
        client=None,
        expected_distance=dest_dist,
        expected_normalization=dest_norm,
    )

    # Invalidate completion status before the first destination mutation (Finding 3 & TOPILMA 5)
    complete_file = out_dir / "build_complete.json"
    had_complete_marker = complete_file.exists()
    complete_data_backup = None
    if had_complete_marker:
        try:
            with open(complete_file, "r", encoding="utf-8") as f:
                complete_data_backup = json.load(f)
        except Exception:
            complete_data_backup = None

    remove_completion_marker(out_dir)

    import chromadb

    client = chromadb.PersistentClient(path=str(out_dir))

    space = dest_dist
    sync_identity = {
        "schema_version": 2,
        "backend": "ollama",
        "model": "bge-m3",
        "model_digest": model_digest,
        "dimension": EXPECTED_DIMENSION,
        "distance": space,
        "normalization": dest_norm,
    }
    embed_fn = build_chroma_embedding_function(identity=sync_identity)
    _align_chroma_collection_configuration(out_dir, col_name)
    try:
        col = client.get_or_create_collection(
            name=col_name,
            metadata={"hnsw:space": space, **sync_identity},
            embedding_function=embed_fn,
        )
    except ValueError as e:
        if "Embedding function conflict" in str(e):
            _align_chroma_collection_configuration(out_dir, col_name)
            col = client.get_or_create_collection(
                name=col_name,
                metadata={"hnsw:space": space, **sync_identity},
                embedding_function=embed_fn,
            )
        else:
            raise
    # Persist identity metadata before any upsert (Finding 1 & TOPILMA 2)
    sync_modify_meta = {
        k: v for k, v in {
            **(col.metadata if isinstance(col.metadata, dict) else {}),
            **sync_identity,
        }.items()
        if not k.startswith("hnsw:")
    }
    col.modify(metadata=sync_modify_meta)

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
    kw_sqlite = out_dir / "keyword_index.sqlite3"
    needs_rebuild = bool(to_upsert or to_delete or not had_complete_marker or not kw_sqlite.exists() or not manifest_file.exists())

    if not needs_rebuild:
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                m_check = json.load(f)
            c_check = complete_data_backup or {}
            kw_mtime_gen = str(int(kw_sqlite.stat().st_mtime))
            if (
                m_check.get("corpus_hash") != corpus_hash
                or m_check.get("total_records") != len(v1_records)
                or m_check.get("model_digest") != model_digest
                or m_check.get("keyword_index_generation") != kw_mtime_gen
                or c_check.get("corpus_hash") != corpus_hash
                or c_check.get("keyword_index_generation") != kw_mtime_gen
                or not _validate_completion_marker_identity(c_check, sync_identity)
            ):
                needs_rebuild = True
        except Exception:
            needs_rebuild = True

    if not to_upsert and not to_delete and not needs_rebuild:
        try:
            verify_v2(col, v1_records, out_dir)
            if complete_data_backup and _validate_completion_marker_identity(complete_data_backup, sync_identity):
                tmp_path = out_dir / "build_complete.json.tmp"
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(complete_data_backup, f, indent=2, ensure_ascii=False)
                os.replace(tmp_path, complete_file)
                print("[sync] Baza allaqachon to'liq sinxron holatda va tasdiqlangan.", flush=True)
                return
            elif not complete_data_backup:
                write_build_complete_atomic(out_dir, sync_identity, started_at)
                print("[sync] Baza allaqachon to'liq sinxron holatda va tasdiqlangan.", flush=True)
                return
            else:
                needs_rebuild = True
        except Exception:
            needs_rebuild = True

    # Invalidate completion status before mutations, keyword rebuild, or manifest update (Finding 5 & TOPILMA 3)
    remove_completion_marker(out_dir)

    if to_delete:
        print(f"[sync] {len(to_delete)} ta yozuv o'chirilmoqda...", flush=True)
        col.delete(ids=to_delete)

    if to_upsert:
        embed_cfg = json.loads(json.dumps(BGE_M3_CONFIG))
        if dest_norm:
            embed_cfg["config"]["ollama"]["normalize"] = True
            embed_cfg["normalization"] = True
        embedder = build_embedder(embed_cfg)
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
        normalization=dest_norm,
    )

    # Update collection metadata in ChromaDB (Finding 5)
    col.modify(metadata={k: v for k, v in {**(col.metadata if isinstance(col.metadata, dict) else {}), **identity}.items() if not k.startswith("hnsw:")})

    # Write build_complete.json atomically
    write_build_complete_atomic(out_dir, identity, started_at)
    print("[sync] Sinxronizatsiya muvaffaqiyatli yakunlandi.", flush=True)


def _close_chroma_client(client) -> None:
    """Safely close PersistentClient to release SQLite locks and resources."""
    if client is None:
        return
    try:
        if hasattr(client, "close") and callable(client.close):
            client.close()
        elif hasattr(client, "_system") and hasattr(client._system, "stop"):
            client._system.stop()
    except Exception:
        pass


def run_renormalize(src_dir: Path, out_dir: Path, batch_size: int = 500) -> None:
    """Read v2 collection, normalize embeddings via L2, and write to a new database.

    - Source database is strictly read-only; no files in src_dir are modified.
    - Output path cannot equal source path or be a non-empty existing directory.
    - Destination collection is created with hnsw:space=cosine and normalization=True.
    - Verification checks: count match, 50 random samples vector = src/norm (1e-5),
      manifests, keyword index rebuilt via subprocess, build_complete.json written at the end.
    - Ollama is NEVER called.
    """
    src_dir = Path(src_dir)
    out_dir = Path(out_dir)

    if not src_dir.exists():
        raise FileNotFoundError(f"Manba v2 baza topilmadi: {src_dir}")
    if not (src_dir / "chroma.sqlite3").exists():
        raise FileNotFoundError(f"Manba v2 SQLite fayli topilmadi: {src_dir / 'chroma.sqlite3'}")

    src_res = src_dir.resolve()
    out_res = out_dir.resolve()

    if out_res == src_res:
        raise ValueError(f"Chiqish yo'li manba bilan bir xil bo'lishi mumkin emas: {out_dir}")
    if out_res.is_relative_to(src_res):
        raise ValueError(f"Chiqish yo'li manba papkasi ichida bo'lishi mumkin emas: {out_dir}")
    if src_res.is_relative_to(out_res):
        raise ValueError(f"Manba papkasi chiqish yo'li ichida bo'lishi mumkin emas: {out_dir}")

    if out_dir.exists():
        try:
            if any(out_dir.iterdir()):
                raise ValueError(f"Chiqish papkasi mavjud va bo'sh emas (ustidan yozish taqiqlangan): {out_dir}")
        except FileNotFoundError:
            pass

    out_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc)
    print(f"[renormalize] Boshlanmoqda: src={src_dir}, out={out_dir}, batch_size={batch_size}...", flush=True)

    src_client = None
    out_client = None
    temp_src_dir = None
    try:
        # Read source identity from manifest or collection
        src_manifest_file = src_dir / "graphify_manifest.json"
        src_identity = {}
        if src_manifest_file.exists():
            try:
                with open(src_manifest_file, "r", encoding="utf-8") as f:
                    src_identity = json.load(f)
            except Exception:
                src_identity = {}

        import shutil
        import tempfile
        temp_src_dir = Path(tempfile.mkdtemp(prefix="graphify_renormalize_src_"))
        shutil.copytree(src_dir, temp_src_dir, dirs_exist_ok=True)

        import chromadb
        src_client = chromadb.PersistentClient(path=str(temp_src_dir))
        try:
            src_col = src_client.get_collection("graphify")
        except Exception as e:
            raise RuntimeError(f"Manba kolleksiyasini ochib bo'lmadi ({src_dir}): {e}") from e

        if not src_identity and src_col.metadata:
            src_identity = dict(src_col.metadata)

        total_records = src_col.count()
        if total_records == 0:
            raise RuntimeError(f"Manba kolleksiyasida yozuvlar topilmadi: {src_dir}")

        # Prepare destination collection
        backend_val = src_identity.get("backend") if src_identity.get("backend") is not None else src_identity.get("graphify:backend", "ollama")
        model_val = src_identity.get("model") if src_identity.get("model") is not None else src_identity.get("graphify:model", "bge-m3")
        digest_val = src_identity.get("model_digest") if src_identity.get("model_digest") is not None else src_identity.get("graphify:model_digest", "")
        dim_val = src_identity.get("dimension") if src_identity.get("dimension") is not None else src_identity.get("graphify:dimension", EXPECTED_DIMENSION)
        schema_val = src_identity.get("schema_version") if src_identity.get("schema_version") is not None else src_identity.get("graphify:schema_version", 2)

        out_identity = dict(src_identity)
        out_identity.update({
            "backend": backend_val,
            "model": model_val,
            "model_digest": digest_val,
            "dimension": int(dim_val),
            "schema_version": int(schema_val),
            "distance": "cosine",
            "normalization": True,
        })
        for k in ["backend", "model", "model_digest", "dimension", "schema_version", "distance", "normalization"]:
            if f"graphify:{k}" in out_identity and k in out_identity:
                out_identity[f"graphify:{k}"] = out_identity[k]

        out_client = chromadb.PersistentClient(path=str(out_dir))
        out_meta = {k: v for k, v in out_identity.items() if not k.startswith("hnsw:")}
        out_meta["hnsw:space"] = "cosine"

        out_col = out_client.get_or_create_collection(
            name="graphify",
            metadata=out_meta,
        )
        col_modify_meta = {k: v for k, v in out_meta.items() if not k.startswith("hnsw:")}
        out_col.modify(metadata=col_modify_meta)
        _align_chroma_collection_configuration(out_dir, "graphify")

        import numpy as np
        all_records = []

        # Read and normalize in batches (Ollama is NEVER called)
        for offset in range(0, total_records, batch_size):
            chunk = src_col.get(
                limit=batch_size,
                offset=offset,
                include=["documents", "metadatas", "embeddings"],
            )
            c_ids = chunk["ids"]
            c_docs = chunk["documents"]
            c_metas = chunk["metadatas"]
            c_embs = chunk["embeddings"]

            if not c_ids:
                continue

            arr = np.array(c_embs, dtype=np.float32)
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            if np.any(norms == 0.0) or np.any(np.isnan(norms)) or np.any(np.isinf(norms)):
                raise ModelIdentityError("Nol vektor normallashtirib bo'lmaydi")
            normed = (arr / norms).tolist()

            out_col.add(
                ids=c_ids,
                documents=c_docs,
                metadatas=c_metas,
                embeddings=normed,
            )

            for rid, doc, meta in zip(c_ids, c_docs, c_metas):
                all_records.append({
                    "id": rid,
                    "document": doc if doc is not None else "",
                    "metadata": meta if meta is not None else {},
                })
            print(f"[renormalize] {min(offset + batch_size, total_records)}/{total_records} normallashtirildi...", flush=True)

        # 1. Total records check
        out_count = out_col.count()
        if out_count != total_records:
            raise AssertionError(f"Yozuvlar soni teng emas: chiqish={out_count}, manba={total_records}")

        # 2. Check 50 random records: output vector == src / norm (1e-5 precision)
        import random
        sample_k = min(50, total_records)
        all_ids = [r["id"] for r in all_records]
        sample_ids = random.sample(all_ids, sample_k) if total_records >= sample_k else list(all_ids)

        src_sample = src_col.get(ids=sample_ids, include=["embeddings"])
        out_sample = out_col.get(ids=sample_ids, include=["embeddings"])

        src_emb_map = dict(zip(src_sample["ids"], src_sample["embeddings"]))
        out_emb_map = dict(zip(out_sample["ids"], out_sample["embeddings"]))

        for sid in sample_ids:
            s_vec = np.array(src_emb_map[sid], dtype=np.float32)
            o_vec = np.array(out_emb_map[sid], dtype=np.float32)
            s_norm = float(np.linalg.norm(s_vec))
            if s_norm == 0.0 or np.isnan(s_norm):
                raise ModelIdentityError(f"Manbada nol vektor ({sid})")
            expected_vec = s_vec / s_norm
            if not np.allclose(o_vec, expected_vec, atol=1e-5, rtol=1e-5):
                raise AssertionError(f"Vektor normallashtirish mos kelmadi ({sid})")
            o_norm = float(np.linalg.norm(o_vec))
            if abs(o_norm - 1.0) > 1e-3:
                raise AssertionError(f"Chiqish vektori normasi 1.0 emas: {o_norm} ({sid})")

        # Clean up src_client references early
        _close_chroma_client(src_client)
        src_client = None

        # Align chroma collection configuration so embedding_function is legacy
        _align_chroma_collection_configuration(out_dir, "graphify")

        # 3. Subprocess rebuild keyword index without embedder
        rebuild_keyword_index_subprocess(out_dir)

        # 4. Verify all records and keyword index content
        verify_v2(out_col, all_records, out_dir)

        # 5. Write manifest and graphify_manifest.json with normalization: True
        kw_sqlite = out_dir / "keyword_index.sqlite3"
        kw_gen = str(int(kw_sqlite.stat().st_mtime)) if kw_sqlite.exists() else "gen1"
        corpus_hash = compute_corpus_hash(all_records)
        v1_mtime = src_identity.get("v1_sqlite_mtime", 0.0)

        digest_val = src_identity.get("model_digest") if src_identity.get("model_digest") is not None else src_identity.get("graphify:model_digest", out_identity.get("model_digest", ""))
        backend_val = src_identity.get("backend") if src_identity.get("backend") is not None else src_identity.get("graphify:backend", "ollama")
        model_val = src_identity.get("model") if src_identity.get("model") is not None else src_identity.get("graphify:model", "bge-m3")
        dim_val = int(src_identity["dimension"]) if "dimension" in src_identity else (int(src_identity["graphify:dimension"]) if "graphify:dimension" in src_identity else None)
        schema_val = int(src_identity["schema_version"]) if "schema_version" in src_identity else (int(src_identity["graphify:schema_version"]) if "graphify:schema_version" in src_identity else 2)

        final_identity = write_manifest_and_identity(
            out_dir=out_dir,
            records=all_records,
            v1_mtime=v1_mtime,
            started_at=started_at,
            model_digest=digest_val,
            distance="cosine",
            corpus_hash=corpus_hash,
            kw_gen=kw_gen,
            normalization=True,
            base_identity=src_identity,
            backend=backend_val,
            model=model_val,
            dimension=dim_val,
            schema_version=schema_val,
        )
        col_modify_meta = {k: v for k, v in final_identity.items() if not k.startswith("hnsw:")}
        out_col.modify(metadata=col_modify_meta)

        # 6. Write build_complete.json at the very end
        write_build_complete_atomic(out_dir, final_identity, started_at)
        print(f"[renormalize] Muvaffaqiyatli yakunlandi: {total_records} ta yozuv normallashtirildi va {out_dir} ga yozildi.", flush=True)
    finally:
        _close_chroma_client(src_client)
        _close_chroma_client(out_client)
        if temp_src_dir and Path(temp_src_dir).exists():
            import shutil
            shutil.rmtree(temp_src_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Build or sync Graphify v2 database")
    parser.add_argument("--v1-db", default="kg_db", help="Path to source v1 database (default: kg_db)")
    parser.add_argument("--out", default=None, help="Output directory for v2 database")
    parser.add_argument("--smoke", action="store_true", help="Smoke run: first 300 records into kg_db_v2_smoke/")
    parser.add_argument("--sync", action="store_true", help="Sync incremental changes from v1")
    parser.add_argument("--renormalize", action="store_true", help="Renormalize existing v2 database embeddings into new database")
    parser.add_argument("--src", default=None, help="Source v2 database path for --renormalize")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for embedding and DB writes")
    parser.add_argument("--max-workers", type=int, default=8, help="Worker threads for parallel Ollama calls")
    args = parser.parse_args()

    v1_dir = Path(args.v1_db)
    if args.renormalize:
        if not args.src:
            raise ValueError("--renormalize rejimi uchun --src <v2 baza> ko'rsatilishi shart")
        if not args.out:
            raise ValueError("--renormalize rejimi uchun --out ko'rsatilishi shart")
        src_path = Path(args.src)
        out_path = Path(args.out)
        batch_size = args.batch_size if args.batch_size != 50 else 500
        run_renormalize(src_path, out_path, batch_size=batch_size)
        return

    if args.out:
        out_dir = Path(args.out)
    elif args.smoke:
        out_dir = Path("kg_db_v2_smoke")
    else:
        out_dir = Path("kg_db_v2")

    # Finding 1 / G4: reject --out that resolves to source or overlaps
    v1_res = v1_dir.resolve()
    out_res = out_dir.resolve()
    if out_res == v1_res:
        raise ValueError(f"Chiqish yo'li v1 manbasi bilan bir xil bo'lishi mumkin emas: out={out_dir}, v1={v1_dir}")
    if out_res.is_relative_to(v1_res):
        raise ValueError(f"Chiqish yo'li v1 manbasi ichida bo'lishi mumkin emas: out={out_dir}, v1={v1_dir}")
    if v1_res.is_relative_to(out_res):
        raise ValueError(f"v1 manbasi chiqish yo'li ichida bo'lishi mumkin emas: out={out_dir}, v1={v1_dir}")

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

    # Fetch model digest without falling back to unknown
    model_digest = get_ollama_model_digest(
        BGE_M3_CONFIG["config"]["ollama"]["base_url"],
        BGE_M3_CONFIG["config"]["ollama"]["model"],
    )

    # Validate persisted collection AND manifest identity before any destination mutation (read-only SQLite first)
    validate_destination_persisted_identity(
        out_dir=out_dir,
        col_name=col_name,
        expected_model="bge-m3",
        expected_digest=model_digest,
        expected_dimension=EXPECTED_DIMENSION,
        client=None,
        expected_distance=col_meta.get("hnsw:space", "l2"),
    )

    # Remove completion marker before first mutation, keyword rebuild, or manifest update
    remove_completion_marker(out_dir)

    # 2. Setup v2 ChromaDB copying full collection metadata and configuration + identity
    import chromadb

    client = chromadb.PersistentClient(path=str(out_dir))
    space = col_meta.get("hnsw:space", "l2")

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
    build_modify_meta = {
        k: v for k, v in {
            **(col.metadata if isinstance(col.metadata, dict) else {}),
            **merged_metadata,
        }.items()
        if not k.startswith("hnsw:")
    }
    col.modify(metadata=build_modify_meta)

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
    col.modify(metadata={k: v for k, v in {**(col.metadata if isinstance(col.metadata, dict) else {}), **identity}.items() if not k.startswith("hnsw:")})

    # 7. Write build_complete.json atomically
    write_build_complete_atomic(out_dir, identity, started_at)
    print("[build_v2] Jarayon muvaffaqiyatli yakunlandi!", flush=True)


if __name__ == "__main__":
    main()
