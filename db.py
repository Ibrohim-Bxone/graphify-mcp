"""Graphify ChromaDB integratsiyasi — yagona baza boshqaruvi va kesh nazorati.

server.py va dashboard.py IKKALASI ham baza bilan shu modul orqali ishlaydi.
Kesh eskirishi kg_db/chroma.sqlite3 va chroma.sqlite3-wal ning mtime
o'zgarishidan aniqlanadi (diskni ortiqcha yuklamaslik uchun TTL bilan keshlangan).
"""

import functools
import ipaddress
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

import chromadb
from chromadb.api.client import SharedSystemClient
from embedders import build_chroma_embedding_function

STORAGE_MODE = os.environ.get("GRAPHIFY_STORAGE_MODE", "local")
if STORAGE_MODE not in ("local", "http"):
    raise ValueError("GRAPHIFY_STORAGE_MODE local yoki http bo'lishi kerak")
DB_PATH = os.environ.get("GRAPHIFY_DB_PATH", str(Path(__file__).parent / "kg_db"))
if STORAGE_MODE == "http":
    CHROMA_HOST = os.environ.get("GRAPHIFY_CHROMA_HOST", "127.0.0.1")
    if CHROMA_HOST == "localhost":
        CHROMA_HOST = "127.0.0.1"
    try:
        if "%" in CHROMA_HOST or not ipaddress.ip_address(CHROMA_HOST).is_loopback:
            raise ValueError
    except ValueError:
        raise ValueError("GRAPHIFY_CHROMA_HOST faqat loopback IP bo'lishi kerak") from None
    port = os.environ.get("GRAPHIFY_CHROMA_PORT", "8993")
    if not port.isascii() or not port.isdecimal() or not 1 <= int(port) <= 65535:
        raise ValueError("GRAPHIFY_CHROMA_PORT 1–65535 oralig'ida bo'lishi kerak")
    CHROMA_PORT = int(port)
COLLECTION = "graphify"
MTIME_CHECK_TTL = 2.0  # soniya: disk mtime tekshiruvini keshlaydigan vaqt oralig'i

_client = None
_col = None
_last_mtime = 0.0
_last_check_time = 0.0


def get_db_mtime() -> float:
    """kg_db ichidagi SQLite fayllarining (chroma.sqlite3 va chroma.sqlite3-wal) eng katta mtime qiymatini qaytaradi."""
    if STORAGE_MODE == "http":
        return 0.0
    mtimes = []
    sqlite_file = Path(DB_PATH) / "chroma.sqlite3"
    if sqlite_file.exists():
        try:
            mtimes.append(sqlite_file.stat().st_mtime)
        except OSError:
            pass
    wal_file = Path(DB_PATH) / "chroma.sqlite3-wal"
    if wal_file.exists():
        try:
            mtimes.append(wal_file.stat().st_mtime)
        except OSError:
            pass
    return max(mtimes) if mtimes else 0.0


def _check_identity_conflict(source_a: dict, name_a: str, source_b: dict, name_b: str) -> None:
    """Compares key identity fields between two sources and raises ModelIdentityError on mismatch."""
    keys_to_compare = [
        ("model", str),
        ("model_digest", str),
        ("dimension", int),
        ("backend", str),
    ]
    for key, cast_type in keys_to_compare:
        val_a = source_a.get(key) if source_a.get(key) is not None else source_a.get(f"graphify:{key}")
        val_b = source_b.get(key) if source_b.get(key) is not None else source_b.get(f"graphify:{key}")
        if val_a is not None and val_b is not None:
            if cast_type(val_a) != cast_type(val_b):
                from embedders import ModelIdentityError
                raise ModelIdentityError(
                    f"Model identifikatsiyasi nomuvofiq ({name_a} vs {name_b}): "
                    f"{key} ({val_a} != {val_b})"
                )


def _align_chroma_collection_configuration(db_path: Path, collection_name: str = COLLECTION) -> None:
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


def get_db_identity(client=None, db_path: str = None) -> dict | None:
    """Reads model identity metadata from collection metadata, SQLite, or graphify_manifest.json.

    Returns dict if DB has identity metadata (v2), or None if DB has no identity (v1).
    Raises ModelIdentityError if there is a conflict between manifest and collection identity metadata.
    """
    manifest_meta = {}
    col_meta = {}

    if STORAGE_MODE == "http":
        if client is not None:
            try:
                col = client.get_collection(COLLECTION)
                if col and col.metadata:
                    c_meta = col.metadata
                    if (
                        c_meta.get("schema_version") == 2
                        or c_meta.get("graphify:schema_version") == 2
                        or "model_digest" in c_meta
                        or "graphify:model_digest" in c_meta
                        or "model" in c_meta
                    ):
                        col_meta.update(c_meta)
            except Exception:
                pass
        if col_meta.get("schema_version") == 2 or col_meta.get("graphify:schema_version") == 2 or "model_digest" in col_meta or "graphify:model_digest" in col_meta or ("model" in col_meta and "dimension" in col_meta):
            return col_meta
        return None

    path = Path(db_path or DB_PATH)

    # 1. Check graphify_manifest.json
    manifest_file = path / "graphify_manifest.json"
    if manifest_file.exists():
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("schema_version") == 2 or "model_digest" in data or "backend" in data or "model" in data:
                manifest_meta.update(data)
        except Exception:
            pass

    # 2. Check client existing collection
    if client is not None:
        try:
            col = client.get_collection(COLLECTION)
            if col and col.metadata:
                c_meta = col.metadata
                if (
                    c_meta.get("schema_version") == 2
                    or c_meta.get("graphify:schema_version") == 2
                    or "model_digest" in c_meta
                    or "graphify:model_digest" in c_meta
                    or "model" in c_meta
                ):
                    col_meta.update(c_meta)
        except Exception:
            pass

    # 3. Check SQLite collection_metadata if local
    if STORAGE_MODE == "local" and not col_meta:
        sqlite_file = path / "chroma.sqlite3"
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
                    """, (COLLECTION,))
                    rows = cur.fetchall()
                    if rows:
                        s_meta = {}
                        for k, s, i, f, b in rows:
                            val = bool(b) if b is not None else i if i is not None else f if f is not None else s
                            s_meta[k] = val
                        if (
                            s_meta.get("schema_version") == 2
                            or s_meta.get("graphify:schema_version") == 2
                            or "model_digest" in s_meta
                            or "graphify:model_digest" in s_meta
                            or "model" in s_meta
                        ):
                            col_meta.update(s_meta)
                finally:
                    conn.close()
            except Exception:
                pass

    has_manifest_id = (
        manifest_meta.get("schema_version") == 2
        or "model_digest" in manifest_meta
        or ("model" in manifest_meta and "dimension" in manifest_meta)
    )
    has_col_id = (
        col_meta.get("schema_version") == 2
        or col_meta.get("graphify:schema_version") == 2
        or "model_digest" in col_meta
        or "graphify:model_digest" in col_meta
        or ("model" in col_meta and "dimension" in col_meta)
    )

    if has_manifest_id and has_col_id:
        _check_identity_conflict(manifest_meta, "manifest", col_meta, "kolleksiya")

    meta = {}
    if manifest_meta:
        meta.update(manifest_meta)
    if col_meta:
        meta.update(col_meta)

    if meta.get("schema_version") == 2 or meta.get("graphify:schema_version") == 2 or "model_digest" in meta or "graphify:model_digest" in meta or ("model" in meta and "dimension" in meta):
        return meta
    return None


def _open_collection(client):
    """Kolleksiyani mos embedder bilan ochadi yoki yaratadi.

    Agar bazada model identifikatsiyasi bo'lsa (v2):
    aynan o'sha embedder quriladi va tekshiriladi; mos kelmasa yoki xato bersa
    RAISE qiladi, hech qachon lokal MiniLM ga qaytmaydi (fail closed).

    Identifikatsiya bo'lmasa (v1):
    bugungi xatti-harakat saqlanadi (models_config bo'yicha yoki lokal default fallback).
    """
    identity = get_db_identity(client=client)
    if identity:
        if STORAGE_MODE == "local":
            _align_chroma_collection_configuration(Path(DB_PATH), COLLECTION)
        embed_fn = build_chroma_embedding_function(identity=identity)
        space = identity.get("distance") or identity.get("hnsw:space") or "cosine"
        kwargs = {"metadata": {"hnsw:space": space}}
        if embed_fn is not None:
            kwargs["embedding_function"] = embed_fn
        # Fail closed: never catch or fall back on identity failure!
        try:
            return client.get_or_create_collection(COLLECTION, **kwargs)
        except ValueError as e:
            if "Embedding function conflict" in str(e) and STORAGE_MODE == "local":
                _align_chroma_collection_configuration(Path(DB_PATH), COLLECTION)
                return client.get_or_create_collection(COLLECTION, **kwargs)
            raise

    embed_fn = build_chroma_embedding_function()
    kwargs = {"metadata": {"hnsw:space": "cosine"}}
    if embed_fn is not None:
        kwargs["embedding_function"] = embed_fn
    try:
        return client.get_or_create_collection(COLLECTION, **kwargs)
    except ValueError as e:
        if embed_fn is not None:
            print(
                f"[graphify] Ogohlantirish: models_config.json'dagi embedder mavjud "
                f"bazadagi bilan mos kelmadi ({e}); saqlangan embedderga qaytildi. "
                f"Almashtirish uchun kg_db/ ni tozalab qaytadan indekslang.",
                file=sys.stderr,
            )
            return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
        raise


def reopen():
    """Tashqi jarayon bazaga yozganda yoki kesh eskirganda klient va kolleksiyani qayta ochadi."""
    global _client, _col, _last_mtime, _last_check_time
    if STORAGE_MODE == "http":
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT, ssl=False)
        collection = _open_collection(client)
        _client, _col = client, collection
        return _col
    if _client is not None:
        try:
            if hasattr(_client, "_system"):
                _client._system.stop()
        except Exception:
            pass
    try:
        SharedSystemClient.clear_system_cache()
    except Exception:
        pass
    _client = chromadb.PersistentClient(path=DB_PATH)
    _col = _open_collection(_client)
    _last_mtime = get_db_mtime()
    _last_check_time = time.monotonic()
    return _col


def check_cache_freshness() -> None:
    """Diskdagi mtime o'zgargan bo'lsa kolleksiyani qayta ochadi.

    Diskka har chaqiruvda murojaat qilmaslik uchun tekshiruv MTIME_CHECK_TTL
    soniya oralig'ida keshlanadi.
    """
    global _last_check_time, _last_mtime
    now = time.monotonic()
    if _col is None:
        reopen()
        return

    if STORAGE_MODE == "http":
        return

    if now - _last_check_time < MTIME_CHECK_TTL:
        return

    _last_check_time = now
    current_mtime = get_db_mtime()
    if current_mtime != _last_mtime:
        reopen()


def notify_write() -> None:
    """Joriy jarayon bazaga yozganidan so'ng mtime keshini yangilaydi."""
    global _last_mtime, _last_check_time
    if STORAGE_MODE == "http":
        return
    _last_mtime = get_db_mtime()
    _last_check_time = time.monotonic()


def get_segment_seq_ids() -> tuple:
    """SQLite dan METADATA va VECTOR segmentlarining hozirgi max seq_id larini qaytaradi."""
    if STORAGE_MODE == "http":
        return None, None
    sqlite_file = Path(DB_PATH) / "chroma.sqlite3"
    if not sqlite_file.exists():
        return None, None
    try:
        conn = sqlite3.connect(str(sqlite_file), timeout=5.0)
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT s.scope, m.seq_id
                FROM segments s
                JOIN collections c ON s.collection = c.id
                JOIN max_seq_id m ON s.id = m.segment_id
                WHERE s.scope IN ('METADATA', 'VECTOR') AND c.name = ?
            """, (COLLECTION,))
            rows = dict(cur.fetchall())
            return rows.get("METADATA"), rows.get("VECTOR")
        finally:
            conn.close()
    except Exception:
        return None, None


def get_sync_threshold() -> int:
    """Kolleksiyaning sync_threshold parametrini qaytaradi (sukut bo'yicha 1000)."""
    if STORAGE_MODE == "http":
        return 1000
    sqlite_file = Path(DB_PATH) / "chroma.sqlite3"
    if not sqlite_file.exists():
        return 1000
    try:
        conn = sqlite3.connect(str(sqlite_file), timeout=5.0)
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT config_json_str FROM collections WHERE name = ?",
                (COLLECTION,),
            )
            row = cur.fetchone()
            if row and row[0]:
                cfg = json.loads(row[0])
                threshold = (
                    cfg.get("keys", {})
                    .get("#embedding", {})
                    .get("float_list", {})
                    .get("vector_index", {})
                    .get("config", {})
                    .get("hnsw", {})
                    .get("sync_threshold")
                )
                if threshold is not None:
                    return int(threshold)
        finally:
            conn.close()
    except Exception:
        pass
    return 1000


def flush_index(target_col=None) -> bool:
    """Agar metadata segmenti bilan vektor segmenti o'rtasida farq bo'lsa (yangi yozuvlar diskka tushmagan),
    chegarani kesib o'tish uchun mavjud yozuvlarni aynan o'z mazmuni va embeddinglari bilan
    upsert qilib, HNSW indeksining diskka (header.bin) to'liq yozilishini kafolatlaydi.

    Farq bo'lmasa (meta_seq <= vec_seq) — hech qanday upsert qilmaydi (0 ta ortiqcha amal).
    """
    if STORAGE_MODE == "http":
        return False  # Indeksni diskka yozish server zimmasida; lokal HNSW hiylasi ishlatilmaydi.
    meta_seq, vec_seq = get_segment_seq_ids()
    if meta_seq is None or vec_seq is None:
        return False

    if meta_seq <= vec_seq:
        return True

    if target_col is None:
        target_col = _col
        if target_col is None:
            target_col = get_collection()
    if target_col is None or target_col.count() == 0:
        return False

    sync_threshold = get_sync_threshold()
    target_seq = vec_seq + sync_threshold
    needed = max(1, target_seq - meta_seq)

    import yozuv
    with yozuv.lock():
        pool = target_col.get(limit=needed, include=["documents", "metadatas", "embeddings"])
        ids = pool.get("ids", [])
        if not ids:
            return False

        target_col.upsert(
            ids=ids,
            documents=pool.get("documents") if pool.get("documents") is not None else None,
            metadatas=pool.get("metadatas") if pool.get("metadatas") is not None else None,
            embeddings=pool.get("embeddings") if pool.get("embeddings") is not None else None,
        )

    notify_write()
    return True


def get_collection():
    """Hozirgi yangilangan kolleksiyani qaytaradi. Eskirgan bo'lsa avtomatik yangilaydi."""
    check_cache_freshness()
    return _col


def get_client():
    """Hozirgi klientni qaytaradi."""
    if _client is None:
        reopen()
    return _client


def db_retry(fn):
    """Kolleksiya eskirib xato berganda klient va kolleksiyani qayta ochib bir marta qayta urinish.

    So'rovdan oldin mtime bo'yicha kesh yangiligini tekshiradi; xatolik yuz berganda zaxira sifatida qayta ochib urinadi.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        check_cache_freshness()
        try:
            return fn(*args, **kwargs)
        except Exception as first_err:
            err_str = str(first_err).lower()
            if isinstance(first_err, TypeError) or not (
                isinstance(first_err, (chromadb.errors.ChromaError, OSError))
                or "id" in err_str
                or "stale" in err_str
            ):
                raise
            print(
                f"[graphify] Baza xatosi ({first_err}), kolleksiya qayta ochilib qayta urinilmoqda...",
                file=sys.stderr,
            )
            try:
                reopen()
                return fn(*args, **kwargs)
            except Exception as retry_err:
                return f"Baza xatosi: {retry_err}"
    return wrapper


class CollectionProxy:
    """Har bir operatsiyadan oldin kesh yangiligini tekshirib, joriy kolleksiyaga yo'naltiruvchi proksi."""

    def __getattr__(self, name):
        if name == "flush_index":
            return flush_index
        check_cache_freshness()
        attr = getattr(_col, name)
        if callable(attr):
            @functools.wraps(attr)
            def wrapper(*args, **kwargs):
                check_cache_freshness()
                try:
                    res = getattr(_col, name)(*args, **kwargs)
                    if name in ("add", "update", "upsert", "delete"):
                        notify_write()
                    return res
                except Exception as first_err:
                    err_str = str(first_err).lower()
                    if isinstance(first_err, TypeError) or not (
                        isinstance(first_err, (chromadb.errors.ChromaError, OSError))
                        or "id" in err_str
                        or "stale" in err_str
                    ):
                        raise
                    print(
                        f"[graphify] Baza xatosi ({first_err}), kolleksiya qayta ochilib qayta urinilmoqda...",
                        file=sys.stderr,
                    )
                    reopen()
                    res = getattr(_col, name)(*args, **kwargs)
                    if name in ("add", "update", "upsert", "delete"):
                        notify_write()
                    return res
            return wrapper
        return attr

    def __repr__(self):
        return repr(_col)


class ClientProxy:
    """Har doim joriy klientga yo'naltiruvchi proksi."""

    def __getattr__(self, name):
        if _client is None:
            reopen()
        return getattr(_client, name)

    def __repr__(self):
        return repr(_client)


# Dastlabki ishga tushirish
reopen()

# Tashqi modullar uchun qulay interfeys
col = CollectionProxy()
client = ClientProxy()
