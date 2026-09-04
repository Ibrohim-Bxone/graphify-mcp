"""Graphify ChromaDB integratsiyasi — yagona baza boshqaruvi va kesh nazorati.

server.py va dashboard.py IKKALASI ham baza bilan shu modul orqali ishlaydi.
Kesh eskirishi kg_db/chroma.sqlite3 va chroma.sqlite3-wal ning mtime
o'zgarishidan aniqlanadi (diskni ortiqcha yuklamaslik uchun TTL bilan keshlangan).
"""

import functools
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

import chromadb
from chromadb.api.client import SharedSystemClient
from embedders import build_chroma_embedding_function

DB_PATH = str(Path(__file__).parent / "kg_db")
COLLECTION = "graphify"
MTIME_CHECK_TTL = 2.0  # soniya: disk mtime tekshiruvini keshlaydigan vaqt oralig'i

_client = None
_col = None
_last_mtime = 0.0
_last_check_time = 0.0


def get_db_mtime() -> float:
    """kg_db ichidagi SQLite fayllarining (chroma.sqlite3 va chroma.sqlite3-wal) eng katta mtime qiymatini qaytaradi."""
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


def _open_collection(client):
    """Kolleksiyani mos embedder bilan ochadi yoki yaratadi.

    models_config.json'dagi embedder mavjud bazadagi bilan mos kelmasa yoki
    xato bersa, saqlangan/default embedderga qaytadi.
    """
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

    if now - _last_check_time < MTIME_CHECK_TTL:
        return

    _last_check_time = now
    current_mtime = get_db_mtime()
    if current_mtime != _last_mtime:
        reopen()


def notify_write() -> None:
    """Joriy jarayon bazaga yozganidan so'ng mtime keshini yangilaydi."""
    global _last_mtime, _last_check_time
    _last_mtime = get_db_mtime()
    _last_check_time = time.monotonic()


def get_segment_seq_ids() -> tuple:
    """SQLite dan METADATA va VECTOR segmentlarining hozirgi max seq_id larini qaytaradi."""
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
