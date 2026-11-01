"""O'xshashlik hisobi moduli."""

import sys
import numpy as np

_warned_non_unit = False
_unit_length_cache = {}

def _get_mutation_signal(collection, raw_col, is_db_col=False):
    """Kolleksiyaning mutatsiya signalini oladi yoki None qaytaradi."""
    for target in (collection, raw_col):
        if target is None:
            continue
        for attr in ("mutation_signal", "seq_id", "max_seq_id", "write_generation", "generation", "version", "max_id"):
            val = getattr(target, attr, None)
            if callable(val):
                try:
                    val = val()
                except Exception:
                    continue
            if val is not None:
                return val
        # Chroma 1.5.9 lokal kolleksiyasida _model.log_position/version yozuvdan keyin ham 0 qoladi
        # (2026-10-29 sinovi) — ishonchli signal emas, shuning uchun ishlatilmaydi va kesh chetlab o'tiladi.

    if is_db_col:
        db = sys.modules.get("db")
        if db is not None and hasattr(db, "get_db_mtime"):
            try:
                mtime = db.get_db_mtime()
                if mtime > 0:
                    return mtime
            except Exception:
                pass

    return None

def _check_unit_length(collection):
    """Kolleksiyadan namuna vektorlar olib, birlik uzunlikda ekanini tekshiradi."""
    global _warned_non_unit
    
    raw_col = collection
    is_db_col = False
    db = sys.modules.get("db")
    if db is not None and (collection is getattr(db, "col", None) or collection is getattr(db, "_col", None)):
        raw_col = db.get_collection()
        is_db_col = True

    count_fn = getattr(raw_col, "count", None) or getattr(collection, "count", None)
    count = None
    count_failed = False
    if callable(count_fn):
        try:
            count = count_fn()
        except Exception:
            count_failed = True

    col_id = (getattr(collection, "id", None) or getattr(raw_col, "id", None) or
              getattr(collection, "name", None) or getattr(raw_col, "name", None))
    mutation_signal = _get_mutation_signal(collection, raw_col, is_db_col=is_db_col)

    if col_id is None or count_failed:
        cache_key = None
    elif count is not None:
        if mutation_signal is not None:
            cache_key = (str(col_id), count, mutation_signal)
        else:
            cache_key = None
    else:
        cache_key = (str(col_id), mutation_signal) if mutation_signal is not None else None

    if cache_key is not None and cache_key in _unit_length_cache:
        return _unit_length_cache[cache_key]
    try:
        res = raw_col.get(limit=5, include=["embeddings"])
        embeddings = res.get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            if cache_key is not None:
                _unit_length_cache[cache_key] = True
            return True
        
        for vec in embeddings:
            norm = np.linalg.norm(vec)
            if abs(norm - 1.0) >= 1e-3:
                if not _warned_non_unit:
                    print("[similarity] Ogohlantirish: L2 space tanlangan, lekin vektorlar birlik uzunlikda emas. "
                          "O'xshashlik sekinroq (to'g'ridan-to'g'ri dot product bilan) hisoblanadi.", file=sys.stderr)
                    _warned_non_unit = True
                if cache_key is not None:
                    _unit_length_cache[cache_key] = False
                return False
        if cache_key is not None:
            _unit_length_cache[cache_key] = True
        return True
    except Exception as e:
        print(f"[similarity] Embedding o'qishda xatolik: {e}", file=sys.stderr)
        return False

def distance_to_similarity(distance: float, space: str, *, query_vec=None, doc_vec=None) -> tuple[float, float]:
    """Masofani o'xshashlikka o'tkazadi.
    
    Returns:
        (clamped_sim, raw_sim): qisilgan qiymat [0, 1] va xom qiymat.
    """
    space = space.lower()
    if space == "cosine" or space == "ip":
        sim = 1.0 - distance
    elif space == "l2":
        # Check if we need to use query_vec and doc_vec
        # This function assumes that if query_vec and doc_vec are provided, we should compute cosine similarity directly
        # But wait, make_scorer handles whether to pass query_vec and doc_vec based on unit length.
        if query_vec is not None and doc_vec is not None:
            q_norm = np.linalg.norm(query_vec)
            d_norm = np.linalg.norm(doc_vec)
            if q_norm == 0 or d_norm == 0:
                sim = 0.0
            else:
                sim = np.dot(query_vec, doc_vec) / (q_norm * d_norm)
        else:
            sim = 1.0 - distance / 2.0
    else:
        raise ValueError(f"Noma'lum space: {space}")

    clamped_sim = max(0.0, min(1.0, sim))
    return float(clamped_sim), float(sim)

def collection_space(collection) -> str:
    """Kolleksiya fazosi (space) ni aniqlaydi: configuration_json -> metadata -> raise."""
    config = getattr(collection, "configuration_json", None)
    if isinstance(config, dict):
        hnsw_config = config.get("hnsw")
        if isinstance(hnsw_config, dict):
            space = hnsw_config.get("space")
            if space:
                return str(space).lower()

    metadata = getattr(collection, "metadata", None) or {}
    space = metadata.get("hnsw:space")
    if not space and "hnsw" in metadata and isinstance(metadata["hnsw"], dict):
        space = metadata["hnsw"].get("space")

    if space:
        return str(space).lower()

    raise ValueError("hnsw:space topilmadi, l2 deb taxmin qilinmaydi")

def make_scorer(collection):
    """Kolleksiya uchun mos o'xshashlik hisoblagichni yaratadi."""
    space = collection_space(collection)
    is_unit = True
    if space.lower() == "l2":
        is_unit = _check_unit_length(collection)
    
    def scorer(distance, query_vec=None, doc_vec=None):
        if space.lower() == "l2" and not is_unit:
            if query_vec is None or doc_vec is None:
                raise ValueError("Birlik emas L2 uchun query_vec va doc_vec kerak")
            return distance_to_similarity(distance, space, query_vec=query_vec, doc_vec=doc_vec)
        return distance_to_similarity(distance, space)
    
    return scorer
