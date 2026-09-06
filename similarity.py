"""O'xshashlik hisobi moduli."""

import sys
import numpy as np

_warned_non_unit = False
_unit_length_cache = {}

def _check_unit_length(collection):
    """Kolleksiyadan namuna vektorlar olib, birlik uzunlikda ekanini tekshiradi."""
    global _warned_non_unit
    
    # Kesh Chroma'ning barqaror UUID'i yoki nomi bo'yicha. Python'ning o'z
    # id() funksiyasi ISHLATILMAYDI: obyekt GC qilingandan keyin xotiradagi
    # o'sha manzil boshqa (butunlay boshqa bazadagi) kolleksiyaga berilishi
    # mumkin va eski (noto'g'ri) natija qaytib qoladi. Barqaror identifikator
    # yo'q bo'lgan obyektlar (masalan testdagi soxta kolleksiya) umuman
    # keshlanmaydi — har safar qayta hisoblanadi, bu xavfsizroq.
    cache_key = getattr(collection, "id", None) or getattr(collection, "name", None)
    if cache_key is not None and cache_key in _unit_length_cache:
        return _unit_length_cache[cache_key]

    try:
        res = collection.get(limit=5, include=["embeddings"])
        embeddings = res.get("embeddings")
        if not embeddings:
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
    except Exception:
        return True

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

def make_scorer(collection):
    """Kolleksiya uchun mos o'xshashlik hisoblagichni yaratadi."""
    metadata = collection.metadata or {}
    
    space = metadata.get("hnsw:space")
    if not space and "hnsw" in metadata and isinstance(metadata["hnsw"], dict):
        space = metadata["hnsw"].get("space")
        
    if not space:
        raise ValueError("hnsw:space topilmadi, l2 deb taxmin qilinmaydi")
        
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
