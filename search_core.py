import re
import math
import sqlite3
import sys
from collections import defaultdict

# IDF-weighted coverage chegarasi: "yetarli dalil" deb topish uchun so'rov
# og'irligining kamida shuncha ulushi hujjatda uchrashi kerak (0.8 = 80%).
MIN_COVERAGE = 0.8

def extract_strict_constraints(query: str) -> list[str]:
    constraints = []
    # Years: 4 digits, possibly bounded
    years = re.findall(r'\b(19\d\d|20\d\d)\b', query)
    constraints.extend(years)
    # Quoted strings
    quotes = re.findall(r'"([^"]+)"', query)
    constraints.extend(quotes)
    return [c.lower() for c in constraints]

def compute_query_term_weights(query: str, kw_index) -> dict[str, float]:
    """So'rov terminlarining IDF og'irligini HISOBLAYDI — bu faqat so'rovning
    o'ziga bog'liq, nomzod matniga emas. Shuning uchun har so'rov uchun BIR
    MARTA chaqirilishi kerak (avval har nomzod uchun qaytadan chaqirilardi —
    N nomzod x M termin SQL so'rovi, sekinlikning asosiy sababi edi)."""
    import keyword_index
    q_tokens = keyword_index.tokenize(query)
    if not q_tokens:
        return {}

    conn = kw_index.conn
    N = conn.execute("SELECT COUNT(DISTINCT parent_id) FROM docs").fetchone()[0]
    if N == 0:
        return {}

    term_weights = {}
    for t in set(q_tokens):
        if kw_index.use_fts5:
            sql = "SELECT COUNT(DISTINCT d.parent_id) FROM fts_idx f JOIN docs d ON d.id = f.id WHERE f.text MATCH ?"
            try:
                df = conn.execute(sql, [f'"{t}"']).fetchone()[0]
            except sqlite3.OperationalError as e:
                print(f"[search_core] FTS5 so'rov xatosi ({e}), df=0 deb olindi", file=sys.stderr)
                df = 0
        else:
            sql = "SELECT COUNT(DISTINCT d.parent_id) FROM py_term_freqs p JOIN docs d ON d.id = p.doc_id WHERE p.term = ?"
            df = conn.execute(sql, [t]).fetchone()[0]

        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        term_weights[t] = max(0.01, idf)
    return term_weights

def coverage_for_text(term_weights: dict[str, float], text: str) -> tuple[float, int]:
    """Oldindan hisoblangan og'irliklar asosida BITTA matn uchun qamrovni
    hisoblaydi. Hech qanday SQL so'rov qilmaydi — sof matn taqqoslash."""
    import keyword_index
    if not term_weights:
        return 0.0, 0
    total_weight = sum(term_weights.values())
    if total_weight == 0:
        return 0.0, 0

    t_tokens = set(keyword_index.tokenize(text))
    matched_weight = 0.0
    matched_terms = 0
    for t, weight in term_weights.items():
        if t in t_tokens:
            matched_weight += weight
            matched_terms += 1

    return matched_weight / total_weight, matched_terms

def check_evidence(query: str, chunk_text: str, q_tokens: list[str],
                    constraints: list[str], term_weights: dict[str, float]) -> bool:
    """So'rov va nomzod bo'yicha oldindan hisoblangan (bir martalik) qiymatlar
    bilan ishlaydi — kw_index'ga SQL so'rov QILMAYDI, chunki
    compute_query_term_weights allaqachon chaqiruvchida bir marta bajarilgan."""
    import keyword_index
    t_tokens = set(keyword_index.tokenize(chunk_text))

    if len(q_tokens) == 1:
        return q_tokens[0] in t_tokens

    text_lower = chunk_text.lower()
    for c in constraints:
        if c not in text_lower:
            return False

    coverage, matched_terms = coverage_for_text(term_weights, chunk_text)

    if coverage >= MIN_COVERAGE and matched_terms >= 2:
        return True

    return False

def hybrid_search(query: str, top_k: int, where: dict | None, *, collection, kw_index, min_similarity: float = 0.10) -> dict:
    import similarity
    
    # 1. Fetch 50 from Vector
    vec_candidates = []
    meta = collection.metadata or {}
    space = meta.get("hnsw:space", "l2").lower()
    needs_embeddings = (space == "l2" and not similarity._check_unit_length(collection))
    
    include_fields = ["documents", "metadatas", "distances"]
    query_kwargs = {}
    scorer = similarity.make_scorer(collection)
    query_vec = None
    
    if needs_embeddings:
        include_fields.append("embeddings")
        emb_fn = getattr(collection, "_embedding_function", None)
        if emb_fn is None:
            from embedders import build_chroma_embedding_function
            emb_fn = build_chroma_embedding_function()
        query_vec = emb_fn([query])[0]
        query_kwargs["query_embeddings"] = [query_vec]
    else:
        query_kwargs["query_texts"] = [query]
        
    try:
        res = collection.query(
            n_results=min(50, getattr(collection, 'count', lambda: 50)() or 50),
            where=where,
            include=include_fields,
            **query_kwargs
        )
        if res and res.get("ids") and res["ids"][0]:
            docs = res["documents"][0] if res.get("documents") else []
            metas = res["metadatas"][0] if res.get("metadatas") else []
            dists = res["distances"][0] if res.get("distances") else []
            embeds = res["embeddings"][0] if needs_embeddings and res.get("embeddings") else [None] * len(docs)
            
            for i, doc_id in enumerate(res["ids"][0]):
                sim, raw_sim = scorer(dists[i], query_vec=query_vec, doc_vec=embeds[i])
                # MIN_SIMILARITY threshold applied ONLY to vector candidates
                # assuming MIN_SIMILARITY is usually imported, let's say 0.25 if not provided
                if sim >= min_similarity:
                    vec_candidates.append({
                        "id": doc_id,
                        "text": docs[i],
                        "meta": metas[i] or {},
                        "sim": sim,
                        "raw_sim": raw_sim
                    })
    except Exception as e:
        import sys
        print(f"[hybrid_search] Vector search error: {e}", file=sys.stderr)
        
    # 2. Fetch 50 from Keyword
    kw_candidates = []
    try:
        # Convert Chroma `where` dict to kwargs for keyword_index
        kw_filters = {}
        if where:
            if "project" in where:
                kw_filters["project"] = where["project"]
            if "kind" in where:
                kw_filters["kind"] = where["kind"]
            if "$and" in where:
                for cond in where["$and"]:
                    if "project" in cond:
                        kw_filters["project"] = cond["project"]
                    if "kind" in cond:
                        kw_filters["kind"] = cond["kind"]
        
        kw_res = kw_index.search(query, limit=50, filters=kw_filters)
        for r in kw_res:
            kw_candidates.append({
                "id": r["id"],
                "text": r["text"],
                "meta": {
                    "project": r["project"],
                    "kind": r["kind"],
                    "parent_id": r["parent_id"]
                },
                "score": r["score"]
            })
    except Exception as e:
        import sys
        print(f"[hybrid_search] Keyword search error: {e}", file=sys.stderr)

    # 3. Group by canonical source
    def get_parent_id(meta):
        return meta.get("parent_id") or meta.get("shortcode") or meta.get("source") or meta.get("id")

    vec_grouped = {}
    vec_all = defaultdict(list)
    for c in vec_candidates:
        pid = get_parent_id(c["meta"])
        vec_all[pid].append(c)
        if pid not in vec_grouped or c["sim"] > vec_grouped[pid]["sim"]:
            vec_grouped[pid] = c
            
    kw_grouped = {}
    kw_all = defaultdict(list)
    for c in kw_candidates:
        pid = get_parent_id(c["meta"])
        kw_all[pid].append(c)
        # FTS5 negative score => smaller is better
        if pid not in kw_grouped or c["score"] < kw_grouped[pid]["score"]:
            kw_grouped[pid] = c

    # Sort each group to get ranks
    vec_ranked = sorted(vec_grouped.values(), key=lambda x: x["sim"], reverse=True)
    kw_ranked = sorted(kw_grouped.values(), key=lambda x: x["score"])

    # 4. RRF Merging
    RRF_K = 60
    rrf_scores = defaultdict(float)
    
    for rank, c in enumerate(vec_ranked):
        pid = get_parent_id(c["meta"])
        rrf_scores[pid] += 1.0 / (RRF_K + rank + 1)
        
    for rank, c in enumerate(kw_ranked):
        pid = get_parent_id(c["meta"])
        rrf_scores[pid] += 1.0 / (RRF_K + rank + 1)
        
    # Combine results
    combined = []
    for pid, rrf in rrf_scores.items():
        # kalit so'z indeksi faqat project/kind/parent_id saqlaydi
        # (keyword_index.py sxemasi) — title/source/date/id/author yo'q.
        # Nomzod FAQAT kalit so'z orqali topilgan bo'lsa (vektor tomonda
        # umuman yo'q — masalan past cosine sababli chetlatilgan), to'liq
        # metadata Chroma'dan alohida o'qib olinadi, aks holda natija
        # "(untitled)" / "source: ?" bilan chiqib, foydalanuvchiga
        # atributsiz ko'rinadi.
        best_c = vec_grouped.get(pid)
        if best_c is None:
            best_c = kw_grouped[pid]
            try:
                enrich = collection.get(ids=[best_c["id"]], include=["metadatas", "documents"])
                if enrich.get("ids"):
                    best_c = dict(best_c)
                    if enrich.get("metadatas") and enrich["metadatas"][0]:
                        best_c["meta"] = enrich["metadatas"][0]
                    if enrich.get("documents") and enrich["documents"][0]:
                        best_c["text"] = enrich["documents"][0]
            except Exception as e:
                print(f"[search_core] Kalit so'z natijasini boyitishda xato: {e}", file=sys.stderr)

        all_chunks_for_pid = {}
        for c in vec_all.get(pid, []):
            all_chunks_for_pid[c["id"]] = c
        for c in kw_all.get(pid, []):
            all_chunks_for_pid[c["id"]] = c
            
        if best_c["id"] in all_chunks_for_pid:
            del all_chunks_for_pid[best_c["id"]]
            
        boshqa = list(all_chunks_for_pid.values())
        
        item = {
            "id": best_c["id"],
            "text": best_c["text"],
            "meta": best_c["meta"],
            "rrf_score": rrf,
            "similarity": vec_grouped[pid]["sim"] if pid in vec_grouped else 0.0,
            "boshqa_boshqalar": boshqa
        }
        combined.append(item)
        
    combined.sort(key=lambda x: x["rrf_score"], reverse=True)

    # 5. Apply "Insufficient evidence" rule
    # So'rovga bog'liq narsalar (tokenlar, qat'iy cheklovlar, IDF og'irliklari)
    # shu yerda BIR MARTA hisoblanadi va har nomzodga uzatiladi — avval har
    # nomzod uchun qaytadan SQL bilan hisoblanardi (N nomzod x M termin so'rov,
    # ko'p natijali so'rovlarda soniyalab kechikish sababi edi).
    import keyword_index as _kw_mod
    q_tokens = _kw_mod.tokenize(query)
    constraints = extract_strict_constraints(query)
    term_weights = compute_query_term_weights(query, kw_index)

    final_results = []
    related_candidates = []

    for c in combined:
        if check_evidence(query, c["text"], q_tokens, constraints, term_weights):
            if len(final_results) < top_k:
                final_results.append(c)
            else:
                related_candidates.append(c)
        else:
            related_candidates.append(c)
            
    if not final_results:
        return {
            "results": [],
            "status": "insufficient_evidence",
            "related_candidates": related_candidates
        }
        
    return {
        "results": final_results,
        "status": "ok",
        "related_candidates": related_candidates
    }
