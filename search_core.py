import os
import re
import math
import sqlite3
import sys
from collections import defaultdict
import search_policy

DENSE_WEIGHT = 1.0
LEXICAL_WEIGHT = 0.2
RRF_K = 60
RERANK_ALPHA = 0.5
RERANK_TOP_N = 20

# IDF-weighted coverage chegarasi: "yetarli dalil" deb topish uchun so'rov
# og'irligining kamida shuncha ulushi hujjatda uchrashi kerak (0.5 = 50%).
# 0.65 tanlandi (2026-11-02, 200 so'rov, kg_db_v2d): R@1 0.30, R@10 0.36, noto'g'ri topildi 0.12;
# 0.5 recall'ni oshiradi, lekin real xotira so'rovlarida top-1 ni yomonlashtirdi.
MIN_COVERAGE = 0.65

FUNCTIONAL_WORDS = {
    "nima", "nimaga", "qanday", "qanaqa", "qaysi", "qayerda", "qachon", "kim", "nega",
    "qiladi", "qilish", "ishlatiladi", "ishlatish", "haqida", "ber", "bering", "ayt", "aytib",
    "tushuntir", "ko'rsat", "va", "bilan", "uchun", "bu", "shu", "ma'lumot", "ma'lumotlar",
    "what", "how", "why", "which", "where", "when", "who", "does", "do", "is", "are",
    "used", "use", "using", "about", "tell", "me", "explain", "show",
    "the", "a", "an", "and", "for", "of", "to", "in", "it", "can", "you", "i"
}

try:
    import keyword_index as _kw_mod
    _new_fws = set()
    for fw in FUNCTIONAL_WORDS:
        _new_fws.update(_kw_mod.tokenize(fw))
    FUNCTIONAL_WORDS.update(_new_fws)
except Exception:
    pass

def _light_stem(word: str) -> str:
    """O'zbek qo'shimchalarini kesuvchi yengil stemmer. FTS uchun emas, faqat qamrov hisoblash uchun."""
    suffixes = [
        "larning", "larni", "lari", "lar", "ning", "niki", "dagi", "dan", "dek", "cha", "ni", "ga", "da",
        "imiz", "ingiz", "si",
        "amiz", "aymiz", "adimi", "ydimi", "adi", "aydi", "ganmiz", "ganmi", "gan", "dimi", "moqda", "moq", "yapti", "yotgan", "masligi", "ligi", "lik", "ish", "ib",
        "di", "li", "yot"
    ]
    suffixes.sort(key=len, reverse=True)
    vowels = "aeiouAEIOU"  # "'" unli emas: "tarmog'i" da oldingi harf g' (undosh)
    
    while len(word) > 3:
        changed = False
        for suf in suffixes:
            if word.endswith(suf) and len(word) - len(suf) >= 3:
                if suf in ("aymiz", "aydi"):
                    word = word[:-(len(suf)-1)] # O'zakdagi 'a' ni saqlaymiz
                else:
                    word = word[:-len(suf)]
                changed = True
                break
                
        if not changed:
            if word.endswith("i") and not word.endswith("mi"):
                stem = word[:-1]
                # -i faqat o'zak >= 5 harf va oldingi harf undosh bo'lsa
                if len(stem) >= 5 and stem[-1] not in vowels:
                    word = stem
                    changed = True
            elif word.endswith("mi"):
                stem = word[:-2]
                # -mi faqat o'zak kamida 3-4 harf bo'lsa (ingliz so'zlari buzilmasligi uchun, masalan 3)
                if len(stem) >= 3:
                    word = stem
                    changed = True
                    
        if not changed:
            break
    return word

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
        
    meaningful = [t for t in q_tokens if t not in FUNCTIONAL_WORDS]
    if meaningful:
        q_tokens = meaningful

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
    stemmed_t_tokens = {_light_stem(t) for t in t_tokens}
    
    matched_weight = 0.0
    matched_terms = 0
    for t, weight in term_weights.items():
        if _light_stem(t) in stemmed_t_tokens:
            matched_weight += weight
            matched_terms += 1

    return matched_weight / total_weight, matched_terms

def check_evidence(query: str, chunk_text: str, q_tokens: list[str],
                    constraints: list[str], term_weights: dict[str, float], title: str = "", date: str = "") -> bool:
    """So'rov va nomzod bo'yicha oldindan hisoblangan (bir martalik) qiymatlar
    bilan ishlaydi — kw_index'ga SQL so'rov QILMAYDI, chunki
    compute_query_term_weights allaqachon chaqiruvchida bir marta bajarilgan."""
    import keyword_index
    combined_text = chunk_text + (" " + title if title else "")
    if re.fullmatch(r'\d{4}-\d{2}-\d{2}(?:T.*)?', str(date)):
        combined_text += " " + str(date)  # Search date metadata; this does not verify time-sensitive claims.
    t_tokens = set(keyword_index.tokenize(combined_text))

    if len(q_tokens) == 1:
        return q_tokens[0] in t_tokens

    text_lower = combined_text.lower()
    for c in constraints:
        if c not in text_lower:
            return False

    coverage, matched_terms = coverage_for_text(term_weights, combined_text)
    
    meaningful_q = set(t for t in q_tokens if t not in FUNCTIONAL_WORDS)
    required_matches = 1 if len(meaningful_q) == 1 else 2

    if coverage >= MIN_COVERAGE and matched_terms >= required_matches:
        return True

    return False

def hybrid_search(query: str, top_k: int, where: dict | None, *, collection, kw_index, min_similarity: float = 0.10) -> dict:
    import similarity
    
    # 1. Fetch 50 from Vector
    vec_candidates = []
    vector_degraded = False
    vector_degraded_error = None
    space = similarity.collection_space(collection)
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
        count = collection.count()
    except Exception:
        count = 50

    try:
        if count == 0:
            res = {}
        else:
            res = collection.query(
                n_results=min(max(50, top_k * 5), count),
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
                candidate_meta = dict(metas[i] or {}, id=doc_id)
                # MIN_SIMILARITY threshold applied ONLY to vector candidates
                # assuming MIN_SIMILARITY is usually imported, let's say 0.25 if not provided
                if (sim >= min_similarity and search_policy.matches_where(candidate_meta, where)
                        and search_policy.usable(candidate_meta, docs[i])):
                    vec_candidates.append({
                        "id": doc_id,
                        "text": docs[i],
                        "meta": candidate_meta,
                        "sim": sim,
                        "raw_sim": raw_sim
                    })
    except Exception as e:
        import sys
        err_msg = str(e).lower()
        if any(s in err_msg for s in ["stale", "does not exist", "not found", "error finding id"]):
            raise
        vector_degraded = True
        vector_degraded_error = str(e)
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
        kw_ids = [r["id"] for r in kw_res]
        if kw_ids:
            try:
                exist_res = collection.get(ids=kw_ids, include=["metadatas"])
                exist_ids = set(exist_res.get("ids", []))
                exist_metas = {exist_res["ids"][i]: exist_res["metadatas"][i] for i in range(len(exist_res["ids"]))}
            except Exception:
                exist_ids = set()
                exist_metas = {}
                
            for r in kw_res:
                if r["id"] in exist_ids:
                    meta_dict = {
                        "project": r["project"],
                        "kind": r["kind"],
                        "parent_id": r["parent_id"],
                        "id": r["id"]
                    }
                    if r["id"] in exist_metas and exist_metas[r["id"]]:
                        meta_dict.update(exist_metas[r["id"]])
                    meta_dict["id"] = r["id"]
                    if (not search_policy.matches_where(meta_dict, where)
                            or not search_policy.usable(meta_dict, r["text"])):
                        continue
                    
                    kw_candidates.append({
                        "id": r["id"],
                        "text": r["text"],
                        "meta": meta_dict,
                        "score": r["score"]
                    })
    except Exception as e:
        import sys
        print(f"[hybrid_search] Keyword search error: {e}", file=sys.stderr)

    # 3. Group by canonical source
    def get_parent_id(meta):
        # save_memory yozuvlarida parent_id yo'q, source="save_memory" — source ham tekshiriladi,
        # aks holda BARCHA xotiralar bitta guruhga birlashib, faqat bittasi natijaga chiqardi.
        return search_policy.source_key(meta)

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
    dense_w = float(os.environ.get("GRAPHIFY_DENSE_WEIGHT", str(DENSE_WEIGHT)))
    lexical_w = float(os.environ.get("GRAPHIFY_LEXICAL_WEIGHT", str(LEXICAL_WEIGHT)))
    rrf_k = int(os.environ.get("GRAPHIFY_RRF_K", str(RRF_K)))
    rrf_scores = defaultdict(float)
    
    for rank, c in enumerate(vec_ranked):
        pid = get_parent_id(c["meta"])
        rrf_scores[pid] += dense_w / (rrf_k + rank + 1)
        
    for rank, c in enumerate(kw_ranked):
        pid = get_parent_id(c["meta"])
        rrf_scores[pid] += lexical_w / (rrf_k + rank + 1)
        
    # Combine results
    import keyword_index as _kw_mod
    q_tokens = _kw_mod.tokenize(query)
    constraints = extract_strict_constraints(query)
    term_weights = compute_query_term_weights(query, kw_index)

    def evidence(candidate):
        meta = candidate.get("meta") or {}
        return check_evidence(query, candidate["text"], q_tokens, constraints, term_weights,
                              title=meta.get("title", ""), date=meta.get("date", ""))

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
                else:
                    continue
            except Exception as e:
                print(f"[search_core] Kalit so'z natijasini boyitishda xato: {e}", file=sys.stderr)
                continue

        all_chunks_for_pid = {}
        for c in vec_all.get(pid, []):
            all_chunks_for_pid[c["id"]] = c
        for c in kw_all.get(pid, []):
            all_chunks_for_pid[c["id"]] = c

        if not evidence(best_c):
            # A generic vector representative must not hide an exact matching sibling chunk.
            best_c = next((c for c in all_chunks_for_pid.values() if evidence(c)), best_c)
            
        if best_c["id"] in all_chunks_for_pid:
            del all_chunks_for_pid[best_c["id"]]
            
        boshqa = list(all_chunks_for_pid.values())
        
        item = {
            "id": best_c["id"],
            "text": best_c["text"],
            "meta": best_c["meta"],
            "rrf_score": rrf,
            "fused_score": rrf,
            "rerank_score": None,
            "similarity": vec_grouped[pid]["sim"] if pid in vec_grouped else 0.0,
            "boshqa_boshqalar": boshqa
        }
        combined.append(item)
        
    combined.sort(key=lambda x: x["fused_score"], reverse=True)

    # 4.5 Rerank stage
    # Sukut bo'yicha o'chiq: 2026-10-29 o'lchovida CPU'da p50 19 s, 45% so'rovda 20 s ichida ulgurmadi,
    # MRR o'sishi ~0.02 (migrate_v2/out-precision). GPU bilan qayta o'lchanmaguncha faqat GRAPHIFY_RERANK=1 bilan.
    rerank_env = os.environ.get("GRAPHIFY_RERANK", "0").strip().lower()
    if rerank_env in ("0", "false", "off", "no"):
        rerank_status = "off"
        rerank_reason = "GRAPHIFY_RERANK=0"
    else:
        try:
            import reranker as reranker_mod
            r_inst, err = reranker_mod.get_reranker_with_reason()
        except Exception as e:
            r_inst = None
            err = f"import error: {e}"

        if r_inst is None:
            rerank_status = "unavailable"
            rerank_reason = err or "reranker load failed"
        elif not combined:
            rerank_status = "on"
            rerank_reason = None
        else:
            alpha = float(os.environ.get("GRAPHIFY_RERANK_ALPHA", str(RERANK_ALPHA)))
            top_n = int(os.environ.get("GRAPHIFY_RERANK_TOP_N", str(RERANK_TOP_N)))
            to_rerank = combined[:top_n]
            pairs = [(query, c["text"]) for c in to_rerank]
            res = r_inst.score_pairs(pairs)
            if isinstance(res, tuple):
                scores, err = res
            else:
                scores, err = res, getattr(r_inst, "last_error", None)
            if scores is None:
                rerank_status = "unavailable"
                rerank_reason = err or "rerank failed"
            else:
                rerank_status = "on"
                rerank_reason = None
                fused_vals = [c["fused_score"] for c in to_rerank]
                f_max = max(fused_vals) if fused_vals else 1.0
                for idx, c in enumerate(to_rerank):
                    c["rerank_score"] = scores[idx]
                    fused_norm = c["fused_score"] / f_max if f_max > 0 else 1.0
                    c["score"] = alpha * c["rerank_score"] + (1.0 - alpha) * fused_norm
                to_rerank.sort(key=lambda x: x["score"], reverse=True)
                combined = to_rerank + combined[top_n:]

    # 5. Apply "Insufficient evidence" rule
    # So'rovga bog'liq narsalar (tokenlar, qat'iy cheklovlar, IDF og'irliklari)
    # shu yerda BIR MARTA hisoblanadi va har nomzodga uzatiladi — avval har
    # nomzod uchun qaytadan SQL bilan hisoblanardi (N nomzod x M termin so'rov,
    # ko'p natijali so'rovlarda soniyalab kechikish sababi edi).

    final_results = []
    related_candidates = []

    for c in combined:
        title = ""
        if c.get("meta") and isinstance(c["meta"], dict):
            title = c["meta"].get("title", "")
        if evidence(c):
            if len(final_results) < top_k:
                final_results.append(c)
            else:
                related_candidates.append(c)
        else:
            related_candidates.append(c)
            
    if not final_results:
        ret = {
            "results": [],
            "status": "insufficient_evidence",
            "related_candidates": related_candidates,
            "rerank": rerank_status,
            "rerank_reason": rerank_reason,
        }
    else:
        ret = {
            "results": final_results,
            "status": "ok",
            "related_candidates": related_candidates,
            "rerank": rerank_status,
            "rerank_reason": rerank_reason,
        }

    if vector_degraded:
        ret["degraded"] = "vector_search_failed"
        ret["degraded_error"] = str(vector_degraded_error)

    return ret
