import os
import sys
import math
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("GRAPHIFY_DIR", str(ROOT / "Graphify"))
os.environ.setdefault("ARCHIVE_DIR", str(ROOT / "Promtlarim" / "instagram"))

sys.path.insert(0, str(ROOT / "Graphify"))
import chromadb
from similarity import distance_to_similarity, make_scorer

def main():
    graphify_dir = Path(os.environ["GRAPHIFY_DIR"])
    new_db_path = graphify_dir / "kg_db-yangi"
    kutilgan_path = graphify_dir / "indeks-smoke-kutilgan.json"
    
    if not kutilgan_path.exists():
        print(f"Xato: {kutilgan_path.name} topilmadi. Kutilgan qiymatlar yo'q.")
        sys.exit(1)
        
    try:
        with open(kutilgan_path, "r", encoding="utf-8") as f:
            kutilgan = json.load(f)
    except Exception as e:
        print(f"Xato: {kutilgan_path.name} o'qishda xatolik: {e}")
        sys.exit(1)
        
    if not new_db_path.exists():
        print("Xato: kg_db-yangi topilmadi.")
        sys.exit(1)
        
    try:
        from embedders import build_chroma_embedding_function
        embed_fn = build_chroma_embedding_function()
    except Exception as e:
        print(f"Embedder yuklanmadi: {e}")
        sys.exit(1)
        
    client = chromadb.PersistentClient(path=str(new_db_path))
    col = client.get_collection("graphify", embedding_function=embed_fn)
    
    errors = []
    
    # Check 5 recent chunks with actual vector search
    try:
        all_res = col.get(limit=5, include=["documents"])
        if all_res and all_res["ids"]:
            for chunk_id, chunk_text in zip(all_res["ids"], all_res["documents"]):
                if chunk_text:
                    q_res = col.query(query_texts=[chunk_text], n_results=10)
                    if not q_res or not q_res["ids"] or not q_res["ids"][0]:
                        errors.append(f"Eng so'nggi chunk '{chunk_id}' vektor qidiruvi orqali topilmadi.")
                    elif chunk_id not in q_res["ids"][0]:
                        errors.append(f"Eng so'nggi chunk '{chunk_id}' o'z matni bilan qidirilganda top-10 da chiqmadi.")
    except Exception as e:
        errors.append(f"Oxirgi chunklarni tekshirishda xatolik: {e}")
    
    queries = [
        ("Dify", None, kutilgan["dify_id"]),
        ("customer retention", None, kutilgan["retention_id"]),
        (kutilgan["instagram_text"], None, f"instagram-file:{kutilgan['instagram_shortcode']}")
    ]
    
    mem_id = kutilgan["memory_id"]
    mem_res = col.get(ids=[mem_id])
    if not mem_res or not mem_res["ids"] or mem_res["ids"][0] != mem_id:
        errors.append(f"Memory so'rovi uchun aniq kutilgan ID ({mem_id}) topilmadi.")
    
    scorer = make_scorer(col)
    
    print("--- Smoke Test So'rovlar ---")
    for q_text, proj_filter, expected_target in queries:
        where_clause = {}
        if proj_filter:
            where_clause["project"] = proj_filter
            
        res = col.query(
            query_texts=[q_text],
            n_results=10,
            where=where_clause if where_clause else None,
            include=["metadatas", "documents", "distances", "embeddings"]
        )
        
        if not res or not res["ids"] or not res["ids"][0]:
            errors.append(f"So'rov natija bermadi: '{q_text}'")
            continue
            
        if proj_filter:
            for meta in res["metadatas"][0]:
                if meta.get("project") != proj_filter:
                    errors.append(f"Loyiha filtri ishlamadi: kutilgan={proj_filter}, amalda={meta.get('project')}")
                    
        found = False
        if expected_target.startswith("instagram-file:"):
            for meta in res["metadatas"][0]:
                if expected_target in meta.get("source", ""):
                    found = True
                    break
        else:
            if expected_target in res["ids"][0]:
                found = True
                
        if not found:
            errors.append(f"So'rov top-10 da kutilgan nishonni topmadi: '{q_text}' -> kutilgan={expected_target}")
                
        emb = res["embeddings"][0][0]
        if len(emb) != 384:
            errors.append(f"Embedding o'lchami noto'g'ri: {len(emb)} (384 kutilgan)")
        for val in emb:
            if not math.isfinite(val):
                errors.append("Embedding ichida chekli bo'lmagan (NaN/Inf) qiymat bor.")
                break
                
        if res["distances"] and res["distances"][0]:
            d = res["distances"][0][0]
            q_emb = embed_fn([q_text])[0]
            q_norm = np.linalg.norm(q_emb)
            d_norm = np.linalg.norm(emb)
            if q_norm > 0 and d_norm > 0:
                expected_sim = np.dot(q_emb, emb) / (q_norm * d_norm)
                expected_dist = 1.0 - expected_sim
                if abs(d - expected_dist) > 1e-4:
                    errors.append(f"Masofa Chroma va matematika orasida farq qilyapti. Kutildi={expected_dist:.4f}, Chroma={d:.4f}")
            
            clamped, raw = scorer(d, query_vec=q_emb, doc_vec=emb)
            if math.isnan(clamped) or math.isnan(raw):
                errors.append(f"Scorer NaN qaytardi: clamped={clamped}, raw={raw}")
                
    if errors:
        print("\nSmoke testda xatolar:")
        for err in errors:
            print(f"- {err}")
        sys.exit(1)
    else:
        print("\nSmoke test muvaffaqiyatli o'tdi.")
        sys.exit(0)

if __name__ == "__main__":
    main()
