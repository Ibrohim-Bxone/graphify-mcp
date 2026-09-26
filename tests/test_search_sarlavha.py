import pytest
from test_yozuv import temp_yozuv
import search_core
import keyword_index
import yozuv

def test_search_sarlavha_va_qoshimchalar(temp_yozuv):
    col, db_dir = temp_yozuv
    fts = keyword_index.get_index()

    # Korpus hujjatlari
    docs = [
        "Bu xotira faqat sarlavhada muhim narsalarni saqlaydi, matni qisqa.",
        "Eskiz tizimiga kirish va tiklash yo'riqnomasi bor, uni o'qib chiqing.",
        "Serverdagi api dokumentatsiya. Bu ai xizmatiga ulangan, mini ilova.",
        "Vertex va Supabase cloud ulangan texnologiyalaridir.",
        "L0A darslari YouTube'ga yuklagan tayyor."
    ]
    metas = [
        {"id": "mem_123", "title": "DJIM HoReCa 2026-07-23 audit xulosasi", "parent_id": "mem_123"},
        {"id": "doc2", "title": "Eskiz VPS", "parent_id": "doc2"},
        {"id": "doc3", "title": "Ingliz so'zlari", "parent_id": "doc3"},
        {"id": "doc4", "title": "Vertex va Supabase", "parent_id": "doc4"},
        {"id": "doc5", "title": "L0A darslari", "parent_id": "doc5"}
    ]
    
    yozuv.upsert(col, [m["id"] for m in metas], docs, metas)

    def do_search(query):
        return search_core.hybrid_search(
            query=query,
            collection=col,
            kw_index=fts,
            top_k=5,
            where=None
        )

    # 1. Sarlavha bo'yicha topilish
    # Sarlavhadagi "DJIM HoReCa 2026-07-23" faqat meta'da mavjud (matnda yo'q)
    res_title = do_search("DJIM HoReCa 2026-07-23 audit xulosasi")
    assert res_title["status"] == "ok"
    assert any("DJIM" in r.get("meta", {}).get("title", "") for r in res_title["results"])

    # 2. Qo'shimchali so'rov (fe'l/egalik)
    # "Eskiz VPS kirish va tiklash yo'riqnomasi" -> text has "Eskiz tizimiga kirish yo'riqnomasi bor"
    res_qo1 = do_search("Eskiz VPS kirish va tiklash yo'riqnomasi")
    assert res_qo1["status"] == "ok"
    assert any("Eskiz" in r["text"] for r in res_qo1["results"])

    res_qo2 = do_search("Supabase cloud ulanganmi?")
    assert res_qo2["status"] == "ok"
    assert any("Supabase" in r["text"] for r in res_qo2["results"])

    # 3. Ingliz so'zlari stem'da buzilmasligi
    # "api", "mini", "ai"
    res_en = do_search("api ai va mini ilova haqida")
    assert res_en["status"] == "ok"
    # Buzilmasligini tekshirish uchun stem funksiyasining o'zini ham test qilishimiz mumkin
    assert search_core._light_stem("api") == "api"
    assert search_core._light_stem("mini") == "mini"
    assert search_core._light_stem("ai") == "ai"

    # 4. Aloqasiz rad etilishi kerak
    res_irr = do_search("DJIM HoReCa uchun banan pishirish retsepti qanday?")
    # 'banan', 'pishirish', 'retsepti' hech qayerda yo'q
    assert res_irr["status"] == "insufficient_evidence"

    # Yana bir test "qilganmiz" so'zi uchun
    res_qil = do_search("L0A darslarini YouTube'ga yuklaganmiz")
    assert res_qil["status"] == "ok"
    assert any("L0A" in r["text"] for r in res_qil["results"])
