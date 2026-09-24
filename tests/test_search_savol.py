import pytest
from unittest.mock import patch

from test_yozuv import temp_yozuv
import search_core
import keyword_index
import yozuv

def test_search_savol(temp_yozuv):
    col, db_dir = temp_yozuv
    fts = keyword_index.get_index()

    # Korpus hujjatlari
    docs = [
        "Shadcn bu UI komponentlari kutubxonasi. U React va Tailwind CSS yordamida ishlaydi. Veb ilovalar yaratishda qulay.",
        "Antigravity is an agentic AI coding framework created by Google Deepmind in 2024. It can automate software tasks.",
        "Bananni tozalash va pishirish uchun maxsus retseptlar mavjud.",
        "Kvant fizikasi 1987 yilda yangi nazariyaga ega bo'ldi."
    ]
    metas = [
        {"id": "doc1", "title": "Shadcn haqida"},
        {"id": "doc2", "title": "Antigravity guide"},
        {"id": "doc3", "title": "Banan tayyorlash"},
        {"id": "doc4", "title": "Fizika"}
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

    # 1. O'zbek savol topilishi kerak (funksional so'zlar: nima, qiladi, qanday, ishlatiladi)
    res_uz = do_search("Shadcn nima qiladi va qanday ishlatiladi?")
    assert res_uz["status"] == "ok"
    assert any("Shadcn" in r["text"] for r in res_uz["results"])

    # 2. Ingliz savol topilishi kerak (funksional so'zlar: what, does, and, how, is, used)
    res_en = do_search("What does Antigravity do and how is it used?")
    assert res_en["status"] == "ok"
    assert any("Antigravity" in r["text"] for r in res_en["results"])

    # 3. Qo'shimchali so'z topiladi ("ilovalar" docda, "ilovalarni" so'rovda)
    res_stem = do_search("Veb ilovalarni yaratish")
    assert res_stem["status"] == "ok"
    assert any("veb ilovalar yaratish" in r["text"].lower() for r in res_stem["results"])

    # 4. Aloqasiz so'rov rad etiladi
    res_irr = do_search("Shadcn yordamida banan pishirish retsepti qanday?")
    # "banan pishirish retsepti" Shadcn docda yo'q, Shadcn banan docda yo'q
    # Bular insufficient_evidence bo'lishi kerak yoki topolmasligi kerak
    # Ammo 2 ta meaningful word match qilmasligi kerak bitta hujjatda
    assert res_irr["status"] == "insufficient_evidence"

    # 5. Bitta so'zli holat o'zgarmaydi
    res_single = do_search("Antigravity")
    assert res_single["status"] == "ok"
    assert any("Antigravity" in r["text"] for r in res_single["results"])
    
    # 6. Yil cheklovli holat
    res_year = do_search("Kvant fizikasi 1987")
    assert res_year["status"] == "ok"
    assert any("Kvant" in r["text"] for r in res_year["results"])

    res_year_bad = do_search("Kvant fizikasi 2024")
    assert res_year_bad["status"] == "insufficient_evidence"
