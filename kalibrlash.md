# Kalibrlash va o'lchov natijalari

## 1. O'xshashlikni to'g'rilash (Cosine sim)

O'xshashlik hisobi vektor fazosiga bog'liq (`similarity.py`):
- Cosine yoki IP fazosida Chroma masofasi $d \in [0, 2]$ bo'lib, o'xshashlik `1.0 - d` formulasi orqali hisoblanadi.
- L2 fazosida esa Chroma kvadratik Evklid masofasini qaytaradi. Vektorlar birlik uzunlikda (`||v|| = 1`, unit length) normallangan bo'lsa, $d = 2(1 - \cos\theta)$ munosabatidan kelib chiqib, haqiqiy kosinus o'xshashligi `1.0 - d / 2.0` bo'ladi.
- Agar vektorlar birlik uzunlikda bo'lmasa, `1 - d/2` noto'g'ri natija beradi; bu holda to'g'ridan-to'g'ri normallangan dot product `(u · v) / (||u|| * ||v||)` hisoblanadi. Barcha holatlarda natija `[0.0, 1.0]` oralig'iga qisiladi (clamped).
O'lchov natijalari quyidagicha:

| So'rov | Nechta natija o'tdi | Ballar |
|---|---|---|
| `Dify` | **atigi 1 ta** | 0.307 |
| `customer retention` | 5 ta | 0.52 / 0.50 / 0.46 / ... |
| `kvant fizikasi bo'yicha nobel mukofoti 1987` (ALOQASIZ) | **5 ta** | 0.49 / 0.47 / ... / 0.43 |

**Chegara (MIN_SIMILARITY) tanlovi:** `0.10`.
**Asos:** Aloqasiz so'rov haqiqiy mos keladigan yozuvdan YUQORI ball oladi. Ya'ni bu model uchun cosine kattaligi mavzuviy moslikni ajratmaydi. Demak, "Dify" ni o'tkazadigan har qanday chegara shovqinni ham o'tkazadi va shovqinni to'sadigan har qanday chegara "Dify" ni ham to'sadi. Shu sababli absolyut chegara ishlamaydi. MIN_SIMILARITY faqatgina axlat filtri (butunlay bog'liqsiz vektorlarni kesish uchun) vazifasini bajaradi va 0.10 qilib belgilandi. Haqiqiy ajratish gibrid qidiruv (kalit so'z + RRF + manba bo'yicha guruhlash) orqali amalga oshirilgan (`search_core.py`, `hybrid_search`, 2026-09-06).

## 2. Qidiruv tezligi o'lchovi (`_record_search_savings` optimizatsiyasi)

`Promtlarim` loyihasi miqyosida (taxminan 4.5M belgi) to'liq o'qish va yangi optimizatsiyalangan qidiruv tezligi solishtirildi.

- **Eski holat** (har qidiruvda butun scope o'qilganda va orqa fon oqimi ishlatilganda):
  - O'rtacha: ~0.2807s

- **Yangi holat** (daemon thread olib tashlangach, kesh promahida kuttirilmaganda):
  - O'rtacha: ~0.1717s

Natija: qidiruv tezligi optimallashdi, poyga holatlari (race conditions) va memory leaks olib tashlandi, thread to'liq yo'q qilindi. Kesh `warm_savings_cache` orqali tashqaridan to'ldirilishi ko'zda tutildi.

## 3. Leksik gate (`MIN_COVERAGE`) va Reranker kalibrlashi (2026-10-29)

### A. Leksik Gate (`MIN_COVERAGE` 0.8 → 0.5)

To'liq 200 ta so'rovlik benchmark (`migrate_v2/out-v2b/EVAL.md`):

| Versiya (Model) | Gate (`MIN_COVERAGE`) | O'rtacha Recall@10 | O'rtacha MRR@10 |
|---|:---:|:---:|:---:|
| **v1** (MiniLM) | `0.8` | 0.45 | 0.43 |
| **v1** (MiniLM) | `0.5` | 0.77 | 0.68 |
| **v1** (MiniLM) | `0.0` | 0.86 | 0.65 |
| **v2** (bge-m3) | `0.8` | 0.45 | 0.44 |
| **v2** (bge-m3) | `0.5` | 0.78 | 0.73 |
| **v2** (bge-m3) | `0.0` | 0.92 | 0.81 |

Precision-set sinovi natijalari (2026-10-29):
- **Gate 0.8:** R@10 = 0.16, FAR = 0.01. Juda qat'iy: so'rov so'zlarining 80% qamrovi talab qilinishi sababli ko'plab to'g'ri semantik natijalarni to'sib qo'yadi.
- **Gate 0.5:** R@10 = 0.50, FAR = 0.24 (**tanlangan muvozanat**). Semantik qidiruv natijalarini yo'qotmaydi, shu bilan birga noto'g'ri javoblar (FAR) darajasini maqbul oraliqda ushlab turadi.
- **Gate 0.0:** R@10 = 0.74, FAR = 1.00. Filtrsiz: barcha shovqin natijalarga o'tib ketadi.

**Qaror:** `search_core.py:18` da `MIN_COVERAGE = 0.5` qilib belgilandi (oldingi 0.8 dan tushirildi).

### B. Cross-Encoder Reranker O'lchovlari va Qarori

`bge-reranker-v2-m3-ONNX` modeli bo'yicha 2026-10-29 sanasidagi o'lchov natijalari:
- **GPU fp16** (20×512 token): 0.23 s.
- **CPU int8**: 2.3–3.2 s (ba'zi og'ir so'rovlarda CPU'da p50 19 s ga yetgan, 20 s byudjetda ulgurmagan).
- **Kechikish (v2 L2 yo'li):** ⚠️ o'lchanmagan.
- **Sifat ta'siri:** Gate 0.5 darajasida reranker bergan qo'shimcha foyda atigi **+0.01 MRR** (v2@0.5 da 0.73 dan 0.74 ga).

**Qaror:** CPU'dagi yuqori kechikish (2.3–3.2 s) va minimal sifat o'sishi (+0.01 MRR) tufayli, reranker sukut bo'yicha o'chiq (`GRAPHIFY_RERANK=0`) qilib qoldirildi (`search_core.py:408`). Faqat GPU mavjud bo'lgan ishlab chiqarish muhitida yoqish tavsiya etiladi.
