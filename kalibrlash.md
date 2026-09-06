# Kalibrlash va o'lchov natijalari

## 1. O'xshashlikni to'g'rilash (Cosine sim)

Oldingi kodda qisilgan (xato) `1 - d` ishlatilgan bo'lsa, yangi kodda `1 - d/2` (haqiqiy cosine) hisoblanadi.
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
