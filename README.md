# Graphify — Claude Code uchun doimiy xotira 🧠

Claude Code'da har yangi chat ochganingizda loyihani noldan tushuntirasizmi? Graphify buni hal qiladi: barcha qarorlar, xulosalar va muhim ma'lumotlar **kompyuteringizdagi lokal bazada** saqlanadi, Claude har yangi chatda undan faqat keraklisini oladi. Natija: kam token sarfi, kam takrorlash, Claude eski qarorlaringizni "eslaydi".

Hammasi lokal ishlaydi — indekslash va qidirish **0 token**, hech qanday API kalit yoki ro'yxatdan o'tish kerak emas. Ma'lumotlaringiz hech qayerga jo'natilmaydi.

> **Eslatma:** bu loyiha GitHub'dagi mashhur `graphify` (kod-bazani bilim grafiga aylantiruvchi CLI, `Graphify-Labs/graphify`) bilan **hech qanday aloqasi yo'q** — nom bir xil bo'lsa-da, vazifasi butunlay boshqa: bu — Claude Code uchun shaxsiy xotira MCP serveri.

## Xususiyatlar

- **`search_knowledge` / `save_memory`** — Claude o'zi vazifa boshida qidiradi, oxirida saqlaydi (qo'lda ham chaqirsa bo'ladi)
- **Loyiha bo'yicha izolyatsiya** — bir loyihaning xotiralari boshqasiga aralashmaydi (xohlasangiz hammasidan qidirsa bo'ladi)
- **Vizual dashboard** (`/graphify` yoki `start_dashboard.bat`) — uchta ko'rinish:
  - **Ro'yxat** — barcha xotiralarni ko'rish, qidirish, filtrlash, o'chirish
  - **Graf** — xotiralar orasidagi semantik o'xshashlik force-directed graf sifatida, loyiha bo'yicha rangli halqa + checkbox filter bilan
  - **Modellar** — embedding modelini tanlash (lokal/OpenAI/Ollama/maxsus API)
- **Token tejash statistikasi** — `search_knowledge` qancha token tejaganining taxminiy hisobi (dashboard'da va `/graphify token` bilan)

## Talablar

- Windows 10/11
- [Python 3.10+](https://www.python.org/downloads/) — o'rnatishda **"Add python.exe to PATH"** katagini albatta belgilang
- Claude Code (desktop yoki CLI)

## O'rnatish (2 daqiqa)

1. Bu repo'ni klonlang yoki ZIP qilib yuklab oling, doimiy joyga qo'ying (masalan `C:\graphify` — keyin joyini o'zgartirmang!)
2. `install.bat` ustiga ikki marta bosing
3. Tugagach Claude Code'ni yopib, qayta oching

Tamom. Tekshirish uchun chatda `/mcp` yozing — ro'yxatda `graphify` ko'rinishi kerak.

## Qanday ishlaydi

Hech narsa qilishingiz shart emas — Claude o'zi:
- vazifa boshida bazadan eski qarorlarni qidiradi (`search_knowledge`)
- sessiya oxirida xulosani saqlaydi (`save_memory`)

Qo'lda ham buyura olasiz:
- *"graphify'dan qidir: to'lov moduli qanday qilingan edi"*
- *"bu qarorni graphify'ga saqla"*
- *"graphify'dagi eslatmalarni ko'rsat"*

`/graphify` slash-komandasi ham o'rnatiladi:

| Komanda | Natija |
|---|---|
| `/graphify` | Vizual dashboardni ochadi (http://localhost:5000) |
| `/graphify list` | Saqlangan xotiralar ro'yxati |
| `/graphify save <matn>` | Matnni xotiraga saqlaydi |
| `/graphify token` | Token tejash statistikasi (taxminiy) |
| `/graphify <so'z>` | Semantik qidiruv |

Xotiralar loyiha bo'yicha avtomatik ajratiladi — bir loyihaning qarorlari boshqasini chalg'itmaydi, lekin kerak bo'lsa hammasidan qidirsa bo'ladi.

## Hujjatlarni qo'shish (ixtiyoriy)

.md/.txt hujjatlaringizni bazaga indekslash:

```
.venv\Scripts\python.exe indexer.py "C:\yo'l\hujjatlar-papkasi" --project LoyihaNomi
```

## Eski loyihani tanishtirish

O'sha loyihada chat ochib, bir marta ayting:
> "Loyihani o'rganib chiqib, asosiy arxitektura, texnologiyalar va muhim qarorlarni graphify'ga saqla"

Keyingi barcha chatlar shu tayyor xotiradan boshlanadi.

## Embedding modelini almashtirish

Standart v1 (jonli) holatda — `all-MiniLM-L6-v2` (384-d, lokal, bepul, internet/kalit shart emas). v2 branchida esa Ollama `bge-m3` (1024-d) modeliga o'tish tayyorlangan. Dashboard'ning "Modellar" bo'limida OpenAI/Ollama/maxsus API'ga o'tsa bo'ladi — lekin bazada allaqachon ma'lumot bo'lsa, bu **buzuvchi o'zgarish**: eski vektorlar yangi model bilan mos kelmaydi. Amalda almashtirish uchun `kg_db/` ni bo'shatib, `indexer.py` bilan qayta indekslash kerak. Batafsil: [`CUSTOM_MODELS.md`](CUSTOM_MODELS.md) va [`migrate_v2/README.md`](migrate_v2/README.md).

## Tizim arxitekturasi: v1 va v2

- **Hozirgi jonli holat**: v1 (jonli) — `all-MiniLM-L6-v2` (384-d, gate 0.8) modelida barqaror ishlab turibdi — v2 branchda, switch kutadi.
- **v2 yangilanishi (`feat/graphify-v2` branchda)**:
  - **Embedding modeli**: Ollama `bge-m3` (1024-d, ko'p tilli — o'zbek va ingliz tillarida yuqori sifat).
  - **Model identity tekshiruvi**: Baza metama'lumotlarida model nomi va o'lchami (`bge-m3`, 1024-d) qat'iy tekshiriladi; agar Ollama yoki model mavjud bo'lmasa, qidiruv to'xtaydi (fail-closed, hech qachon noto'g'ri modelga o'tib ketmaydi).
  - **Leksik gate (0.5)**: So'rov so'zlarining kamida 50% qamrovi (`MIN_COVERAGE = 0.5`) talab qilinadi (eski 0.8 dan tushirilgan, bu semantik moslikni to'sib qo'ymaydi).
  - **Cross-encoder reranker**: Ixtiyoriy ONNX reranker qo'llab-quvvatlanadi, biroq resurs tejash va past kechikish uchun sukutda o'chiq (`GRAPHIFY_RERANK=0`).
  - To'liq o'tish va sinxronizatsiya yo'riqnomasi: [`migrate_v2/README.md`](migrate_v2/README.md).


## Maslahatlar

- **Sifat > miqdor**: hamma narsani emas, qaror va xulosalarni saqlang
- Eskirgan xotiralarni o'chirtirib boring: *"mem_xxx ni o'chir"*
- Butun xotira `kg_db/` papkasida — backup qilsangiz, hech narsa yo'qolmaydi

## O'chirish

`uninstall.bat` ni ishga tushiring — Claude Code'dan uziladi. Keyin papkani qo'lda o'chirsangiz bo'ladi.

## Muammolar

| Muammo | Yechim |
|---|---|
| "Python topilmadi" | Python'ni PATH bilan qayta o'rnating |
| `/mcp` da graphify yo'q | Claude Code'ni to'liq yopib qayta oching |
| Birinchi qidiruv sekin | Normal — model birinchi marta yuklanadi |
| Papkani ko'chirdim, ishlamayapti | `install.bat` ni yangi joyda qayta ishga tushiring |

## Litsenziya

MIT — [LICENSE](LICENSE) fayliga qarang.
