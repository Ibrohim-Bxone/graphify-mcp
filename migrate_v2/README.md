# Graphify v2 Migratsiya va Baholash Qo'llanmasi

Ushbu paket Graphify ma'lumotlar bazasini `v1` (Lokal MiniLM-L6-v2, 384 o'lcham) holatidan `v2` (Ollama `bge-m3`, 1024 o'lcham) holatiga yonma-yon (side-by-side) xavfsiz o'tkazish, sinxronizatsiya qilish va qidiruv sifatini baholash uchun mo'ljallangan.

---

## 🗄️ 1. Bazalar Holati va Operatsion Bog'liqlik

### Bazalar Holati
- **`../Graphify/kg_db`** — Jonli `v1` bazasi (`all-MiniLM-L6-v2`, 384 o'lcham, gate 0.8). Hozirgi kunda ishlab turgan yagona jonli tizim bazasi.
- **`../Graphify/kg_db_v2`** — 28-oktabrdagi CHALA build (manifest yo'q). Ushbu katalog eskirgan deb hisoblanadi va foydalanilmaydi.
- **`../Graphify/kg_db_v2b`** — 29-oktabrdagi to'liq va tasdiqlangan `v2` bazasi (31 099 ta yozuv, `graphify_manifest.json`: distance `l2`, normalization `false`, `build_complete.json`). Hozirgi barcha sinov, baholash va o'tishlar uchun asosiy to'liq baza.

### Operatsion Bog'liqlik (Fail-Closed)
- `v2` arxitekturasida ma'lumotlar bazasi ochilganda model identifikatsiyasi (`graphify_manifest.json` va kolleksiya metama'lumotlari) qat'iy tekshiriladi (`db.py:_open_collection`).
- Tizim to'g'ri ishlashi uchun **Ollama** xizmati va **`bge-m3`** modeli doimiy ishlab turishi shart.
- Agar Ollama serveri ishlamayotgan bo'lsa yoki `bge-m3` modeli topilmasa, tizim `ModelIdentityError` xatosini ko'taradi va **fail-closed** rejimida to'xtaydi (xatolik yashirilmaydi va hech qachon noto'g'ri modelga yoki MiniLM ga qaytib ketmaydi).

### Kuzatilgan Drift va Majburiy Sinxronizatsiya
- Hozirgi `kg_db_v2b` bazasi v1 ning vaqtinchalik nusxasi (`kg_db_v1snap`) asosida qurilgan.
- Jonli `kg_db` bazasida yangi ma'lumotlar yozilishi davom etganligi sababli, jonli baza bilan `kg_db_v2b` o'rtasida ma'lumotlar farqi (drift) mavjud.
- Shu sababli, **jonli o'tish (switch) oldidan `--sync` buyrug'ini ishga tushirish MAJBURIYDIR!**

---

## 🚀 2. Buyruqlar Qo'llanmasi

Barcha buyruqlar virtual muhit orqali loyiha ildizida bajariladi.

### A. Qisqa sinov (Smoke Run)

1. **v2 Smoke bazasini qurish (dastlabki 300 ta yozuv):**
   ```bash
   .venv/Scripts/python.exe migrate_v2/build_v2.py --smoke
   ```
   *Natija:* `kg_db_v2_smoke/` katalogida 300 ta yozuv uchun 1024-o'lchamli bge-m3 embeddinglar, `manifest.jsonl`, `graphify_manifest.json`, `build_complete.json` va `keyword_index.sqlite3` yaratiladi. Chiqish yo'li v1 manbasi bilan bir xil bo'lishi qat'iyan man etiladi.

2. **Smoke qidiruv baholashini o'tkazish (5 ta inline so'rov, < 5 daqiqa):**
   ```bash
   .venv/Scripts/python.exe migrate_v2/eval_v2.py --smoke
   ```
   *Natija:* Hech qachon jonli fayllar ochilmaydi — alohida vaqtinchalik katalogda `kg_db_v1_replica` va `kg_db_v2_replica` yaratilib, xolis baholanadi. `migrate_v2/out/eval.json` (har bir so'rov bo'yicha ranklar, model identifikatsiyasi, corpus xeshi) va `migrate_v2/out/EVAL.md` shakllantiriladi. Vektor xatoliklari yutilmaydi, darhol to'xtatiladi.

---

### B. To'liq Baza Qurish (Full Build)

To'liq 31,000+ yozuvli v1 bazasini `kg_db_v2b` ga ko'chirish (hozirgi to'liq build: `kg_db_v2b`):

```bash
.venv/Scripts/python.exe migrate_v2/build_v2.py --v1-db kg_db --out kg_db_v2b --batch-size 50 --max-workers 8
```

- v1 bazasi bitta snapshot tranzaksiyasida (`BEGIN`) faqat SQLite orqali o'qish rejimida (`mode=ro`) ochiladi, v1 fayllariga yozilmaydi.
- Hujjat matni bo'lmagan (documentless) yozuvlar ham to'liq saqlanadi (`LEFT JOIN`).
- Kolleksiyaning to'liq konfiguratsiyasi va metama'lumotlari nusxalanadi hamda model identifikatsiyasi (`schema_version`: 2, `model`: `bge-m3`, `dimension`: 1024, `model_digest`) bog'lanadi.
- Jarayon to'xtab qolsa ham davom ettirish mumkin (resumable: mavjud xeshlar solishtirilib, o'zgarganlar qayta ishlanadi).
- Kalit so'z indeksi muvaffaqiyatli qayta qurilgach va barcha yozuvlar tasdiqlangach, `build_complete.json` atomik tarzda yoziladi.

---

### C. Sinxronizatsiya (Incremental Sync)

v1 bazasiga yangi yozuvlar qo'shilganda yoki mavjudlari o'zgarganda v2 ni manifest bo'yicha yangilash:

```bash
.venv/Scripts/python.exe migrate_v2/build_v2.py --sync --v1-db kg_db --out kg_db_v2b
```

- `manifest.jsonl` dagi SHA-256 xeshlar orqali yangi, o'zgargan va o'chirilgan yozuvlarni aniqlaydi.
- Faqat o'zgargan/yangi yozuvlar qayta embed qilinadi, o'chirilganlari v2 dan chiqariladi.
- Kalit so'z indeksi (`keyword_index.sqlite3`) avtomatik ravishda yangilanadi va to'liq tekshiriladi.
- Sinxronizatsiya tugagach `build_complete.json` yangilanadi.

---

### D. To'liq Baholash (Full Evaluation)

Tashqi benchmark so'rovlar fayli orqali to'liq baholash:

```bash
.venv/Scripts/python.exe migrate_v2/eval_v2.py --v1-db kg_db --v2-db kg_db_v2b --queries "<repo>/migrate_v2/eval_queries.jsonl"
```

- Jonli fayllar o'rniga vaqtinchalik replikalar (`kg_db_v1_replica` va `kg_db_v2_replica`) ishlatiladi.
- 4 ta so'rov toifasi (`uz_need`, `en_need`, `uz_name`, `en_name`) bo'yicha Recall@10 va MRR@10 hisoblanadi.
- Leksik gate (`MIN_COVERAGE`: `0.8`, `0.5`, `0.0`) darajalarida taqqoslanadi.

---

## ⚠️ 3. Jonli O'tish (Switch) Tartibi

> [!IMPORTANT]
> O'tish faqat to'liq sinov, tekshiruv va barcha tayyorgarlik choralari ko'rilgandan so'ng amalga oshiriladi (`migrate_v2/switch_check.py` qadamlariga muvofiq).

### 1-qadam: Barcha yozuvchi xizmatlarni to'xtatish
Barcha yozuvchi jarayonlar va sessiyalarni to'liq to'xtating:
- Claude Code MCP serverlari
- OpenAI Codex CLI MCP sessiyalari
- Graphify HTTP server (`server_http.py`)
- Telegram bot xizmatlari (`bot/` yoki `kanal-konveyer/`)

### 2-qadam: Yakuniy muzlatish va sinxronizatsiya
Yozuvchilar to'xtatilgan holatda jonli `kg_db` dagi so'nggi o'zgarishlarni v2 ga o'tkazing:
```bash
.venv/Scripts/python.exe migrate_v2/build_v2.py --sync --v1-db kg_db --out kg_db_v2b
```

### 3-qadam: Baza butunligini to'liq tekshirish (Read-only)
Switch oldi tekshiruv skriptini ishga tushiring:
```bash
.venv/Scripts/python.exe migrate_v2/switch_check.py --v1-db kg_db --v2-db kg_db_v2b
```
Checklist quyidagilarni tekshiradi va barchasi `PASS` bo'lishi shart:
- Barcha Graphify yozuvchi jarayonlari to'xtatilganligi (faol jarayonlar 0 bo'lishi kerak).
- v1 va v2 o'rtasida ma'lumotlar tafovuti (drift) 0 ekanligi.
- v2 model identifikatsiyasi (`bge-m3`) va 1-so'rovlik self-test o'tganligi.
- Baza to'liqligi (100% yozuvlar va kalit so'z indeksi tasdig'i).
- Barcha sozlama fayllari mavjudligi va holati.

### 4-qadam: Barcha sozlama fayllarida `GRAPHIFY_DB_PATH` ni yangilash
Barcha mijozlar va xizmatlar sozlamalarida yo'lni `kg_db_v2b` ga yo'naltiring:
- **`~/.claude.json`**:
  `"env": { "GRAPHIFY_DB_PATH": "<repo>/kg_db_v2b" }`
- **`~/.codex/config.toml`**:
  `env = { GRAPHIFY_DB_PATH = "<repo>/kg_db_v2b" }`
- **Bot va konveyer `.env` fayllari (`bot/.env`, `kanal-konveyer/.env`):**
  `GRAPHIFY_DB_PATH=<repo>/kg_db_v2b`
*(Izoh: v2 bazasida model identifikatsiyasi `graphify_manifest.json` va kolleksiya metama'lumotlariga biriktirilganligi sababli, `models_config.json` yaratish shart emas — tizim avtomatik ravishda `bge-m3` ga bog'lanadi va model mos kelmasa fail-closed to'xtaydi).*

### 5-qadam: Xizmatlarni qayta ishga tushirish
MCP serverlar, HTTP server va bot xizmatlarini qayta ishga tushiring.

### 6-qadam: Jonli model identifikatsiyasi va test qidiruv so'rovi
Jonli tizimda 1 ta test qidiruv so'rovi bilan v2 model identifikatsiyasini tekshiring:
```bash
.venv/Scripts/python.exe -c "import db; col = db.get_collection(); print('Kolleksiya:', col.name, 'Yozuvlar soni:', col.count())"
```

### 7-qadam: Yozish rejimini qayta ochish
Tizim to'liq v2 rejimida ishlashga tayyor.

---

## 🔄 4. Orqaga Qaytish va Ma'lumotlarni Yarashtirish (Rollback & Reconciliation)

Agar v2 bilan kutilmagan muammo yuzaga kelsa, `switch_check.py` spetsifikatsiyasi bo'yicha v1 ga xavfsiz qaytish tartibi:

### 1-qadam: v2 dagi yozuvchi jarayonlarni zudlik bilan to'xtatish
Barcha mijozlarni (MCP, HTTP, Bot) zudlik bilan to'xtating.

### 2-qadam: v2 da yaratilgan yangi ma'lumotlarni v1 ga yarashtirish (Reconciliation)
Switch vaqtidagi v2 snapshot manifestiga (`manifest.jsonl`) nisbatan v2 faol bo'lgan davrda amalga oshirilgan barcha o'zgarishlarni aniqlash va v1 bilan to'liq yarashtirish:
1. **Qo'shilgan yangi IDlar (Added):** Switch snapshot manifestida mavjud bo'lmagan, ammo v2 da yangi yaratilgan barcha IDlarni aniqlash va ularning matni hamda metama'lumotlarini v1 bazasiga kiritish (`v1_col.upsert` yoki `server.save_memory()`).
2. **O'zgartirilgan IDlar (Modified):** Switch snapshot manifestidagi `doc_sha256` yoki `meta_sha256` xeshlari joriy v2 dagi xeshlar bilan mos kelmagan barcha yozuvlarni aniqlash va v1 dagi tegishli yozuvlarni yangi hujjat matni va metama'lumotlar bilan yangilash.
3. **O'chirilgan IDlar (Deleted):** Switch snapshot manifestida mavjud bo'lgan, ammo v2 jonli ishlashi davomida o'chirilgan barcha IDlarni aniqlash va ularni v1 bazasidan ham o'chirish (`v1_col.delete(ids=...)`).
4. **v1 kalit so'z indeksini qayta qurish:** Barcha o'zgarishlar kiritilgach, v1 kalit so'z indeksini to'liq qayta qurish:
   ```bash
   .venv/Scripts/python.exe -c "import keyword_index; keyword_index.rebuild()"
   ```

### 3-qadam: Sozlama fayllarida `GRAPHIFY_DB_PATH` ni v1 ga qaytarish
- `~/.claude.json`, `~/.codex/config.toml` va `.env` fayllarida `GRAPHIFY_DB_PATH` qiymatini yana `kg_db` ga yo'naltiring:
  `GRAPHIFY_DB_PATH=<repo>/kg_db`

### 4-qadam: models_config.json ni olib tashlash yoki lokal rejimga qaytarish
Mavjud bo'lsa, `models_config.json` ni olib tashlang yoki lokal MiniLM rejimiga qaytaring.

### 5-qadam: Xizmatlarni qayta ishga tushirish
Xizmatlarni qayta ishga tushiring. Graphify avtomatik ravishda v1 bazasini va lokal MiniLM embedderini yuklaydi.

### 6-qadam: v1 ishini tekshirib, yozishni qayta ochish
Jonli tizimda qidiruv va yozuvlar mavjudligini tasdiqlab, xizmatlarni odatiy foydalanishga topshiring.

---

## ⚙️ 5. Muhit O'zgaruvchilari (Environment Variables)

Graphify kod-bazasida mavjud barcha 22 ta `GRAPHIFY_*` muhit o'zgaruvchilari:

| O'zgaruvchi | Sukut qiymati | Manba moduli | Tavsifi / Vazifasi |
|---|:---:|---|---|
| `GRAPHIFY_ALWAYS_OPEN` | `shared,Promtlarim` | `server.py` | Har doim ochiq bo'lgan (barcha loyihalardan birga qidiriladigan) umumiy loyihalar ro'yxati (vergul bilan ajratilgan). |
| `GRAPHIFY_AUTHOR` | `—` | `server.py` | Yangi xotira saqlanganda muallif yoki hissa qo'shuvchi identifikatori (agar so'rovda ko'rsatilmagan bo'lsa). |
| `GRAPHIFY_CHROMA_HOST` | `127.0.0.1` | `db.py` | Chroma HTTP serverining IP manzili (`GRAPHIFY_STORAGE_MODE=http` bo'lganda, faqat loopback IP). |
| `GRAPHIFY_CHROMA_PORT` | `8993` | `db.py` | Chroma HTTP serverining porti (`GRAPHIFY_STORAGE_MODE=http` bo'lganda, 1–65535 oralig'ida). |
| `GRAPHIFY_DB_PATH` | `<repo>/kg_db` | `db.py`, `yozuv.py`, `keyword_index.py` | ChromaDB va SQLite indekslari saqlanadigan ma'lumotlar bazasi katalogining mutlaq yoki nisbiy yo'li. |
| `GRAPHIFY_DENSE_WEIGHT` | `1.0` | `search_core.py` | Gibrid qidiruvda (RRF) zich vektorli semantik natijalar uchun vazn koeffitsienti. |
| `GRAPHIFY_DIR` | `<repo>/Graphify` | `indeks-qayta-qur.py`, `indeks_tekshir.py`, `indeks-smoke.py` | Audit va indeksni qayta qurish yordamchi skriptlari uchun Graphify loyihasi ildiz katalogi yo'li. |
| `GRAPHIFY_EVAL_WORKER_TIMEOUT` | `5400.0` | `migrate_v2/eval_v2.py` | `eval_v2.py` da baholash worker jarayonining maksimal ishlash vaqti (taymout, sekundlarda; 90 daqiqa). |
| `GRAPHIFY_HTTP_TOKEN` | `—` | `server_http.py` | Graphify HTTP serveriga kirish uchun bearer avtorizatsiya tokeni (agar o'rnatilmagan bo'lsa, avtorizatsiya talab qilinmaydi). |
| `GRAPHIFY_KEYWORD_REBUILD_TIMEOUT` | `1800.0` | `migrate_v2/build_v2.py` | `build_v2.py` da `keyword_index.rebuild()` subprocess jarayoni taymout vaqti (sekundlarda; 30 daqiqa). |
| `GRAPHIFY_LEXICAL_WEIGHT` | `0.2` | `search_core.py` | Gibrid qidiruvda (RRF) leksik/kalit so'z (BM25/FTS) natijalari uchun vazn koeffitsienti. |
| `GRAPHIFY_MIN_COVERAGE` | — | `migrate_v2/precision_eval.py` (faqat o'rnatadi) | ⚠️ Hech bir modul O'QIMAYDI. Haqiqiy gate `search_core.py:18` da qattiq `MIN_COVERAGE = 0.5`; uni o'zgartirish uchun kodni tahrirlash kerak. `precision_eval.py` gate'ni to'g'ridan-to'g'ri `search_core.MIN_COVERAGE` ga yozadi. |
| `GRAPHIFY_MIN_SIMILARITY` | `0.10` | `server.py` | Vektorli qidiruv natijalarini birlamchi saralash uchun minimal o'xshashlik chegarasi (axlat/shovqin filtri). |
| `GRAPHIFY_RERANK` | `0` | `search_core.py` | Qidiruvdan so'ng cross-encoder rerankerni yoqish (`1` / `true`) yoki o'chirish (`0` / `false`), sukut bo'yicha o'chiq. |
| `GRAPHIFY_RERANKER_MODEL` | `—` | `reranker.py` | Maxsus ONNX reranker model fayli yo'li (agar ko'rsatilmasa, sukut bo'yicha `onnx/model_int8.onnx` yoki `onnx/model_fp16.onnx` ishlatiladi). |
| `GRAPHIFY_RERANK_ALPHA` | `0.5` | `search_core.py` | Reranklangan ball va RRF ballarini birlashtirishdagi aralashtirish koeffitsienti (alpha). |
| `GRAPHIFY_RERANK_BUDGET_S` | `3.0` | `reranker.py` | Reranker xulosa (inference) hisoblash jarayoniga ajratilgan maksimal vaqt byudjeti (sekundlarda). |
| `GRAPHIFY_RERANK_DEVICE` | `auto` | `reranker.py` | Reranker hisoblash qurilmasi (`auto`, `cuda`, `cpu`). |
| `GRAPHIFY_RERANK_THREADS` | `1` | `reranker.py` | ONNX Runtime CPU execution provider uchun intra-op oqimlar soni. |
| `GRAPHIFY_RERANK_TOP_N` | `20` | `search_core.py` | RRF natijalaridan rerankerga uzatiladigan nomzodlar soni (top-N). |
| `GRAPHIFY_RRF_K` | `60` | `search_core.py` | Reciprocal Rank Fusion (RRF) algoritmi formulasidagi $k$ doimiysi (`score = 1 / (k + rank)`). |
| `GRAPHIFY_STORAGE_MODE` | `local` | `db.py` | ChromaDB saqlash va ulanish rejimi (`local` yoki `http`). |
