# Graphify v2 Migratsiya va Baholash Qo'llanmasi

Ushbu paket Graphify ma'lumotlar bazasini `v1` (Lokal MiniLM-L6-v2, 384 o'lcham) holatidan `v2` (Ollama `bge-m3`, 1024 o'lcham) holatiga yonma-yon (side-by-side) xavfsiz o'tkazish, sinxronizatsiya qilish va qidiruv sifatini baholash uchun mo'ljallangan.

---

## 🚀 1. Buyruqlar Qo'llanmasi

Barcha buyruqlar virtual muhit orqali loyiha ildizida (`Graphify/`) bajariladi.

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

To'liq 31,000+ yozuvli v1 bazasini `kg_db_v2/` ga ko'chirish:

```bash
.venv/Scripts/python.exe migrate_v2/build_v2.py --v1-db kg_db --out kg_db_v2 --batch-size 50 --max-workers 8
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
.venv/Scripts/python.exe migrate_v2/build_v2.py --sync --v1-db kg_db --out kg_db_v2
```

- `manifest.jsonl` dagi SHA-256 xeshlar orqali yangi, o'zgargan va o'chirilgan yozuvlarni aniqlaydi.
- Faqat o'zgargan/yangi yozuvlar qayta embed qilinadi, o'chirilganlari v2 dan chiqariladi.
- Kalit so'z indeksi (`keyword_index.sqlite3`) avtomatik ravishda yangilanadi va to'liq tekshiriladi.
- Sinxronizatsiya tugagach `build_complete.json` yangilanadi.

---

### D. To'liq Baholash (Full Evaluation)

Tashqi benchmark so'rovlar fayli (`queries.jsonl`) orqali to'liq baholash:

```bash
.venv/Scripts/python.exe migrate_v2/eval_v2.py --v1-db kg_db --v2-db kg_db_v2 --queries "D:/claude projects/.agy/S-14ecd8/til-test/data/queries.jsonl"
```

- Jonli fayllar o'rniga vaqtinchalik replikalar (`kg_db_v1_replica` va `kg_db_v2_replica`) ishlatiladi.
- 4 ta so'rov toifasi (`uz_need`, `en_need`, `uz_name`, `en_name`) bo'yicha Recall@10 va MRR@10 hisoblanadi.
- Leksik gate (`MIN_COVERAGE`: `0.8`, `0.5`, `0.0`) darajalarida taqqoslanadi.

---

## ⚠️ 2. To'liq O'tish Tartibi (Production Switch Procedure)

> [!IMPORTANT]
> O'tish faqat to'liq sinov, tekshiruv va foydalanuvchi tasdig'idan so'ng amalga oshiriladi.

### 1-qadam: Yozuvchi xizmatlarni to'xtatish
Barcha yozuvchi jarayonlar va MCP sessiyalarini to'xtating:
- Claude Code MCP serverlari
- OpenAI Codex CLI MCP sessiyalari
- Graphify HTTP server (`server_http.py`)
- Telegram bot (`bot/` yoki `kanal-konveyer/`)

### 2-qadam: O'tish oldi tayyorgarligini tekshirish (Read-only)
```bash
.venv/Scripts/python.exe migrate_v2/switch_check.py --v1-db kg_db --v2-db kg_db_v2
```
Checklist quyidagilarni tekshiradi:
- Barcha Graphify yozuvchi jarayonlari to'xtatilganligini (faol jarayonlar 0 bo'lishi kerak).
- v1 va v2 o'rtasida farq (drift) yo'qligini.
- v2 model identifikatsiyasi va 1-so'rovlik self-test o'tganligini.
- Barcha sozlama fayllari holatini.

### 3-qadam: Yakuniy muzlatish va sinxronizatsiya
Yozuvchilar to'xtatilgan holatda oxirgi o'zgarishlarni v2 ga o'tkazish:
```bash
.venv/Scripts/python.exe migrate_v2/build_v2.py --sync --v1-db kg_db --out kg_db_v2
```

### 4-qadam: Barcha sozlama fayllarida `GRAPHIFY_DB_PATH` ni yangilash
Barcha mijozlar uchun yo'lni `kg_db_v2` ga yo'naltiring:
- **`~/.claude.json`**:
  `"env": { "GRAPHIFY_DB_PATH": "D:\\claude projects\\graphify-ekotizim\\Graphify\\kg_db_v2" }`
- **`~/.codex/config.toml`**:
  `env = { GRAPHIFY_DB_PATH = "D:\\claude projects\\graphify-ekotizim\\Graphify\\kg_db_v2" }`
- **Bot `.env` va tizim muhiti:**
  `GRAPHIFY_DB_PATH=D:\claude projects\graphify-ekotizim\Graphify\kg_db_v2`
*(Izoh: v2 bazasida model identifikatsiyasi `graphify_manifest.json` va kolleksiya metama'lumotlariga biriktirilganligi sababli, `models_config.json` yaratish shart emas — tizim avtomatik ravishda `bge-m3` ga bog'lanadi va model mos kelmasa xato berib to'xtaydi).*

### 5-qadam: Xizmatlarni qayta ishga tushirish
MCP serverlar, HTTP server va bot xizmatlarini ishga tushiring.

### 6-qadam: Jonli model identifikatsiyasi tekshiruvi
Birinchi qidiruv so'rovi orqali `bge-m3` modeli to'g'ri ishlayotganligini tekshiring:
```bash
.venv/Scripts/python.exe -c "import db; col = db.get_collection(); print(col.name, col.count())"
```

### 7-qadam: Yozish rejimini qayta ochish
Tizim to'liq v2 rejimida ishlashga tayyor.

---

## 🔄 3. Orqaga Qaytish va Ma'lumotlarni Yarashtirish (Rollback & Reconciliation)

Agar v2 bilan kutilmagan muammo yuzaga kelsa, v1 ga xavfsiz qaytish tartibi:

### 1-qadam: v2 dagi yozuvchilarni to'xtatish
Barcha mijozlarni (MCP, HTTP, Bot) zudlik bilan to'xtating.

### 2-qadam: v2 da yozilgan yangi ma'lumotlarni yarashtirish (Reconciliation)
Switch vaqtidagi v2 snapshot manifestiga (`manifest.jsonl`) nisbatan v2 faol bo'lgan davrda amalga oshirilgan barcha o'zgarishlarni aniqlash va v1 bilan to'liq yarashtirish:
1. **Qo'shilgan yangi IDlar (Added):** Switch snapshot manifestida mavjud bo'lmagan, lekin v2 da yangi yaratilgan barcha IDlarni aniqlash va ularning matni hamda metama'lumotlarini v1 bazasiga kiritish (`v1_col.upsert` yoki `server.save_memory()`).
2. **O'zgartirilgan IDlar (Modified):** Switch snapshot manifestidagi `doc_sha256` yoki `meta_sha256` xeshlari joriy v2 dagi xeshlar bilan mos kelmagan barcha yozuvlarni aniqlash va v1 dagi tegishli yozuvlarni yangi hujjat matni va metama'lumotlar bilan yangilash.
3. **O'chirilgan IDlar (Deleted):** Switch snapshot manifestida mavjud bo'lgan, ammo v2 jonli ishlashi davomida o'chirilgan barcha IDlarni aniqlash va ularni v1 bazasidan ham o'chirish (`v1_col.delete(ids=...)`).
4. **v1 kalit so'z indeksini qayta qurish:** Barcha o'zgarishlar kiritilgach, v1 kalit so'z indeksini to'liq qayta qurish:
  ```bash
  .venv/Scripts/python.exe -c "import keyword_index; keyword_index.rebuild()"
  ```

### 3-qadam: Sozlama fayllarini v1 ga qaytarish
- `~/.claude.json`, `~/.codex/config.toml` va `.env` fayllarida `GRAPHIFY_DB_PATH` qiymatini yana `kg_db` ga yo'naltiring:
  `GRAPHIFY_DB_PATH=D:\claude projects\graphify-ekotizim\Graphify\kg_db`

### 4-qadam: Xizmatlarni qayta ishga tushirish
Xizmatlarni qayta ishga tushiring. Graphify avtomatik ravishda v1 bazasini va lokal MiniLM embedderini yuklaydi.

### 5-qadam: v1 yaxlitligini tekshirish
Jonli tizimda qidiruv va yozuvlar mavjudligini tasdiqlang.
