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
   *Natija:* `kg_db_v2_smoke/` katalogida 300 ta yozuv uchun 1024-o'lchamli bge-m3 embeddinglar, `manifest.jsonl`, `build_info.json` va `keyword_index.sqlite3` yaratiladi.

2. **Smoke qidiruv baholashini o'tkazish (5 ta inline so'rov, < 5 daqiqa):**
   ```bash
   .venv/Scripts/python.exe migrate_v2/eval_v2.py --smoke
   ```
   *Natija:* `migrate_v2/out/eval.json` va `migrate_v2/out/EVAL.md` hisobotlari shakllantiriladi.

---

### B. To'liq Baza Qurish (Full Build)

To'liq 31,000+ yozuvli v1 bazasini `kg_db_v2/` ga ko'chirish:

```bash
.venv/Scripts/python.exe migrate_v2/build_v2.py --out kg_db_v2 --batch-size 50 --max-workers 8
```

- v1 bazasi faqat SQLite orqali o'qish rejimida (`mode=ro`) ochiladi, v1 fayllariga yozilmaydi.
- Jarayon to'xtab qolsa ham davom ettirish mumkin (resumable: mavjud IDlar o'tkazib yuboriladi).
- Ollama xatoliklarida eksponensial kutish (backoff) bilan qayta urinish yo'lga qo'yilgan.

---

### C. Sinxronizatsiya (Incremental Sync)

v1 bazasiga yangi yozuvlar qo'shilganda yoki mavjudlari o'zgarganda v2 ni manifest bo'yicha yangilash:

```bash
.venv/Scripts/python.exe migrate_v2/build_v2.py --sync --out kg_db_v2
```

- `manifest.jsonl` dagi SHA-256 xeshlar orqali yangi, o'zgargan va o'chirilgan yozuvlarni aniqlaydi.
- Faqat o'zgargan/yangi yozuvlar qayta embed qilinadi, o'chirilganlari v2 dan chiqariladi.
- Kalit so'z indeksi (`keyword_index.sqlite3`) avtomatik ravishda yangilanadi.

---

### D. To'liq Baholash (Full Evaluation)

Tashqi benchmark so'rovlar fayli (`queries.jsonl`) orqali to'liq baholash:

```bash
.venv/Scripts/python.exe migrate_v2/eval_v2.py --v1-db kg_db --v2-db kg_db_v2 --queries "D:/claude projects/.agy/S-14ecd8/til-test/data/queries.jsonl"
```

- v1 va v2 alohida Python jarayonlarida xolis baholanadi.
- 4 ta so'rov toifasi (`uz_need`, `en_need`, `uz_name`, `en_name`) bo'yicha Recall@10 va MRR@10 hisoblanadi.
- Leksik gate (`MIN_COVERAGE`: `0.8`, `0.5`, `0.0`) darajalarida taqqoslanadi.

---

## ⚠️ 2. Ikki Bosqichli Qo'lda O'tish (Manual Switch)

> [!CAUTION]
> Ushbu o'tish hozir **amalga oshirilmaydi**, faqat hujjatlashtirilgan. O'tish faqat to'liq baholash va foydalanuvchi tasdig'idan so'ng qo'lda bajariladi.

v2 bazasini faol (live) tizimga aylantirish uchun quyidagi 2 bosqich bajariladi:

### 1-bosqich: Muhit o'zgaruvchisini yo'naltirish
Tizimda `GRAPHIFY_DB_PATH` o'zgaruvchisini `kg_db_v2` ga qaratish:
- **Windows PowerShell:**
  ```powershell
  $env:GRAPHIFY_DB_PATH = "D:\claude projects\graphify-ekotizim\Graphify\kg_db_v2"
  ```
- **Tizim doimiy o'zgaruvchisi (.env yoki System Env):**
  ```env
  GRAPHIFY_DB_PATH=D:\claude projects\graphify-ekotizim\Graphify\kg_db_v2
  ```

### 2-bosqich: `models_config.json` faylini yaratish
Loyiha ildizida (`Graphify/models_config.json`) quyidagi konfiguratsiyani yaratish:
```json
{
  "embedder_type": "ollama",
  "config": {
    "ollama": {
      "base_url": "http://localhost:11434",
      "model": "bge-m3"
    }
  }
}
```

Shundan so'ng MCP server va Telegram bot xizmatlarini qayta ishga tushirish kifoya.

---

## 🔄 3. Orqaga Qaytish (Rollback)

Agar v2 bilan kutilmagan muammo yuzaga kelsa, zudlik bilan v1 ga qaytish tartibi:

1. `Graphify/models_config.json` faylini o'chirish yoki nomini o'zgartirish (masalan, `models_config.json.bak`):
   ```powershell
   Remove-Item "models_config.json" -ErrorAction SilentlyContinue
   ```
2. `GRAPHIFY_DB_PATH` muhit o'zgaruvchisini bekor qilish (yoki `kg_db` ga qaytarish):
   ```powershell
   $env:GRAPHIFY_DB_PATH = "D:\claude projects\graphify-ekotizim\Graphify\kg_db"
   ```
3. Xizmatlarni qayta yuklash: Graphify yana asl `kg_db` va lokal MiniLM bilan ishlayveradi.
