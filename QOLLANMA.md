# Graphify — Foydalanish Qo'llanmasi

Graphify — Claude Code uchun **lokal, doimiy xotira**. Barcha qarorlar, promptlar va sessiya xulosalari kompyuteringizdagi vektor bazada saqlanadi. Har yangi chatda Claude bu bazadan faqat kerakli qismini oladi (~1–2K token), loyihani noldan qayta o'rganmaydi.

Hammasi lokal ishlaydi: indekslash va qidirish **0 token**, API kalit kerak emas.

## Bir martalik o'rnatish (allaqachon qilingan ✅)

1. `.venv` — Python muhiti va kutubxonalar o'rnatilgan
2. MCP server `~/.claude.json` ga `graphify` nomi bilan ulangan (user scope — **hamma loyihada, hamma chatda** ishlaydi)

## Kundalik foydalanish — hech narsa qilish shart emas

Yangi chat ochasiz va oddiy ishlayverasiz. Claude o'zi:

- **Vazifa boshida** `search_knowledge` bilan eski qarorlarni qidiradi
- **Sessiya oxirida** `save_memory` bilan xulosani saqlaydi

Qo'lda ham buyura olasiz:

- *"graphify'dan qidir: auth qanday qilingan edi"*
- *"bu qarorni graphify'ga saqla"*
- *"graphify'dagi eslatmalarni ko'rsat"* (list_memories)
- *"mem_xxx eslatmasini o'chir"* (eskirgan qaror bo'lsa)

## Boshqa loyihalarda ishlatish

Server user scope'da ulangani uchun **hamma loyihada avtomatik** ishlaydi — qo'shimcha o'rnatish kerak emas. Har bir xotira qaysi loyiha papkasida saqlanganini avtomatik eslab qoladi (project yorlig'i), shuning uchun bir loyihaning qarorlari boshqasini chalg'itmaydi. Qidiruvda "faqat shu loyihadan qidir" deb ham buyura olasiz.

Eski loyiha uchun bazani tezroq to'ldirish: o'sha loyihada chat ochib, *"loyihani o'rganib chiqib, asosiy arxitektura va qarorlarni graphify'ga saqla"* deng — Claude bir marta o'rganib, xulosalarni saqlaydi, keyingi chatlar shu xotiradan foydalanadi.

## Hujjatlarni indekslash (ixtiyoriy, kerak bo'lganda)

Loyiha hujjatlari, spetsifikatsiya yoki eslatma fayllarini (.md, .txt) bazaga qo'shish:

```powershell
& "D:\claude projects\Graphify\.venv\Scripts\python.exe" "D:\claude projects\Graphify\indexer.py" "C:\yo'l\papka-yoki-fayl"
```

Qayta ishga tushirsangiz, o'sha fayllar yangilanadi (dublikat bo'lmaydi).

## Maslahatlar

- **Sifat > miqdor.** Hamma xom narsani emas, qaror va xulosalarni saqlang — shovqin qidiruvni buzadi.
- Eskirgan qarorlarni o'chirib boring (`delete_memory`), aks holda Claude eski ma'lumotga tayanadi.
- Baza `kg_db/` papkasida — uni backup qilsangiz, butun xotira saqlanadi.

## Muammo bo'lsa

- Claude'da graphify tool'lari ko'rinmasa: chatda `/mcp` yozib server holatini tekshiring; Claude Code'ni qayta ishga tushiring.
- Birinchi ishga tushishda embedding modeli (~80MB) yuklab olinadi — birinchi qidiruv sekinroq bo'lishi normal.
