# Graphify ↔ ChatGPT

Ikki xil ulanish yo'li bor. Ular boshqa-boshqa narsa — aralashtirmang.

## 1. Codex (ChatGPT Plus ichida) — TAYYOR, to'liq o'qish+yozish

Codex CLI/app stdio MCP'ni qo'llaydi, ya'ni `server.py` o'zgarishsiz ishlaydi.

`~/.codex/config.toml` dagi `[mcp_servers.graphify]` yo'li tuzatildi (2026-09-05).
Ilgari `D:\claude projects\Graphify\...` ni ko'rsatardi — u papka
`graphify-ekotizim\` ichiga ko'chirilgani uchun ulanish uzilgan edi.

Ishlashi uchun Codex'ni qayta ishga tushiring. Tekshirish: Codex'da
`search_knowledge` asbobi ro'yxatda ko'rinishi kerak.

⚠️ `PROJECT` cwd papka nomidan olinadi — Codex qaysi papkadan ochilgan bo'lsa,
saqlangan xotira o'sha nom bilan taglanadi.

## 2. chatgpt.com (brauzer/ilova) — `server_http.py` + tunnel

ChatGPT custom connector faqat **public HTTPS + streamable HTTP** bilan ishlaydi,
stdio bilan emas. Shuning uchun `server_http.py` yozildi: u `server.py` dagi
o'sha FastMCP namunasini oladi, hech nimani o'zgartirmaydi, faqat HTTP orqali
chiqaradi va ustiga `search` / `fetch` aliaslarini qo'shadi (ChatGPT aynan shu
ikki nomni kutadi; ikkalasi ham `project="all"` bilan qidiradi, chunki bu yerda
loyiha konteksti yo'q).

### Ishga tushirish

```bash
cd "D:/claude projects/graphify-ekotizim/Graphify" && ./.venv/Scripts/python.exe server_http.py
```

Ekranga to'liq URL chiqadi: `http://127.0.0.1:8787/mcp/<token>`.
Token `.http-token` faylida saqlanadi (`.gitignore` ga qo'shilgan).

### Tunnel (cloudflared — bepul, akkaunt shart emas)

Hozircha o'rnatilmagan. O'rnatish:

```bash
winget install --id Cloudflare.cloudflared
```

Keyin:

```bash
cloudflared tunnel --url http://127.0.0.1:8787
```

U `https://<tasodifiy>.trycloudflare.com` beradi. ChatGPT'ga beriladigan manzil:
`https://<tasodifiy>.trycloudflare.com/mcp/<token>`

### ChatGPT'da ulash

Settings → Connectors → Advanced → Developer mode → Create connector →
MCP server URL: yuqoridagi to'liq URL, Authentication: **No authentication**.

### Cheklovlar — bilib turing

- **Plus tarifida custom connector faqat o'qish** (search/fetch). `save_memory`
  chatgpt.com'dan ishlamaydi. To'liq yozish uchun Business/Enterprise kerak,
  yoki 1-yo'ldan (Codex) foydalaning.
- **Quick tunnel URL'i har ishga tushirishda o'zgaradi** va ChatGPT'da qayta
  kiritish kerak bo'ladi. Barqaror manzil kerak bo'lsa — nomlangan cloudflare
  tunnel (domen talab qiladi).
- **Xavfsizlik:** tunnel ochiq internetda. Yagona himoya — URL ichidagi token.
  Uni hech kimga bermang; sizib ketsa `.http-token` ni o'chirib serverni qayta
  ishga tushiring (yangi token yaraladi).
- Kompyuter o'chsa yoki tunnel yopilsa — ChatGPT connector ishlamaydi.
