# Claude Code: Loyihalarni Graphify bazasiga qo'shish (100% aniq)

## Maqsad
Claude.ai dagi barcha projects (57 ta) → Graphify vektor bazasiga indekslash.
- Har project **o'z nomi bilan alohida yorliq** oladi
- Matn **hech qachon** LLM kontekstiga tushmaydı → **0 token**
- Dublikat indexlash yo'q — qayta ishga tushirsak update bo'ladi

## Tayyorlik

**Fayl manzillari (o'tkazmasin):**
- `D:\claude projects\Graphify\import_claude_projects.py` — konvertor skript (allaqachon yozilgan)
- `D:\claude projects\Graphify\indexer.py` — indekser (mavjud)
- `D:\claude projects\Graphify\.venv\Scripts\python.exe` — Python (virtual muhit)
- `%USERPROFILE%\Downloads\claude-projects.json` — brauzerdan yuklab olingan

---

## QADAM 1: Brauzerdan projects yuklab olish (100% manual)

**Qo'lda bajaring, men help berayotgan bo'lganman:**

1. Bu chatni **ochiq qoldir** (yopma)
2. **Yangi brauzer tab**da https://claude.ai/projects ochish
3. JavaScript console (F12 → Console) qo'lga oling
4. Quyidagi kodi copy-paste qilib console'da Run qiling:

```javascript
async function downloadProjects() {
  const orgs = await fetch('/api/organizations', {headers:{accept:'application/json'}})
    .then(r => r.json());
  const workOrg = orgs.find(o => !(o.capabilities || []).includes('api_individual')).uuid;
  const projects = await fetch(`/api/organizations/${workOrg}/projects`, {headers:{accept:'application/json'}})
    .then(r => r.json());
  
  const dump = [];
  for (const p of projects) {
    const det = await fetch(`/api/organizations/${workOrg}/projects/${p.uuid}`, {headers:{accept:'application/json'}})
      .then(r => r.ok ? r.json() : null);
    if (!det) continue;
    
    const docs = await fetch(`/api/organizations/${workOrg}/projects/${p.uuid}/docs`, {headers:{accept:'application/json'}})
      .then(r => r.ok ? r.json() : []);
    
    dump.push({
      uuid: p.uuid,
      name: p.name,
      description: p.description || '',
      created_at: p.created_at,
      updated_at: p.updated_at,
      archived: !!p.archived_at,
      instructions: det.prompt_template || '',
      docs: (Array.isArray(docs) ? docs : []).map(d => ({
        name: d.file_name || d.name || 'doc',
        content: d.content || ''
      }))
    });
  }
  
  const blob = new Blob([JSON.stringify(dump, null, 2)], {type: 'application/json'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'claude-projects.json';
  a.click();
  console.log(`✓ Yuklab olindi: ${dump.length} ta project`);
}
downloadProjects();
```

**Natija:** `claude-projects.json` fayli Downloads papkangizga tushadi.

**Tekshirish:** `cd %USERPROFILE%\Downloads && dir claude-projects.json` (fayl bo'lishi kerak)

---

## QADAM 2: Python skriptini ishga tushirish (Claude Code orqali)

Claude Code (Desktop) terminalida **quyidagi buyruqni** ishga tushiring:

```bash
cd D:\claude projects\Graphify && .venv\Scripts\python.exe import_claude_projects.py "%USERPROFILE%\Downloads\claude-projects.json"
```

### Nima bo'ladi?

1. ✅ `knowledge/projects/` papkasi yaratiladi
2. ✅ Har project → `NAME.md` (kontenti bo'lsa)
3. ✅ 57 ta projectdan ~13-20 tasida content bo'ladi (bo'shlari o'tkazib yuboriladi)
4. ✅ **Har bir .md fayl indexer.py bilan tanlangan project nomi bilan** (`situatsion-sentre`, `misp`, `uzb-cloud`, ...) **yorliqlanadi**
5. ✅ Chromadb bazasiga (`kg_db/`) qo'shiladi → **0 token sarflanmaydi**

### Kutilgan output:

```
yozildi: situatsion-sentre.md  (14523 belgi)
yozildi: misp.md  (8234 belgi)
yozildi: uzb-cloud.md  (6123 belgi)
...
13 ta .md yozildi, 44 ta bo'sh project o'tkazib yuborildi.

Indexing situatsion-sentre...
Indexing misp...
...
✓ Tayyor. 13 ta project alohida yorliq bilan bazaga qo'shildi.
```

---

## QADAM 3: Natijani tekshirish

**3a. Fayllar tekshirish:**
```bash
dir D:\claude projects\Graphify\knowledge\projects\
```
Burada `.md` fayllarini ko'rmalisu.

**3b. Bazani tekshirish:**
```bash
cd D:\claude projects\Graphify && .venv\Scripts\python.exe -c "
import chromadb
client = chromadb.PersistentClient(path='kg_db')
col = client.get_or_create_collection('graphify')
print(f'Total chunks: {col.count()}')
results = col.query(query_texts=['situation center'], n_results=3)
for i, (id, meta, dist) in enumerate(zip(results['ids'][0], results['metadatas'][0], results['distances'][0])):
    print(f'{i+1}. {meta.get(\"project\", \"?\")} → {meta.get(\"title\", id)} (yaqinlik: {1-dist:.2f})')
"
```

**3c. Claude Code chatida qidirish (test):**
```
graphify dan qidir: "situation center" yoki "misp"
```

Claude qidiruvda topgan natijalarni ko'rsatadi va ularni qaysi projectdan olinganini yozadi.

---

## Xatolar va yechim

| Xato | Yechim |
|---|---|
| "Python topilmadi" | `.venv\Scripts\python.exe` to'liq yo'li bilan chiqaring |
| "claude-projects.json topilmadi" | Downloads papkasida fayl borligini tekshiring; brauzer console'da kodi qayta ishga tushiring |
| "Permission error (chromadb)" | `kg_db/` papka o'chirilgan bo'lsa, qayta yaratiladi (xatosi yo'q) |
| Indexing sekin (birinchi marta) | Normal — embedding modeli (80 MB) yuklanayapti, keyingi ishga tushirishda tez bo'ladi |
| Fayllar `.md` emas, `.txt` ko'rinadi | Xatosi yo'q — Python faqat `.md` yozadi, Windows Explorer'da ko'rinishi boshqa bo'lsa ham |

---

## Qandaysiz?

### Agar o'tkazib yuborsangiz (re-import):
Skriptni qayta ishga tushiring — dublikat bo'lmaydi, mavjud chunk'lar "upsert" (yangilanadi) bo'ladi.

### Agar boshqa projectni indexlamosiz:
```bash
.venv\Scripts\python.exe indexer.py "D:\path\to\your\markdown" --project MyProject --kind doc
```

### Agar hammasi bitta yorliq ostida bo'lsin istasangiz:
```bash
.venv\Scripts\python.exe import_claude_projects.py "%USERPROFILE%\Downloads\claude-projects.json" --single-label claude-projects
```

---

## Token samaradorligi

| Qadam | Token |
|---|---|
| Projects yuklab olish (JS console) | 0 |
| .md ga konvertatsiya | 0 |
| Embedding + indexing | 0 (lokal model) |
| **Jami** | **0 ✓** |

**Agar 246K token ortilsa, Claude ularni o'qib turgan edi → skript bekor.**

---

## XULOSA: Ishlar ketma-ketligi

1. ✅ **Brauzer tab**da: JavaScript console → kod ishga tushir → `claude-projects.json` Downloads'ga tush
2. ✅ **Claude Code terminal**da: `import_claude_projects.py` buyrug'i → 57 project indekslanadi
3. ✅ **Tekshirish**: `graphify query "misp"` yoki `graphify query "situation center"` — topiladi
4. ✅ **Done**: Hamma loyiha bilimlar bazasida, har bitta o'z nomi bilan qidiruvga tayyor

---

**Agar qandaysiz bo'lsa yoki xato bo'lsa, bu prompt bilan Claude Code yoki men bilan chat oching — skript ready.**
