"""Graphify setup — registers/unregisters the MCP server in Claude Code.

Run automatically by install.bat / uninstall.bat. Idempotent: safe to re-run.
"""

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent.resolve()
CLAUDE_JSON = Path.home() / ".claude.json"
CLAUDE_MD = Path.home() / ".claude" / "CLAUDE.md"
COMMANDS_DIR = Path.home() / ".claude" / "commands"


def command_body():
    launcher = str(HERE / "start_dashboard.bat")
    return f"""---
description: Graphify xotira bazasi — dashboard ochish, qidirish, saqlash, ro'yxat, token statistikasi
argument-hint: "[bo'sh=dashboard] | qidiruv so'zi | list | save <matn> | token"
allowed-tools: Bash, mcp__graphify__search_knowledge, mcp__graphify__save_memory, mcp__graphify__list_memories, mcp__graphify__delete_memory, mcp__graphify__token_stats
---

Foydalanuvchi `/graphify $ARGUMENTS` buyrug'ini chaqirdi. Argumentga qarab ish tut:

**1. Bo'sh yoki "dashboard"/"ochish":** dashboardni background'da ishga tushir:
```
cmd /c "{launcher}"
```
Keyin ayt: brauzerda http://localhost:5000 ni och. Port band bo'lsa — allaqachon ochiq, shunchaki o'sha manzilni och.

**2. "list"/"ro'yxat":** `list_memories` bilan xotiralarni ro'yxat qil.

**3. "save " bilan boshlansa:** keyingi matnni `save_memory` bilan saqla (mos title/kind tanla).

**4. "token"/"stats"/"statistika"/"tejash":** `token_stats` bilan token tejash foizini qisqa ko'rsat (taxminiy).

**5. Aks holda (qidiruv so'zi):** `search_knowledge` bilan qidir, natijalarni o'zbekcha, o'qishga qulay ko'rsat.

Javobni qisqa va amaliy qil.
"""


MD_START = "<!-- graphify:start -->"
MD_END = "<!-- graphify:end -->"
MD_SECTION = f"""{MD_START}
## Graphify (persistent knowledge base)

The `graphify` MCP server is the user's persistent cross-chat memory (local vector DB, 0 API tokens).

- At the start of a non-trivial task, call `search_knowledge` with the task topic to recover prior decisions, prompts and session summaries before re-reading many files or asking the user.
- At the end of a substantial session, call `save_memory` with a short self-contained summary (what was done, what was decided, next steps), kind="summary" or "decision".
- If a search result contradicts current code, trust the code and suggest `delete_memory` for the stale entry.
{MD_END}
"""


def load_config():
    if CLAUDE_JSON.exists():
        backup = CLAUDE_JSON.with_suffix(f".json.backup-graphify-{datetime.now():%Y%m%d%H%M%S}")
        shutil.copy2(CLAUDE_JSON, backup)
        print(f"  zaxira nusxa: {backup.name}")
        return json.loads(CLAUDE_JSON.read_text(encoding="utf-8"))
    return {}


def save_config(cfg):
    CLAUDE_JSON.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def install():
    cfg = load_config()
    cfg.setdefault("mcpServers", {})["graphify"] = {
        "type": "stdio",
        "command": str(HERE / ".venv" / "Scripts" / "python.exe"),
        "args": [str(HERE / "server.py")],
    }
    save_config(cfg)
    print("  [OK] MCP server ro'yxatdan o'tdi (~/.claude.json)")

    CLAUDE_MD.parent.mkdir(parents=True, exist_ok=True)
    text = CLAUDE_MD.read_text(encoding="utf-8") if CLAUDE_MD.exists() else ""
    if MD_START not in text:
        CLAUDE_MD.write_text(text.rstrip() + "\n\n" + MD_SECTION, encoding="utf-8")
        print("  [OK] Ko'rsatma qo'shildi (~/.claude/CLAUDE.md)")
    else:
        print("  [OK] Ko'rsatma allaqachon mavjud (~/.claude/CLAUDE.md)")

    COMMANDS_DIR.mkdir(parents=True, exist_ok=True)
    (COMMANDS_DIR / "graphify.md").write_text(command_body(), encoding="utf-8")
    print("  [OK] /graphify buyrug'i o'rnatildi (~/.claude/commands/)")

    print("  Embedding modeli tekshirilmoqda (birinchi marta ~80MB yuklanadi)...")
    sys.path.insert(0, str(HERE))
    import server  # noqa: E402 — triggers DB init

    # warm up the embedder so the first real search is fast
    if server._col.count():
        server._col.query(query_texts=["warmup"], n_results=1)
    else:
        server._col.upsert(ids=["_warmup"], documents=["warmup"],
                           metadatas=[{"id": "_warmup", "kind": "note", "title": "warmup",
                                       "source": "setup", "project": "shared",
                                       "date": "2000-01-01", "tags": ""}])
        server._col.delete(ids=["_warmup"])
    print("  [OK] Model tayyor")


def uninstall():
    if CLAUDE_JSON.exists():
        cfg = load_config()
        if cfg.get("mcpServers", {}).pop("graphify", None) is not None:
            save_config(cfg)
            print("  [OK] MCP server o'chirildi (~/.claude.json)")
    if CLAUDE_MD.exists():
        text = CLAUDE_MD.read_text(encoding="utf-8")
        if MD_START in text and MD_END in text:
            head, rest = text.split(MD_START, 1)
            _, tail = rest.split(MD_END, 1)
            CLAUDE_MD.write_text((head.rstrip() + "\n" + tail.lstrip()).strip() + "\n", encoding="utf-8")
            print("  [OK] Ko'rsatma o'chirildi (~/.claude/CLAUDE.md)")
    cmd_file = COMMANDS_DIR / "graphify.md"
    if cmd_file.exists():
        cmd_file.unlink()
        print("  [OK] /graphify buyrug'i o'chirildi")
    print("  Eslatma: xotira bazasi (kg_db/) o'chirilmadi — kerak bo'lmasa papkani qo'lda o'chiring.")


if __name__ == "__main__":
    if "--uninstall" in sys.argv:
        uninstall()
    else:
        install()
