"""Claude.ai official data export -> Graphify knowledge base.

Kirish ma'lumoti:
    archive-claude/extracted/
      - conversations.json (343 suhbat)
      - projects/*.json (58 project)
      - design_chats/*.json (14 design chat)

Chiqish .md fayllari:
    knowledge/chats/<sana>-<slug>.md
    knowledge/projects/<slug>.md
    knowledge/design-chats/<slug>.md

Foydalanish:
    .venv\\Scripts\\python.exe import_claude_export.py
    .venv\\Scripts\\python.exe import_claude_export.py --no-index --limit 5
    .venv\\Scripts\\python.exe import_claude_export.py --only chats
"""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
DEFAULT_EXPORT_DIR = Path(r"D:\claude projects\archive-claude\extracted")
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"

CHATS_DIR = ROOT / "knowledge" / "chats"
PROJECTS_DIR = ROOT / "knowledge" / "projects"
DESIGN_DIR = ROOT / "knowledge" / "design-chats"


def slugify(name: str) -> str:
    s = re.sub(r"[^\w\s-]", "", name, flags=re.UNICODE).strip().lower()
    s = re.sub(r"[\s_]+", "-", s)
    return s[:60] or "item"


def chat_to_markdown(conv: dict) -> str:
    name = (conv.get("name") or "").strip() or "Suhbat"
    uuid_str = conv.get("uuid") or ""
    sana = (conv.get("created_at") or "")[:10]

    parts = [f"# {name}", ""]
    meta = []
    if sana:
        meta.append(f"Sana: {sana}")
    if uuid_str:
        meta.append(f"UUID: {uuid_str}")
    if meta:
        parts += ["  \n".join(meta), ""]

    for m in conv.get("chat_messages", []):
        text = ""
        if m.get("content"):
            content_parts = [
                item.get("text", "").strip()
                for item in m["content"]
                if isinstance(item, dict) and item.get("type") == "text" and item.get("text")
            ]
            text = "\n\n".join(p for p in content_parts if p).strip()
        if not text:
            text = (m.get("text") or "").strip()
        if not text:
            continue
        sender = m.get("sender")
        speaker = "Human" if sender == "human" else "Assistant"
        parts += [f"**{speaker}:**", "", text, ""]

    return "\n".join(parts).strip() + "\n"


def project_to_markdown(p: dict) -> str:
    name = (p.get("name") or "").strip() or "Project"
    parts = [f"# {name}", ""]
    meta = []
    if p.get("uuid"):
        meta.append(f"UUID: {p['uuid']}")
    if p.get("description"):
        meta.append(p["description"])
    if p.get("created_at"):
        meta.append(f"Yaratilgan: {p['created_at'][:10]}")
    if p.get("updated_at"):
        meta.append(f"Yangilangan: {p['updated_at'][:10]}")
    if meta:
        parts += ["  \n".join(meta), ""]

    if p.get("prompt_template") and p["prompt_template"].strip():
        parts += ["## Project instructions", "", p["prompt_template"].strip(), ""]

    for d in p.get("docs", []):
        content = (d.get("content") or "").strip()
        if not content:
            continue
        doc_name = d.get("filename") or d.get("name") or "doc"
        parts += [f"## Hujjat: {doc_name}", "", content, ""]

    return "\n".join(parts).strip() + "\n"


def extract_design_message_text(m: dict) -> str:
    c = m.get("content")
    if isinstance(c, dict):
        c = c.get("content")
    if isinstance(c, str):
        return c.strip()
    return ""


def design_to_markdown(d: dict) -> str:
    title = (d.get("title") or "").strip() or "Design Chat"
    uuid_str = d.get("uuid") or ""
    sana = (d.get("created_at") or "")[:10]
    proj = d.get("project")
    proj_name = proj.get("name") if isinstance(proj, dict) else (str(proj) if proj else "")

    parts = [f"# {title}", ""]
    meta = []
    if sana:
        meta.append(f"Sana: {sana}")
    if uuid_str:
        meta.append(f"UUID: {uuid_str}")
    if proj_name:
        meta.append(f"Loyiha: {proj_name}")
    if meta:
        parts += ["  \n".join(meta), ""]

    for m in d.get("messages", []):
        text = extract_design_message_text(m)
        if not text:
            continue
        role = m.get("role")
        speaker = "Human" if role in ("human", "user") else "Assistant"
        parts += [f"**{speaker}:**", "", text, ""]

    return "\n".join(parts).strip() + "\n"


def process_chats(export_dir: Path, limit: int = None):
    conv_file = export_dir / "conversations.json"
    if not conv_file.exists():
        print(f"Ogohlantirish: {conv_file} topilmadi.")
        return [], 0

    CHATS_DIR.mkdir(parents=True, exist_ok=True)
    with open(conv_file, "r", encoding="utf-8") as f:
        convs = json.load(f)

    if limit:
        convs = convs[:limit]

    written = []
    skipped = 0
    used_names = set()

    for c in convs:
        sana = (c.get("created_at") or "")[:10] or "noma-lum"
        name = (c.get("name") or "").strip() or "chat"
        slug = slugify(name)
        base_name = f"{sana}-{slug}"
        fname = base_name
        n = 2
        while fname in used_names:
            fname = f"{base_name}-{n}"
            n += 1
        used_names.add(fname)

        path = CHATS_DIR / f"{fname}.md"
        md = chat_to_markdown(c)
        path.write_text(md, encoding="utf-8")
        written.append((path, "claude-chat-arxiv"))

    return written, skipped


def process_projects(export_dir: Path, limit: int = None, include_empty: bool = False):
    proj_dir = export_dir / "projects"
    if not proj_dir.is_dir():
        print(f"Ogohlantirish: {proj_dir} topilmadi.")
        return [], 0

    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    proj_files = sorted(proj_dir.glob("*.json"))
    if limit:
        proj_files = proj_files[:limit]

    written = []
    skipped = 0
    used_slugs = set()

    for f in proj_files:
        p = json.loads(f.read_text(encoding="utf-8"))
        has_content = bool((p.get("prompt_template") or "").strip()) or any(
            bool((d.get("content") or "").strip()) for d in p.get("docs", [])
        )
        if not has_content and not include_empty:
            skipped += 1
            continue

        slug = slugify(p.get("name") or "project")
        base_slug = slug
        n = 2
        while slug in used_slugs:
            slug = f"{base_slug}-{n}"
            n += 1
        used_slugs.add(slug)

        path = PROJECTS_DIR / f"{slug}.md"
        md = project_to_markdown(p)
        path.write_text(md, encoding="utf-8")
        written.append((path, slug))

    return written, skipped


def process_design(export_dir: Path, limit: int = None):
    design_dir = export_dir / "design_chats"
    if not design_dir.is_dir():
        print(f"Ogohlantirish: {design_dir} topilmadi.")
        return [], 0

    DESIGN_DIR.mkdir(parents=True, exist_ok=True)
    design_files = sorted(design_dir.glob("*.json"))
    if limit:
        design_files = design_files[:limit]

    written = []
    skipped = 0
    used_slugs = set()

    for f in design_files:
        d = json.loads(f.read_text(encoding="utf-8"))
        title = (d.get("title") or "").strip() or "design"
        proj = d.get("project")
        proj_name = proj.get("name") if isinstance(proj, dict) else (str(proj) if proj else "")
        if title.lower() == "chat" and proj_name:
            slug_source = f"{proj_name}-{title}"
        else:
            slug_source = title or proj_name or "design"

        slug = slugify(slug_source)
        base_slug = slug
        n = 2
        while slug in used_slugs:
            slug = f"{base_slug}-{n}"
            n += 1
        used_slugs.add(slug)

        path = DESIGN_DIR / f"{slug}.md"
        md = design_to_markdown(d)
        path.write_text(md, encoding="utf-8")
        written.append((path, "claude-design-arxiv"))

    return written, skipped


def main():
    ap = argparse.ArgumentParser(description="Claude.ai export ma'lumotlarini Graphify bazasiga indekslash")
    ap.add_argument("export_dir", nargs="?", default=str(DEFAULT_EXPORT_DIR),
                    help=f"eksport papkasi yo'li (default: {DEFAULT_EXPORT_DIR})")
    ap.add_argument("--no-index", action="store_true", help="faqat .md yasash, indekslamaslik")
    ap.add_argument("--only", choices=["chats", "projects", "design"],
                    help="faqat bir qismini ishga tushirish (chats, projects, design)")
    ap.add_argument("--limit", type=int, default=None, help="sinov uchun har bir turdan ko'pi bilan N ta olish")
    ap.add_argument("--include-empty", action="store_true", help="kontenti yo'q bo'sh projectlarni ham yozish")
    args = ap.parse_args()

    export_path = Path(args.export_dir)
    if not export_path.exists():
        print(f"Xato: Eksport papkasi topilmadi: {export_path}")
        return 1

    run_chats = args.only in (None, "chats")
    run_projects = args.only in (None, "projects")
    run_design = args.only in (None, "design")

    written_chats, skipped_chats = [], 0
    written_projects, skipped_projects = [], 0
    written_design, skipped_design = [], 0

    if run_chats:
        written_chats, skipped_chats = process_chats(export_path, limit=args.limit)
        print(f"Suhbatlar:    {len(written_chats)} ta .md yozildi, {skipped_chats} ta o'tkazib yuborildi.")

    if run_projects:
        written_projects, skipped_projects = process_projects(
            export_path, limit=args.limit, include_empty=args.include_empty
        )
        print(f"Projectlar:   {len(written_projects)} ta .md yozildi, {skipped_projects} ta o'tkazib yuborildi.")

    if run_design:
        written_design, skipped_design = process_design(export_path, limit=args.limit)
        print(f"Design chat:  {len(written_design)} ta .md yozildi, {skipped_design} ta o'tkazib yuborildi.")

    total_written = len(written_chats) + len(written_projects) + len(written_design)
    total_skipped = skipped_chats + skipped_projects + skipped_design
    print(f"\nJami: {total_written} ta .md fayl yaratildi, {total_skipped} ta o'tkazib yuborildi.")

    if args.no_index or total_written == 0:
        print("Indekslash o'tkazib yuborildi (--no-index).")
        return 0

    py = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable
    idx = str(ROOT / "indexer.py")

    print("\n--- Indekslash boshlanmoqda ---")
    t_start = time.monotonic()

    # 1. Suhbatlar: bitta chaqiruvda papka beriladi (limit bo'lsa har biri alohida)
    if written_chats:
        print("Suhbatlar indekslanmoqda (--project claude-chat-arxiv --kind note)...")
        if args.limit:
            for path, _ in written_chats:
                subprocess.run([py, idx, str(path), "--project", "claude-chat-arxiv", "--kind", "note"], check=True)
        else:
            subprocess.run([py, idx, str(CHATS_DIR), "--project", "claude-chat-arxiv", "--kind", "note"], check=True)

    # 2. Design chatlar: bitta chaqiruvda papka beriladi (limit bo'lsa har biri alohida)
    if written_design:
        print("Design chatlar indekslanmoqda (--project claude-design-arxiv --kind note)...")
        if args.limit:
            for path, _ in written_design:
                subprocess.run([py, idx, str(path), "--project", "claude-design-arxiv", "--kind", "note"], check=True)
        else:
            subprocess.run([py, idx, str(DESIGN_DIR), "--project", "claude-design-arxiv", "--kind", "note"], check=True)

    # 3. Projectlar: har biri o'z yorlig'i bilan alohida
    if written_projects:
        print(f"Projectlar indekslanmoqda ({len(written_projects)} ta alohida yorliq)...")
        for path, label in written_projects:
            subprocess.run([py, idx, str(path), "--project", label, "--kind", "doc"], check=True)

    elapsed = time.monotonic() - t_start
    print(f"\nIndekslash muvaffaqiyatli yakunlandi. Ketgan vaqt: {elapsed:.2f} soniya.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
