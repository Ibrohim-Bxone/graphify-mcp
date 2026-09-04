"""Claude.ai Projects -> Graphify knowledge base.

Brauzerdan yuklab olingan claude-projects.json ni har bir project uchun alohida
.md fayliga aylantiradi (knowledge/projects/), so'ng ularni indexer.py bilan
bazaga indekslaydi.

Matn hech qachon LLM kontekstiga tushmaydi -> 0 token.

Foydalanish:
    .venv\\Scripts\\python.exe import_claude_projects.py "%USERPROFILE%\\Downloads\\claude-projects.json"
    .venv\\Scripts\\python.exe import_claude_projects.py <json> --no-index   # faqat .md yasash
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
OUT_DIR = ROOT / "knowledge" / "projects"


def slugify(name: str) -> str:
    s = re.sub(r"[^\w\s-]", "", name, flags=re.UNICODE).strip().lower()
    s = re.sub(r"[\s_]+", "-", s)
    return s[:60] or "project"


def to_markdown(p: dict) -> str:
    parts = [f"# {p['name']}", ""]
    meta = []
    if p.get("description"):
        meta.append(p["description"])
    if p.get("created_at"):
        meta.append(f"Yaratilgan: {p['created_at'][:10]}")
    if p.get("updated_at"):
        meta.append(f"Yangilangan: {p['updated_at'][:10]}")
    if p.get("archived"):
        meta.append("Holat: arxivlangan")
    if meta:
        parts += ["  \n".join(meta), ""]

    if p.get("instructions"):
        parts += ["## Project instructions", "", p["instructions"].strip(), ""]

    for d in p.get("docs", []):
        if not d.get("content"):
            continue
        parts += [f"## Hujjat: {d.get('name', 'doc')}", "", d["content"].strip(), ""]

    return "\n".join(parts).strip() + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", help="brauzerdan yuklangan claude-projects.json")
    ap.add_argument("--no-index", action="store_true", help="faqat .md yasash, indekslamaslik")
    ap.add_argument("--include-empty", action="store_true",
                    help="instructions/hujjati yo'q bo'sh projectlarni ham yozish")
    ap.add_argument("--single-label", metavar="NOM",
                    help="hammasini bitta yorliq ostida saqlash (default: har project o'z nomi bilan)")
    args = ap.parse_args()

    data = json.loads(Path(args.json_path).read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    written = []  # (path, label) juftliklari
    skipped = 0
    used = set()
    for p in data:
        has_content = bool(p.get("instructions")) or any(d.get("content") for d in p.get("docs", []))
        if not has_content and not args.include_empty:
            skipped += 1
            continue
        label = slugify(p["name"])
        n = 2
        while label in used:
            label = f"{slugify(p['name'])}-{n}"
            n += 1
        used.add(label)
        path = OUT_DIR / f"{label}.md"
        md = to_markdown(p)
        path.write_text(md, encoding="utf-8")
        written.append((path, label))
        print(f"yozildi: {path.name}  ({len(md)} belgi)")

    print(f"\n{len(written)} ta .md yozildi, {skipped} ta bo'sh project o'tkazib yuborildi.")

    if args.no_index or not written:
        print("Indekslash o'tkazib yuborildi.")
        return 0

    py = sys.executable
    idx = str(ROOT / "indexer.py")
    if args.single_label:
        subprocess.run([py, idx, str(OUT_DIR), "--project", args.single_label, "--kind", "doc"], check=True)
    else:
        for path, label in written:
            subprocess.run([py, idx, str(path), "--project", label, "--kind", "doc"], check=True)
    print(f"\nTayyor. {len(written)} ta project alohida yorliq bilan bazaga qo'shildi.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
