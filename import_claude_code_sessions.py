"""Claude Code local sessions -> Graphify knowledge base.

Lokal Claude Code sessiyalarini (.jsonl) o'qib, ularni tozalangan Markdown
shakliga keltiradi (knowledge/code-sessions/<papka-slug>/<sana>-<slug>.md),
so'ng indexer.py orqali Graphify vektor bazasiga indekslaydi.

Foydalanish:
    # Standart manba (C:\\Users\\user\\.claude\\projects)
    .venv\\Scripts\\python.exe import_claude_code_sessions.py

    # Sinov / test uchun namuna papkadan o'qish, indekslamaslik
    .venv\\Scripts\\python.exe import_claude_code_sessions.py --src .sample-sessions --no-index

    # Faqat bitta papka va limit bilan
    .venv\\Scripts\\python.exe import_claude_code_sessions.py --src .sample-sessions --only D--claude-projects-AI-academiya --limit 1 --no-index
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
DEFAULT_SRC = Path(r"C:\Users\user\.claude\projects")
OUTPUT_BASE = ROOT / "knowledge" / "code-sessions"


def slugify(name: str) -> str:
    """Nomdan xavfsiz fayl/yorliq slug yasaydi."""
    s = re.sub(r"[^\w\s-]", "", name, flags=re.UNICODE).strip().lower()
    s = re.sub(r"[\s_]+", "-", s)
    return s[:60] or "chat"


def folder_to_project_label(folder_name: str) -> str:
    """Papka nomidan cc- prefiksli indexer project yorlig'i yasaydi.

    Masalan:
        D--claude-projects-AI-academiya -> cc-ai-academiya
        D--claude-projects-TIUE         -> cc-tiue
        D--claude-projects              -> cc-claude-projects
        C--Users-user-Desktop-Uzbekonacisco-uz -> cc-uzbekonacisco-uz
    """
    s = folder_name.strip()
    m = re.match(r"^[A-Za-z]--claude-projects-(.+)$", s, re.IGNORECASE)
    if m:
        base = m.group(1)
    elif re.match(r"^[A-Za-z]--claude-projects$", s, re.IGNORECASE):
        base = "claude-projects"
    else:
        s_clean = re.sub(r"^[A-Za-z]--", "", s)
        desktop_m = re.search(r"desktop-(.+)$", s_clean, re.IGNORECASE)
        if desktop_m:
            base = desktop_m.group(1)
        else:
            base = s_clean
    return f"cc-{slugify(base)}"


def format_tools_chain(tools: list[str]) -> str:
    """Vositalar zanjirini ixcham ko'rinishga keltiradi.

    Ketma-ket takrorlangan bir xil vositalarni xN bilan qisqartiradi:
        ['ToolSearch', 'Bash', 'Bash'] -> '_[tools: ToolSearch -> Bash x2]_'
    """
    if not tools:
        return ""
    grouped: list[tuple[str, int]] = []
    for t in tools:
        if grouped and grouped[-1][0] == t:
            grouped[-1] = (t, grouped[-1][1] + 1)
        else:
            grouped.append((t, 1))
    items = []
    for name, count in grouped:
        if count > 1:
            items.append(f"{name} x{count}")
        else:
            items.append(name)
    chain_str = " -> ".join(items)
    return f"_[tools: {chain_str}]_"


def parse_session_file(file_path: Path) -> tuple[dict | None, str | None]:
    """JSONL sessiya faylini xotirani tejab qatorma-qator o'qiydi.

    Faqat text va tool_use (marker sifatida) saqlanadi.
    tool_result, thinking va boshqa ortiqcha ma'lumotlar tashlanadi.
    """
    session_id = None
    first_ts = None
    cwds = []
    custom_titles = []
    ai_titles = []
    first_user_text = None
    messages = []
    raw_size = file_path.stat().st_size

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    continue

                t = data.get("type")
                ts = data.get("timestamp") or data.get("created_at")
                if ts and not first_ts:
                    first_ts = str(ts)

                cwd = data.get("cwd")
                if cwd and cwd not in cwds:
                    cwds.append(str(cwd))

                sid = data.get("sessionId")
                if sid and not session_id:
                    session_id = str(sid)

                if t == "custom-title":
                    ct = (data.get("customTitle") or "").strip()
                    if ct:
                        custom_titles.append(ct)
                elif t == "ai-title":
                    at = (data.get("aiTitle") or "").strip()
                    if at:
                        ai_titles.append(at)
                elif t in ("user", "assistant"):
                    msg = data.get("message") or {}
                    raw_content = msg.get("content")

                    text_parts = []
                    tool_names = []
                    if isinstance(raw_content, str):
                        s = raw_content.strip()
                        if s:
                            text_parts.append(s)
                    elif isinstance(raw_content, list):
                        for b in raw_content:
                            if isinstance(b, dict):
                                b_type = b.get("type")
                                if b_type == "text":
                                    txt = (b.get("text") or "").strip()
                                    if txt:
                                        text_parts.append(txt)
                                elif b_type == "tool_use":
                                    name = b.get("name") or "tool"
                                    tool_names.append(name)
                                # tool_result, thinking, image, document e'tiborga olinmaydi
                            elif isinstance(b, str):
                                s = b.strip()
                                if s:
                                    text_parts.append(s)

                    content_str = "\n\n".join(text_parts).strip()
                    if t == "user":
                        if content_str:
                            if not first_user_text:
                                first_user_text = " ".join(content_str.split())[:60]
                            messages.append({"role": "Human", "text": content_str, "tools": []})
                    elif t == "assistant":
                        if content_str or tool_names:
                            messages.append({"role": "Assistant", "text": content_str, "tools": tool_names})
    except OSError as e:
        return None, f"Faylni o'qishda xatolik: {e}"

    # Sarlavha tanlash tartibi:
    # 1) custom-title qatori
    # 2) ai-title qatori
    # 3) birinchi user xabarining dastlabki 60 belgisi
    # 4) chat
    title = ""
    if custom_titles:
        title = custom_titles[-1]
    elif ai_titles:
        title = ai_titles[-1]
    elif first_user_text:
        title = first_user_text
    else:
        title = "chat"

    sana = (first_ts or "")[:10]
    cwd_str = ", ".join(cwds) if cwds else ""
    session_id = session_id or file_path.stem

    return {
        "title": title,
        "sana": sana,
        "session_id": session_id,
        "cwd": cwd_str,
        "messages": messages,
        "raw_size": raw_size,
    }, None


def session_to_markdown(session_data: dict) -> str:
    """Sessiya ma'lumotlaridan toza Markdown matnini tuzadi."""
    title = session_data["title"]
    sana = session_data["sana"]
    session_id = session_data["session_id"]
    cwd = session_data["cwd"]
    messages = session_data["messages"]

    parts = [f"# {title}", ""]
    meta = []
    if sana:
        meta.append(f"Sana: {sana}")
    if session_id:
        meta.append(f"Session ID: {session_id}")
    if cwd:
        meta.append(f"CWD: {cwd}")
    if meta:
        parts += ["  \n".join(meta), ""]

    pending_tools: list[str] = []

    def flush_pending_tools():
        nonlocal pending_tools
        if pending_tools:
            parts.extend([format_tools_chain(pending_tools), ""])
            pending_tools = []

    for msg in messages:
        role = msg["role"]
        text = msg.get("text", "")
        tools = msg.get("tools", [])

        if role == "Human":
            flush_pending_tools()
            if text:
                parts.extend(["**Human:**", "", text, ""])
        elif role == "Assistant":
            if not text:
                pending_tools.extend(tools)
            else:
                flush_pending_tools()
                parts.extend(["**Assistant:**", "", text, ""])
                if tools:
                    parts.extend([format_tools_chain(tools), ""])

    flush_pending_tools()

    content = "\n".join(parts).strip() + "\n"
    content = re.sub(r"\n{4,}", "\n\n\n", content)
    return content


def save_session_markdown(folder_out_dir: Path, session_data: dict) -> tuple[Path, int]:
    """Sessiyani .md fayliga yozadi (idempotent, dublikatsiz)."""
    sana = session_data["sana"]
    slug = slugify(session_data["title"])
    session_id = session_data["session_id"]

    base_name = f"{sana}-{slug}" if sana else slug
    target_file = folder_out_dir / f"{base_name}.md"

    # Agar fayl mavjud bo'lsa, u aynan shu sessiyaga tegishlimi tekshiramiz
    if target_file.exists():
        try:
            head = target_file.read_text(encoding="utf-8", errors="ignore")[:500]
            if f"Session ID: {session_id}" not in head:
                target_file = folder_out_dir / f"{base_name}-{session_id[:8]}.md"
        except OSError:
            pass

    md_content = session_to_markdown(session_data)
    target_file.write_text(md_content, encoding="utf-8")
    md_size = len(md_content.encode("utf-8"))
    return target_file, md_size


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Claude Code local sessions (.jsonl) -> Graphify knowledge base"
    )
    ap.add_argument(
        "--src",
        default=str(DEFAULT_SRC),
        help=f"Manba papka (default: {DEFAULT_SRC})",
    )
    ap.add_argument(
        "--no-index",
        action="store_true",
        help="Faqat .md yasash, bazaga indekslamaslik",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Ko'pi bilan N ta sessiyani qayta ishlash",
    )
    ap.add_argument(
        "--only",
        type=str,
        default=None,
        metavar="PAPKA_SLUG",
        help="Faqat ko'rsatilgan papka-slugni qayta ishlash (masalan, D--claude-projects-AI-academiya)",
    )
    args = ap.parse_args()

    src_dir = Path(args.src)
    if not src_dir.exists():
        print(f"Xatolik: manba papka topilmadi: {src_dir}")
        return 1

    # Papkalar tuzilishini aniqlash:
    # 1) Agar src_dir ichida to'g'ridan-to'g'ri .jsonl fayllar bo'lsa:
    direct_jsonls = list(src_dir.glob("*.jsonl"))
    if direct_jsonls:
        folders = [(src_dir.name, src_dir)]
    else:
        # 2) src_dir ichidagi har bir subdirektoriya bitta papka-slug
        folders = [
            (d.name, d)
            for d in sorted(src_dir.iterdir())
            if d.is_dir() and not d.name.startswith(".")
        ]

    if args.only:
        folders = [(name, p) for name, p in folders if name == args.only]
        if not folders:
            print(f"Ogohlantirish: '--only {args.only}' bo'yicha papka topilmadi.")
            return 0

    total_read = 0
    total_written = 0
    total_skipped = 0
    total_raw_bytes = 0
    total_md_bytes = 0

    # {folder_slug: (folder_out_dir, project_label, written_count)}
    folders_summary = {}

    limit_reached = False
    start_time = time.time()

    for folder_slug, folder_path in folders:
        if limit_reached:
            break

        jsonl_files = sorted(folder_path.glob("*.jsonl"))
        if not jsonl_files:
            continue

        folder_out_dir = OUTPUT_BASE / folder_slug
        folder_out_dir.mkdir(parents=True, exist_ok=True)
        project_label = folder_to_project_label(folder_slug)

        folder_written_count = 0

        for f in jsonl_files:
            if args.limit is not None and total_read >= args.limit:
                limit_reached = True
                break

            total_read += 1
            data, err = parse_session_file(f)
            if err:
                print(f"Xato [{f.name}]: {err}")
                total_skipped += 1
                continue

            if not data or not data["messages"]:
                print(f"Bo'sh sessiya o'tkazib yuborildi: {folder_slug}/{f.name}")
                total_skipped += 1
                continue

            target_file, md_size = save_session_markdown(folder_out_dir, data)
            total_written += 1
            folder_written_count += 1
            total_raw_bytes += data["raw_size"]
            total_md_bytes += md_size

            print(
                f"Yozildi: {folder_slug}/{target_file.name} "
                f"(xom: {data['raw_size']:,} B -> md: {md_size:,} B)"
            )

        if folder_written_count > 0:
            folders_summary[folder_slug] = (folder_out_dir, project_label, folder_written_count)

    elapsed = time.time() - start_time
    pct_saved = (
        (1 - (total_md_bytes / total_raw_bytes)) * 100
        if total_raw_bytes > 0
        else 0.0
    )

    print("\n" + "=" * 50)
    print("YAKUNIY PROGRESS VA STATISTIKA:")
    print(f"  O'qilgan sessiyalar soni       : {total_read}")
    print(f"  Yozilgan .md fayllar soni       : {total_written}")
    print(f"  Bo'sh/xabarsiz o'tkazilganlar   : {total_skipped}")
    print(f"  Xom .jsonl umumiy hajmi        : {total_raw_bytes:,} bayt ({total_raw_bytes / (1024*1024):.2f} MB)")
    print(f"  Yaratilgan .md umumiy hajmi    : {total_md_bytes:,} bayt ({total_md_bytes / 1024:.2f} KB)")
    print(f"  Hajm qisqarishi (tejaldi)      : {pct_saved:.1f}%")
    print(f"  Sarflangan vaqt                : {elapsed:.2f} soniya")
    print("=" * 50)

    if args.no_index:
        print("\nIndekslash o'tkazib yuborildi (--no-index).")
        return 0

    if not folders_summary:
        print("\nIndekslash uchun yangi fayl yo'q.")
        return 0

    py = sys.executable
    idx = str(ROOT / "indexer.py")

    print("\nGraphify vektor bazasiga indekslash boshlanmoqda...")
    for folder_slug, (folder_out_dir, project_label, count) in folders_summary.items():
        print(f"\nIndekslanmoqda: {folder_slug} ({count} ta fayl) -> --project {project_label} --kind note")
        cmd = [py, idx, str(folder_out_dir), "--project", project_label, "--kind", "note"]
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"Xatolik: indekslash to'xtadi (kod: {res.returncode})")
            return res.returncode

    print(f"\nTayyor. Barcha sessiyalar muvaffaqiyatli indekslandi.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
