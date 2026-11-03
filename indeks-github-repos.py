"""github-repos dan qimmatli hujjatlarni tanlab indekslash skripti.

Ushbu skript `D:\\claude projects\\graphify-ekotizim\\Promtlarim\\github-repos` ichidagi 18 ta repozitoriyani
tahlil qilib, 8497+ ta fayldan faqat eng yuqori qiymatga ega bo'lgan 300-800 ta hujjatni
tanlaydi va `indexer.py` orqali `--project github-repos --kind doc` sifatida indekslaydi.

Tanlash mezonlari va asoslari:
1. Chiqarib tashlanganlar (Exclude):
   - Taqiqlangan repolar: `dify`, `freellmapi`
   - Dasturiy / qurish / paket kataloglari: node_modules, .git, dist, build, vendor, .venv, .changeset, .github, target, out, tmp
   - Test va benchmark kataloglari: tests, test, testing, spec, specs, e2e, mocks, evals, benchmarks
   - Avtomatik generatsiya qilingan boilerplate: composio-skills (832 ta bir xil integratsiya wrapperi), open-design media shablonlari
   - Boshqa platformalar / muharrirlar duplikat shimlari: .kiro, .opencode, .cursor, .codex, .codebuddy, .gemini, .kimi, .qwen, .trae, legacy-command-shims
   - Tarjimalar: barcha til papkalari (zh-cn, ja, ru, es, de, fr, va h.k.) va til suffiksiga ega fayllar (*.zh.md, README_*.md)
   - Meta va shovqin fayllar: CHANGELOG, LICENSE, CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, requirements.txt, 50 baytdan kichik fayllar

2. Kiritilganlar (Inclusion):
   - Har bir reponing asosiy yo'riqnomalari: README.md, CLAUDE.md, AGENTS.md, ARCHITECTURE.md, CONTEXT.md, GUIDE.md
   - Barcha haqiqiy skilllar: SKILL.md (Claude Code / AI agent protokollari)
   - Loyihaning texnik hujjatlari: docs/ (tarjimalarsiz va rejalashtirish qoralamalarisiz)
   - Claude qoidalari: .claude/*.md
   - Tizimli promptlar: system-prompts-and-models-of-ai-tools va system_prompts_leaks dagi asosiy model promptlari
   - Darsliklar: learn-claude-code (s01_* dan s20_* gacha README qo'llanmalari)
   - Eng yaxshi amaliyotlar: claude-code-best-practice (best-practice va tutorial hujjatlari)
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from db import flush_index

SRC_DIR = Path(r"D:\claude projects\graphify-ekotizim\Promtlarim\github-repos")
GRAPHIFY_DIR = Path(__file__).resolve().parent
INDEXER_PATH = GRAPHIFY_DIR / "indexer.py"

EXTS = {".md", ".txt", ".rst"}

EXCLUDE_REPOS = {"dify", "freellmapi"}

EXCLUDE_DIRS = {
    "node_modules", ".git", "dist", "build", "vendor", "__pycache__", ".venv",
    ".changeset", ".github", "target", "out", "coverage", "tmp", "temp",
    "tests", "test", "testing", "spec", "specs", "e2e", "mocks", "test-fixtures",
    "composio-skills", "evals", "benchmarks", "benchmark", "benchmark-models",
    "de-de", "es", "ja-jp", "ko-kr", "pt-br", "ru", "th", "tr",
    "ur", "vi-vn", "zh-cn", "zh-tw", "zh", "ja", "ko", "fr", "de", "pt", "it", "ar",
    "id", "pl", "nl", "uk", "readmes", "i18n", "changelog",
    ".kiro", ".opencode", ".cursor", ".codex", ".codex-plugin", ".codebuddy",
    ".gemini", ".hermes", ".kimi", ".openclaw", ".qwen", ".trae", "flatpak",
    "legacy-command-shims", "site", "design-systems", "design-templates", "plugins",
    "packages", "apps", "clipper", "craft", "figma-plugin", "tools", "story",
    "old", "raw", "bundled-skills", "injected-reminders", "slash-commands", "mcp-servers",
    ".agents", "assets", "charts", "e2e-coverage", ".plan", "plans", "drafts", "videos",
    "rfc-drafts", "deployment", "handoffs", "adr", "superpowers", "tips"
}

LANG_CODES = {
    "zh", "zh-cn", "zh-tw", "zh_cn", "zh_tw", "ja", "ja-jp", "ko", "ko-kr",
    "ru", "es", "de", "de-de", "fr", "pt", "pt-br", "it", "th", "tr", "ur",
    "ar", "pl", "nl", "uk", "vi", "vi-vn"
}

LANG_RE = re.compile(r"[-_.](?:[a-z]{2}|[a-z]{2}[-_][a-z]{2})$", re.I)


def is_translation(f: Path, rel: Path) -> bool:
    for p in rel.parts:
        if p.lower() in LANG_CODES:
            return True
    stem = f.stem.lower()
    if stem.startswith("readme_") or stem.startswith("readme-"):
        return True
    m = LANG_RE.search(stem)
    if m:
        c = m.group(0)[1:].lower()
        if c in LANG_CODES:
            return True
    return False


def is_meta_or_noise(f: Path) -> bool:
    stem = f.stem.upper()
    if any(stem.startswith(x) for x in ["CHANGELOG", "CHANGES", "HISTORY", "RELEASE", "WHATSNEW"]):
        return True
    if any(stem.startswith(x) for x in ["LICENSE", "LICENCE", "COPYING", "UNLICENSE"]):
        return True
    if any(stem.startswith(x) for x in ["CONTRIBUTING", "CODE_OF_CONDUCT", "SECURITY", "GOVERNANCE", "MAINTAINERS", "AUTHORS", "CONTRIBUTORS"]):
        return True
    if "PULL_REQUEST_TEMPLATE" in stem or "ISSUE_TEMPLATE" in stem:
        return True
    if f.name.lower() in {"requirements.txt", "requirements-dev.txt", "roadmap.md", "references.md"}:
        return True
    return False


def collect_valuable_files(src_dir: Path, target_repo: str = "") -> tuple[dict[str, list[Path]], dict[str, int]]:
    selected_by_repo: dict[str, list[Path]] = {}
    raw_counts_by_repo: dict[str, int] = {}

    for repo in sorted(src_dir.iterdir()):
        if not repo.is_dir() or repo.name.lower() in EXCLUDE_REPOS:
            continue
        if target_repo and repo.name != target_repo:
            continue

        raw_count = 0
        matched: list[Path] = []

        for f in repo.rglob("*"):
            if not f.is_file() or f.suffix.lower() not in EXTS:
                continue
            raw_count += 1

            rel = f.relative_to(repo)
            if any(p.lower() in EXCLUDE_DIRS for p in rel.parts):
                continue
            try:
                if f.stat().st_size < 50:
                    continue
            except OSError:
                continue
            if is_translation(f, rel) or is_meta_or_noise(f):
                continue

            parts_lower = [p.lower() for p in rel.parts]
            is_root = len(rel.parts) == 1
            name_upper = f.name.upper()

            inc = False
            # 1. Asosiy qo'llanmalar (root)
            if is_root and (name_upper.startswith("README") or name_upper in {"CLAUDE.MD", "AGENTS.MD", "ARCHITECTURE.MD", "CONTEXT.MD", "GUIDE.MD"}):
                inc = True
            # 2. SKILL.md (har bir skillning asosiy ta'rifi)
            elif name_upper == "SKILL.MD":
                if repo.name == "open-design":
                    pass
                elif repo.name == "gstack" and (rel.parent.name.startswith("ios-") or "openclaw" in parts_lower):
                    pass
                else:
                    inc = True
            # 3. .claude qoidalari va yo'riqnomalari
            elif ".claude" in parts_lower and name_upper.endswith(".MD") and "agent-teams" not in parts_lower:
                inc = True
            # 4. open-design asosiy arxitektura hujjatlari
            elif repo.name == "open-design" and "docs" in parts_lower:
                if not any(x in f.name.lower() for x in ["announcement", "whats-new", "blog-", "theater", "pets", "troubleshooting", "setup"]):
                    inc = True
            # 5. system-prompts-and-models-of-ai-tools
            elif repo.name == "system-prompts-and-models-of-ai-tools":
                inc = True
            # 6. system_prompts_leaks (asosiy modellar promptlari)
            elif repo.name == "system_prompts_leaks" and len(rel.parts) == 2 and rel.parts[0] not in {"Misc", "assets"}:
                if not name_upper.startswith("README") and not any(k in f.stem.lower() for k in ["personality", "voice-mode"]):
                    inc = True
            # 7. learn-claude-code darslik qo'llanmalari
            elif repo.name == "learn-claude-code" and any(p.startswith("s") and "_" in p for p in parts_lower) and name_upper == "README.MD":
                inc = True
            # 8. claude-code-best-practice qo'llanmalari
            elif repo.name == "claude-code-best-practice" and any(d in parts_lower for d in {"best-practice", "tutorial"}) and "agent-teams" not in parts_lower:
                inc = True

            if inc:
                matched.append(f)

        raw_counts_by_repo[repo.name] = raw_count
        selected_by_repo[repo.name] = matched

    return selected_by_repo, raw_counts_by_repo


def main():
    parser = argparse.ArgumentParser(description="github-repos qimmatli hujjatlarini tanlab indekslash")
    parser.add_argument("--quruq", "--dry-run", action="store_true", help="Faqat tanlangan fayllar va statistikasini ko'rsatish (indekslamasdan)")
    parser.add_argument("--repo", default="", help="Faqat bitta reponi tanlash/indekslash")
    parser.add_argument("--batch-size", type=int, default=40, help="indexer.py ga birdaniga uzatiladigan fayllar soni")
    args = parser.parse_args()

    selected_by_repo, raw_counts = collect_valuable_files(SRC_DIR, target_repo=args.repo)

    total_raw = sum(raw_counts.values())
    total_selected = sum(len(files) for files in selected_by_repo.values())

    print("=" * 70)
    print("       github-repos TANLAB INDEKSLASH HISOBOTI")
    print("=" * 70)
    print(f"{'Repo nomi':<38} | {'Dastlabki':<9} | {'Tanlangan':<9} | {'Qisqarish'}")
    print("-" * 70)
    for repo, files in selected_by_repo.items():
        raw = raw_counts.get(repo, 0)
        sel = len(files)
        pct = f"{(1.0 - (sel / raw)) * 100:.1f}%" if raw > 0 else "0.0%"
        print(f"{repo:<38} | {raw:<9} | {sel:<9} | {pct}")
    print("-" * 70)
    reduction_pct = (1.0 - (total_selected / total_raw)) * 100 if total_raw > 0 else 0
    print(f"{'JAMI:':<38} | {total_raw:<9} | {total_selected:<9} | {reduction_pct:.1f}%")
    print("=" * 70)

    if args.quruq:
        print("\n[QURUQ REJIM] Hech qanday yozuv kiritilmadi. Nishon: 300-800 oralig'ida bo'lishi kerak.")
        if 300 <= total_selected <= 800:
            print(f"[OK] Tanlangan fayllar soni ({total_selected}) belgilangan 300-800 oralig'iga to'liq mos keladi!")
        else:
            print(f"[OGOHLANTIRISH] Tanlangan fayllar soni ({total_selected}) 300-800 oralig'idan tashqarida!")
        return

    # Haqiqiy indekslash
    all_files: list[Path] = []
    for files in selected_by_repo.values():
        all_files.extend(files)

    print(f"\nHaqiqiy indekslash boshlanmoqda: {len(all_files)} ta fayl...")
    batch_size = max(1, args.batch_size)
    total_batches = (len(all_files) + batch_size - 1) // batch_size

    for idx in range(total_batches):
        batch = all_files[idx * batch_size : (idx + 1) * batch_size]
        print(f"[{idx + 1}/{total_batches}] Batch indekslanmoqda ({len(batch)} fayl)...")
        cmd = [
            sys.executable,
            str(INDEXER_PATH),
            *[str(p) for p in batch],
            "--project", "github-repos",
            "--kind", "doc",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        if res.returncode != 0:
            print(f"Xatolik yuz berdi (batch {idx + 1}):", res.stderr, file=sys.stderr)
            sys.exit(res.returncode)

    flush_index()

    print(f"\nMuvaffaqiyatli yakunlandi! Jami {len(all_files)} ta fayl 'github-repos' loyihasiga indekslandi.")


if __name__ == "__main__":
    main()
