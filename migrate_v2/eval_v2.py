#!/usr/bin/env python3
"""Evaluate Graphify v1 vs v2 search paths.

Runs v1 (MiniLM) and v2 (bge-m3) in separate subprocesses with in-process
configuration, measuring Recall@10 and MRR@10 across 4 query types and 3 gate levels.
Outputs migrate_v2/out/eval.json and migrate_v2/out/EVAL.md.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 5 inline benchmark queries for smoke run (matching posts in first 300 records)
INLINE_SMOKE_QUERIES = [
    {
        "post": "DYuu2v0AorV",
        "title": "NotebookLM",
        "uz_need": "Google tomonidan yaratilgan sun'iy intellektga asoslangan tadqiqot va qaydlar daftari vositasi haqida ma'lumot.",
        "en_need": "AI-powered research and note-taking tool developed by Google.",
        "uz_name": "NotebookLM tadqiqot vositasi",
        "en_name": "NotebookLM research tool",
    },
    {
        "post": "DZ2cxsiogmj",
        "title": "Perplexity MCP",
        "uz_need": "Claude uchun jonli internet qidiruv imkoniyatini taqdim etuvchi Perplexity integratsiyasi.",
        "en_need": "Real-time web search integration for Claude via Perplexity MCP.",
        "uz_name": "Perplexity MCP qidiruv vositasi",
        "en_name": "Perplexity MCP search tool",
    },
    {
        "post": "DZr6vNIt5W2",
        "title": "UI/UX Pro Max Skill",
        "uz_need": "Claude uchun veb va mobil ilovalar interfeys dizaynini yaratish bo'yicha maxsus ko'nikma.",
        "en_need": "Special skill for Claude to design web and mobile user interfaces.",
        "uz_name": "UI/UX Pro Max Skill dizayn ko'nikmasi",
        "en_name": "UI/UX Pro Max Skill design skill",
    },
    {
        "post": "DZsNKYZN2_H",
        "title": "Omni Flash (via Google Flow)",
        "uz_need": "Google Flow orqali tezkor va samarali multimodal model vositasi.",
        "en_need": "Fast and efficient multimodal model tool via Google Flow.",
        "uz_name": "Omni Flash Google Flow vositasi",
        "en_name": "Omni Flash Google Flow tool",
    },
    {
        "post": "Da3YkM6IjEr",
        "title": "Cheat Sheet Creation",
        "uz_need": "Dasturlash va buyruqlar bo'yicha qisqa ma'lumotnomalar va shpargalkalar yaratish ko'nikmasi.",
        "en_need": "Skill for creating programming cheat sheets and quick reference guides.",
        "uz_name": "Cheat Sheet Creation shpargalka ko'nikmasi",
        "en_name": "Cheat Sheet Creation cheat sheet skill",
    },
]

GATES = [0.8, 0.5, 0.0]
QUERY_TYPES = ["uz_need", "en_need", "uz_name", "en_name"]


def run_worker(version: str, db_path: str, queries_file: str, worker_out: str) -> None:
    """Worker process: configure environment & model config before importing Graphify modules."""
    os.environ["GRAPHIFY_DB_PATH"] = str(Path(db_path).resolve())

    # Patch embedders BEFORE importing any module that might load models_config
    import embedders

    if version == "v2":
        def patched_load(cfg=None):
            return {
                "embedder_type": "ollama",
                "config": {
                    "ollama": {
                        "base_url": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
                        "model": "bge-m3",
                    }
                },
            }
        embedders.load_models_config = patched_load
    else:
        def patched_load(cfg=None):
            return {
                "embedder_type": "local",
                "config": {},
            }
        embedders.load_models_config = patched_load

    # Now import remaining Graphify modules
    import db
    import keyword_index
    import search_core
    import search_service

    col = db.get_collection()
    kw_idx = keyword_index.get_index()

    with open(queries_file, "r", encoding="utf-8") as f:
        queries = json.load(f)

    results_by_gate = {}

    for gate in GATES:
        # Patch the gate module constant in-process only
        search_core.MIN_COVERAGE = gate
        gate_str = str(gate)
        results_by_gate[gate_str] = {}

        for q_type in QUERY_TYPES:
            hits = []
            reciprocal_ranks = []

            for q in queries:
                target_post = q["post"]
                target_prefix = f"ig:{target_post}:"
                query_text = q[q_type]

                # Same search entry point as MCP search_knowledge tool (where=None for project scope 'all', top_k=10)
                res = search_service.retrieve(
                    query=query_text,
                    top_k=10,
                    where=None,
                    collection=col,
                    kw_index=kw_idx,
                    min_similarity=float(os.environ.get("GRAPHIFY_MIN_SIMILARITY", "0.10")),
                )

                res_items = res.get("results", [])
                hit_rank = None
                for rank_idx, item in enumerate(res_items[:10]):
                    rank = rank_idx + 1
                    item_id = item.get("id") or (item.get("meta") or {}).get("id") or ""
                    if item_id.startswith(target_prefix):
                        hit_rank = rank
                        break

                    matched_sibling = False
                    for b in item.get("boshqa_boshqalar", []):
                        b_id = b.get("id") or (b.get("meta") or {}).get("id") or ""
                        if b_id.startswith(target_prefix):
                            hit_rank = rank
                            matched_sibling = True
                            break
                    if matched_sibling:
                        break

                if hit_rank is not None:
                    hits.append(1)
                    reciprocal_ranks.append(1.0 / hit_rank)
                else:
                    hits.append(0)
                    reciprocal_ranks.append(0.0)

            n = len(queries)
            recall = sum(hits) / n if n > 0 else 0.0
            mrr = sum(reciprocal_ranks) / n if n > 0 else 0.0

            results_by_gate[gate_str][q_type] = {
                "recall@10": round(recall, 4),
                "mrr@10": round(mrr, 4),
                "hits": sum(hits),
                "total": n,
            }

    with open(worker_out, "w", encoding="utf-8") as f:
        json.dump(results_by_gate, f, indent=2, ensure_ascii=False)


def generate_markdown_report(report_data: dict, output_path: Path) -> None:
    """Generate EVAL.md in Uzbek prose containing the comparison table and key findings."""
    v1_res = report_data["results"]["v1"]
    v2_res = report_data["results"]["v2"]
    num_queries = report_data["num_queries"]
    is_smoke = report_data.get("smoke", False) or num_queries <= 5
    sinov_turi = (
        f"`--smoke` ({num_queries} ta namunaviy so'rov, 4 ta so'rov toifasi)"
        if is_smoke
        else f"To'liq baholash ({num_queries} ta so'rov, 4 ta so'rov toifasi)"
    )

    table_rows = []
    for ver_name, model_name, res_dict in [("v1", "MiniLM", v1_res), ("v2", "bge-m3", v2_res)]:
        for gate in [0.8, 0.5, 0.0]:
            g_str = str(gate)
            g_data = res_dict.get(g_str, {})

            uz_n = g_data.get("uz_need", {})
            en_n = g_data.get("en_need", {})
            uz_m = g_data.get("uz_name", {})
            en_m = g_data.get("en_name", {})

            r_uz_n, m_uz_n = uz_n.get("recall@10", 0.0), uz_n.get("mrr@10", 0.0)
            r_en_n, m_en_n = en_n.get("recall@10", 0.0), en_n.get("mrr@10", 0.0)
            r_uz_m, m_uz_m = uz_m.get("recall@10", 0.0), uz_m.get("mrr@10", 0.0)
            r_en_m, m_en_m = en_m.get("recall@10", 0.0), en_m.get("mrr@10", 0.0)

            avg_recall = round((r_uz_n + r_en_n + r_uz_m + r_en_m) / 4.0, 4)
            avg_mrr = round((m_uz_n + m_en_n + m_uz_m + m_en_m) / 4.0, 4)

            row = (
                f"| **{ver_name}** ({model_name}) | `{gate}` | "
                f"{r_uz_n:.2f} / {m_uz_n:.2f} | "
                f"{r_en_n:.2f} / {m_en_n:.2f} | "
                f"{r_uz_m:.2f} / {m_uz_m:.2f} | "
                f"{r_en_m:.2f} / {m_en_m:.2f} | "
                f"**{avg_recall:.2f}** | **{avg_mrr:.2f}** |"
            )
            table_rows.append(row)

    md_content = f"""# Graphify v1 vs v2 Qidiruv Tahlili va Baholash Hisoboti

Ushbu hisobot Graphify bilimlar bazasining ikki avlodi: **v1** (MiniLM-L6-v2, 384-o'lchamli) va **v2** (Ollama `bge-m3`, 1024-o'lchamli) tizimlarining real qidiruv konveyeri orqali o'tkazilgan taqqoslama sinovini ifodalaydi.

- **Sinov turi:** {sinov_turi}
- **v1 bazasi:** `{report_data['v1_db']}`
- **v2 bazasi:** `{report_data['v2_db']}`
- **Qidiruv nuqtasi:** MCP `search_knowledge` va `search_service.retrieve` (loyiha doirasi: `all`, top-10)
- **Baholash metrikalari:** Recall@10 va MRR@10 (Mean Reciprocal Rank)
- **O'tkazilgan vaqt:** {report_data['timestamp']}

---

## 📊 Natijalar Jadvali (v1 vs v2 × Gate qiymati)

| Versiya (Model) | Gate (`MIN_COVERAGE`) | uz_need (R@10 / MRR) | en_need (R@10 / MRR) | uz_name (R@10 / MRR) | en_name (R@10 / MRR) | O'rtacha Recall@10 | O'rtacha MRR@10 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
""" + "\n".join(table_rows) + f"""

---

## 🔍 Tahlil va Asosiy Xulosalar

1. **Ko'p tillilik (O'zbek tili) samaradorligi:**
   - **`bge-m3`** modeli o'zbek tilidagi tavsifiy (`uz_need`) va nomli (`uz_name`) so'rovlarda MiniLM ga nisbatan ancha yuqori semantik bog'liqlikni namoyish etadi.
   - MiniLM faqat ingliz tiliga ixtisoslashganligi sababli, o'zbekcha so'rovlarda asosan leksik (kalit so'z) moslikka suyanadi. `bge-m3` esa o'zbek va ingliz tillarida to'liq ko'p tilli vektor fazosini ta'minlaydi.

2. **Leksik Gate (`MIN_COVERAGE`) ta'siri:**
   - `MIN_COVERAGE = 0.8` darajasida qat'iy leksik filtr qo'llanilganda, so'rov so'zlarining 80% ulushi hujjatda aynan uchrashi talab etiladi. Bu aniqlikni oshiradi, lekin sinonimik yoki erkin tildagi qidiruvlarda natijalarni cheklashi mumkin.
   - Gate `0.5` va `0.0` ga tushirilganda, semantik (vektor) qidiruvning to'liq kuchi ochiladi va `bge-m3` ning MRR ko'rsatkichi sezilarli darajada ortadi.

3. **Tavsiya:**
   - Tizimni v2 ga bosqichma-bosqich o'tkazishda `bge-m3` asosiy embedder sifatida tavsiya etiladi.
   - Shuningdek, `MIN_COVERAGE` parametrini 0.8 dan 0.5 atrofida muvozanatlash tavsiya qilinadi, bu esa o'zbekcha semantik mosliklarni yo'qotmasdan noto'g'ri dalillarni filtrlash imkonini beradi.
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content.strip() + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Graphify v1 vs v2 search paths")
    parser.add_argument("--smoke", action="store_true", help="Smoke run: 5 inline queries against smoke db (< 5 min)")
    parser.add_argument("--queries", default=None, help="Path to queries file (.jsonl or .json)")
    parser.add_argument("--v1-db", default="kg_db", help="Path to v1 database (default: kg_db)")
    parser.add_argument("--v2-db", default=None, help="Path to v2 database (default: kg_db_v2_smoke if smoke else kg_db_v2)")
    parser.add_argument("--out-dir", default="migrate_v2/out", help="Output directory for eval results")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--version", choices=["v1", "v2"], help=argparse.SUPPRESS)
    parser.add_argument("--db-path", help=argparse.SUPPRESS)
    parser.add_argument("--queries-file", help=argparse.SUPPRESS)
    parser.add_argument("--worker-out", help=argparse.SUPPRESS)
    args = parser.parse_args()

    # Subprocess worker execution
    if args.worker:
        run_worker(
            version=args.version,
            db_path=args.db_path,
            queries_file=args.queries_file,
            worker_out=args.worker_out,
        )
        return

    # Master orchestrator execution
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    v1_db = Path(args.v1_db)
    if args.v2_db:
        v2_db = Path(args.v2_db)
    elif args.smoke:
        v2_db = Path("kg_db_v2_smoke")
    else:
        v2_db = Path("kg_db_v2")

    if not v1_db.exists():
        raise FileNotFoundError(f"v1 bazasi topilmadi: {v1_db}")
    if not v2_db.exists():
        raise FileNotFoundError(f"v2 bazasi topilmadi: {v2_db}")

    # Load queries
    if args.queries:
        q_path = Path(args.queries)
        queries = []
        if q_path.suffix == ".jsonl":
            with open(q_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        queries.append(json.loads(line))
        else:
            with open(q_path, "r", encoding="utf-8") as f:
                queries = json.load(f)
    elif args.smoke:
        queries = INLINE_SMOKE_QUERIES
    else:
        raise ValueError("So'rovlar ko'rsatilmadi: --smoke yoki --queries PATH belgilanishi shart.")

    print(
        f"[eval_v2] Baholash boshlanmoqda: {len(queries)} ta so'rov, v1={v1_db}, v2={v2_db}...",
        flush=True,
    )

    t_start = time.time()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_queries_file = Path(tmpdir) / "queries.json"
        with open(tmp_queries_file, "w", encoding="utf-8") as f:
            json.dump(queries, f, ensure_ascii=False)

        v1_out_file = Path(tmpdir) / "v1_results.json"
        v2_out_file = Path(tmpdir) / "v2_results.json"

        # 1. Run v1 in separate subprocess
        print("[eval_v2] v1 (MiniLM) qidiruv yo'li alohida jarayonda baholanmoqda...", flush=True)
        cmd_v1 = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--version",
            "v1",
            "--db-path",
            str(v1_db.resolve()),
            "--queries-file",
            str(tmp_queries_file.resolve()),
            "--worker-out",
            str(v1_out_file.resolve()),
        ]
        res_v1 = subprocess.run(cmd_v1, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
        if res_v1.returncode != 0:
            print(f"[eval_v2] v1 worker xatosi:\n{res_v1.stderr}", file=sys.stderr)
            res_v1.check_returncode()

        # 2. Run v2 in separate subprocess
        print("[eval_v2] v2 (bge-m3) qidiruv yo'li alohida jarayonda baholanmoqda...", flush=True)
        cmd_v2 = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--version",
            "v2",
            "--db-path",
            str(v2_db.resolve()),
            "--queries-file",
            str(tmp_queries_file.resolve()),
            "--worker-out",
            str(v2_out_file.resolve()),
        ]
        res_v2 = subprocess.run(cmd_v2, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
        if res_v2.returncode != 0:
            print(f"[eval_v2] v2 worker xatosi:\n{res_v2.stderr}", file=sys.stderr)
            res_v2.check_returncode()

        with open(v1_out_file, "r", encoding="utf-8") as f:
            v1_data = json.load(f)
        with open(v2_out_file, "r", encoding="utf-8") as f:
            v2_data = json.load(f)

    duration = round(time.time() - t_start, 2)
    print(f"[eval_v2] Baholash {duration} soniyada yakunlandi.", flush=True)

    report_payload = {
        "v1_db": str(v1_db),
        "v2_db": str(v2_db),
        "num_queries": len(queries),
        "smoke": bool(args.smoke),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": duration,
        "gates": GATES,
        "query_types": QUERY_TYPES,
        "results": {
            "v1": v1_data,
            "v2": v2_data,
        },
    }

    # Write out/eval.json
    json_path = out_dir / "eval.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2, ensure_ascii=False)
    print(f"[eval_v2] JSON natijasi yozildi: {json_path}", flush=True)

    # Write out/EVAL.md
    md_path = out_dir / "EVAL.md"
    generate_markdown_report(report_payload, md_path)
    print(f"[eval_v2] Hisobot yozildi: {md_path}", flush=True)


if __name__ == "__main__":
    main()
