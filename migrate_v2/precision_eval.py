#!/usr/bin/env python3
"""Evaluate Graphify v2 search precision on the labelled query set.

Measures Recall@1/@5/@10, MRR@10, false answer rates on near-miss and unanswerable
queries, latency (p50/p95), and recommends optimal reranker alpha, lexical fusion weight,
and acceptance threshold from measured numbers.
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def get_parent_prefix(doc_id: str) -> str:
    """Extract parent prefix from a chunk id or doc id."""
    if not doc_id:
        return ""
    clean = doc_id.strip()
    parts = clean.rsplit(":", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0] + ":"
    if clean.endswith(":"):
        return clean
    return clean + ":"


def extract_item_ids(item: dict) -> list[str]:
    """Extract all identifiable IDs and aliases from a retrieved item."""
    ids = []
    meta = item.get("meta") or {}
    for key in ("id", "canonical_id", "parent_id", "source"):
        val = item.get(key) or meta.get(key)
        if val and isinstance(val, str):
            ids.append(val.strip())

    try:
        import search_policy
        sk = search_policy.source_key(meta)
        if sk:
            ids.append(str(sk).strip())
        sc = search_policy.shortcode(meta)
        if sc:
            ids.append(str(sc).strip())
            ids.append(f"ig:{sc}".strip())
    except Exception:
        pass

    return list(dict.fromkeys(ids))


def is_hit(item: dict, expected_ids: list[str]) -> bool:
    """Check if retrieved item or any of its siblings matches expected IDs or parent prefix."""
    if not expected_ids:
        return False

    clean_expected = [str(e).strip() for e in expected_ids if e]
    if not clean_expected:
        return False

    all_candidate_items = [item]
    for sibling in item.get("boshqa_boshqalar", []):
        if isinstance(sibling, dict):
            all_candidate_items.append(sibling)

    for cand in all_candidate_items:
        cand_ids = extract_item_ids(cand)
        for cand_id in cand_ids:
            cand_prefix = get_parent_prefix(cand_id)
            for exp_id in clean_expected:
                if cand_id == exp_id:
                    return True
                if cand_id.startswith(exp_id + ":"):
                    return True
                if exp_id.startswith(cand_id + ":"):
                    return True
                exp_prefix = get_parent_prefix(exp_id)
                if cand_prefix and exp_prefix and cand_prefix == exp_prefix:
                    return True
                if cand_prefix and cand_prefix.startswith(exp_prefix):
                    return True
                if exp_prefix and exp_prefix.startswith(cand_prefix):
                    return True

    return False


def compute_percentile(values: list[float], p: float) -> float:
    """Compute the p-th percentile of a list of numeric values."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    n = len(sorted_v)
    if n == 1:
        return float(sorted_v[0])
    k = (n - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_v[int(k)])
    d0 = sorted_v[f] * (c - k)
    d1 = sorted_v[c] * (k - f)
    return float(d0 + d1)


def compute_metrics_at_threshold(query_results: list[dict], threshold: float = 0.0) -> dict:
    """Compute precision metrics at a given candidate score threshold."""
    groups = {
        "overall": {"ans": [], "nm": [], "un": [], "all_lat": []},
        "uz": {"ans": [], "nm": [], "un": [], "all_lat": []},
        "en": {"ans": [], "nm": [], "un": [], "all_lat": []},
    }

    for q in query_results:
        lang = q.get("lang", "uz")
        kind = q.get("kind", "answerable")
        expected_ids = q.get("expected_ids", [])
        latency = float(q.get("latency_ms", 0.0))

        top10 = q.get("top10", [])
        filtered_top10 = [
            r for r in top10
            if float(r.get("score") if r.get("score") is not None else r.get("fused_score", 0.0)) >= threshold
        ]

        target_groups = [groups["overall"]]
        if lang in groups:
            target_groups.append(groups[lang])

        for g in target_groups:
            g["all_lat"].append(latency)

        if kind == "answerable":
            first_hit_rank = None
            for idx, r in enumerate(filtered_top10):
                if is_hit(r, expected_ids):
                    first_hit_rank = idx + 1
                    break

            rec = {
                "hit@1": 1 if first_hit_rank == 1 else 0,
                "hit@5": 1 if first_hit_rank is not None and first_hit_rank <= 5 else 0,
                "hit@10": 1 if first_hit_rank is not None and first_hit_rank <= 10 else 0,
                "rr@10": (1.0 / first_hit_rank) if first_hit_rank is not None and first_hit_rank <= 10 else 0.0,
            }
            for g in target_groups:
                g["ans"].append(rec)
        elif kind == "near_miss":
            has_answer = 1 if len(filtered_top10) > 0 else 0
            for g in target_groups:
                g["nm"].append(has_answer)
        elif kind == "unanswerable":
            has_answer = 1 if len(filtered_top10) > 0 else 0
            for g in target_groups:
                g["un"].append(has_answer)

    out = {}
    for scope, data in groups.items():
        n_ans = len(data["ans"])
        recall1 = sum(r["hit@1"] for r in data["ans"]) / n_ans if n_ans > 0 else 0.0
        recall5 = sum(r["hit@5"] for r in data["ans"]) / n_ans if n_ans > 0 else 0.0
        recall10 = sum(r["hit@10"] for r in data["ans"]) / n_ans if n_ans > 0 else 0.0
        mrr10 = sum(r["rr@10"] for r in data["ans"]) / n_ans if n_ans > 0 else 0.0

        n_nm = len(data["nm"])
        far_nm = sum(data["nm"]) / n_nm if n_nm > 0 else 0.0

        n_un = len(data["un"])
        far_un = sum(data["un"]) / n_un if n_un > 0 else 0.0

        n_non_ans = n_nm + n_un
        far_total = (sum(data["nm"]) + sum(data["un"])) / n_non_ans if n_non_ans > 0 else 0.0

        latencies = data["all_lat"]
        p50 = compute_percentile(latencies, 50.0)
        p95 = compute_percentile(latencies, 95.0)

        out[scope] = {
            "num_answerable": n_ans,
            "recall@1": round(recall1, 4),
            "recall@5": round(recall5, 4),
            "recall@10": round(recall10, 4),
            "mrr@10": round(mrr10, 4),
            "num_near_miss": n_nm,
            "far_near_miss": round(far_nm, 4),
            "num_unanswerable": n_un,
            "far_unanswerable": round(far_un, 4),
            "num_non_answerable": n_non_ans,
            "false_answer_rate": round(far_total, 4),
            "p50_latency_ms": round(p50, 2),
            "p95_latency_ms": round(p95, 2),
        }

    return out


def find_optimal_threshold(query_results: list[dict], tolerance: float = 0.02) -> dict:
    """Find acceptance threshold keeping answerable Recall@10 within tolerance of best while minimising false answers."""
    observed_scores = set()
    for q in query_results:
        for r in q.get("top10", []):
            sc = float(r.get("score") if r.get("score") is not None else r.get("fused_score", 0.0))
            observed_scores.add(round(sc, 4))

    candidate_set = {0.0}
    for step in range(1, 20):
        candidate_set.add(round(step * 0.05, 4))

    if observed_scores:
        max_obs = max(observed_scores)
        if max_obs < 0.1:
            for i in range(1, 51):
                candidate_set.add(round(i * (max_obs / 50.0), 5))
        candidate_set.update(observed_scores)

    candidate_thresholds = sorted(candidate_set)

    threshold_evals = []
    for th in candidate_thresholds:
        m = compute_metrics_at_threshold(query_results, threshold=th)
        threshold_evals.append({
            "threshold": th,
            "recall@10": m["overall"]["recall@10"],
            "mrr@10": m["overall"]["mrr@10"],
            "false_answer_rate": m["overall"]["false_answer_rate"],
            "far_near_miss": m["overall"]["far_near_miss"],
            "far_unanswerable": m["overall"]["far_unanswerable"],
            "metrics": m,
        })

    best_recall10 = max(e["recall@10"] for e in threshold_evals) if threshold_evals else 0.0
    min_acceptable_recall = max(0.0, best_recall10 - tolerance)

    admissible = [e for e in threshold_evals if e["recall@10"] >= min_acceptable_recall]
    if not admissible:
        admissible = threshold_evals

    # Minimise false answer rate; tie-break on higher recall, then higher threshold
    chosen = min(
        admissible,
        key=lambda e: (e["false_answer_rate"], -e["recall@10"], -e["threshold"]),
    )

    opt_th = chosen["threshold"]
    key_steps = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, opt_th]
    sampled_table = []
    seen_th = set()
    for th in sorted(set(key_steps)):
        closest = min(threshold_evals, key=lambda e: abs(e["threshold"] - th))
        if closest["threshold"] not in seen_th:
            seen_th.add(closest["threshold"])
            sampled_table.append({
                "threshold": closest["threshold"],
                "recall@10": closest["recall@10"],
                "mrr@10": closest["mrr@10"],
                "false_answer_rate": closest["false_answer_rate"],
                "far_near_miss": closest["far_near_miss"],
                "far_unanswerable": closest["far_unanswerable"],
            })

    sampled_table.sort(key=lambda x: x["threshold"])

    return {
        "optimal_threshold": opt_th,
        "best_recall10": best_recall10,
        "recall10_at_threshold": chosen["recall@10"],
        "mrr10_at_threshold": chosen["mrr@10"],
        "far_at_threshold": chosen["false_answer_rate"],
        "far_near_miss_at_threshold": chosen["far_near_miss"],
        "far_unanswerable_at_threshold": chosen["far_unanswerable"],
        "metrics_at_optimal": chosen["metrics"],
        "threshold_table": sampled_table,
    }


def get_default_grid() -> list[dict]:
    """Generate small grid: rerank off/on, alpha 0.3/0.5/0.7, lexical 0.2/0.5, gate 0.8/0.5."""
    configs = []
    # Rerank off configs
    for lex in (0.2, 0.5):
        for gate in (0.8, 0.5):
            name = f"rerank_off_lex{lex}_gate{gate}"
            configs.append({
                "name": name,
                "rerank": False,
                "alpha": 0.5,
                "lexical_weight": lex,
                "gate": gate,
            })

    # Rerank on configs
    for alpha in (0.3, 0.5, 0.7):
        for lex in (0.2, 0.5):
            for gate in (0.8, 0.5):
                name = f"rerank_on_a{alpha}_lex{lex}_gate{gate}"
                configs.append({
                    "name": name,
                    "rerank": True,
                    "alpha": alpha,
                    "lexical_weight": lex,
                    "gate": gate,
                })

    return configs


def parse_grid_arg(grid_arg: Optional[str]) -> list[dict]:
    """Parse --grid argument into a list of config dicts."""
    all_configs = get_default_grid()
    if not grid_arg or grid_arg in ("all", "full", "small", "default"):
        return all_configs

    if grid_arg == "fast":
        return [
            {"name": "rerank_off_lex0.2_gate0.8", "rerank": False, "alpha": 0.5, "lexical_weight": 0.2, "gate": 0.8},
            {"name": "rerank_on_a0.3_lex0.2_gate0.8", "rerank": True, "alpha": 0.3, "lexical_weight": 0.2, "gate": 0.8},
            {"name": "rerank_on_a0.5_lex0.2_gate0.8", "rerank": True, "alpha": 0.5, "lexical_weight": 0.2, "gate": 0.8},
            {"name": "rerank_on_a0.7_lex0.2_gate0.8", "rerank": True, "alpha": 0.7, "lexical_weight": 0.2, "gate": 0.8},
        ]

    p = Path(grid_arg)
    if p.exists() and p.is_file():
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data

    if grid_arg.startswith("[") and grid_arg.endswith("]"):
        try:
            parsed = json.loads(grid_arg)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass

    selected_names = [n.strip() for n in grid_arg.split(",") if n.strip()]
    filtered = [c for c in all_configs if c["name"] in selected_names]
    return filtered if filtered else all_configs


def run_worker(db_path: str, queries_file: str, worker_out: str, config: dict) -> None:
    """Worker process: configure environment and model config, execute retrieve on queries."""
    os.environ["GRAPHIFY_STORAGE_MODE"] = "local"
    os.environ["GRAPHIFY_DB_PATH"] = str(Path(db_path).resolve())
    os.environ["GRAPHIFY_RERANK"] = "1" if config.get("rerank", True) else "0"
    os.environ["GRAPHIFY_RERANK_ALPHA"] = str(config.get("alpha", 0.5))
    os.environ["GRAPHIFY_LEXICAL_WEIGHT"] = str(config.get("lexical_weight", 0.2))
    os.environ["GRAPHIFY_MIN_COVERAGE"] = str(config.get("gate", 0.8))

    import embedders

    def patched_load(cfg=None):
        return {
            "embedder_type": "ollama",
            "config": {
                "ollama": {
                    "base_url": os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
                    "model": "bge-m3",
                }
            },
        }

    embedders.load_models_config = patched_load

    retrieval_errors = []
    orig_stderr = sys.stderr

    class StderrCatcher:
        def write(self, s):
            if any(k in s for k in [
                "[hybrid_search] Vector search error:",
                "[hybrid_search] Keyword search error:",
                "Vector search error:",
                "Keyword search error:",
            ]):
                retrieval_errors.append(s.strip())
            return orig_stderr.write(s)

        def flush(self):
            return orig_stderr.flush()

    sys.stderr = StderrCatcher()

    try:
        import db
        import keyword_index
        import search_core
        import search_service

        search_core.MIN_COVERAGE = float(config.get("gate", 0.8))
        search_core.LEXICAL_WEIGHT = float(config.get("lexical_weight", 0.2))
        search_core.RERANK_ALPHA = float(config.get("alpha", 0.5))

        col = db.get_collection()
        kw_idx = keyword_index.get_index()

        q_path = Path(queries_file)
        queries = []
        if q_path.suffix == ".jsonl":
            with open(q_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        queries.append(json.loads(line))
        else:
            with open(q_path, "r", encoding="utf-8") as f:
                queries = json.load(f)

        query_results = []
        for q in queries:
            query_text = q.get("query", "")
            t0 = time.perf_counter()
            res = search_service.retrieve(
                query=query_text,
                top_k=10,
                where=None,
                collection=col,
                kw_index=kw_idx,
                min_similarity=float(os.environ.get("GRAPHIFY_MIN_SIMILARITY", "0.10")),
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0

            if retrieval_errors:
                if any("Vector search error:" in err for err in retrieval_errors):
                    raise RuntimeError(f"Vector search xatosi aniqlandi: {retrieval_errors[0]}")
                raise RuntimeError(f"Qidiruv xatosi aniqlandi: {retrieval_errors[0]}")
            if res.get("status") == "error":
                raise RuntimeError(f"Qidiruv xatoligi: {res.get('message', 'unknown')}")

            res_items = res.get("results", [])
            top10_records = []
            for r in res_items[:10]:
                raw_score = r.get("score") if r.get("score") is not None else r.get("fused_score", 0.0)
                meta_dict = r.get("meta") or {}
                sibling_records = []
                for b in r.get("boshqa_boshqalar", []):
                    b_meta = b.get("meta") or {}
                    sibling_records.append({
                        "id": b.get("id") or b_meta.get("id") or "",
                        "meta": b_meta,
                    })

                top10_records.append({
                    "id": r.get("id") or meta_dict.get("id") or "",
                    "score": float(raw_score),
                    "fused_score": float(r.get("fused_score", 0.0)) if r.get("fused_score") is not None else None,
                    "rerank_score": float(r.get("rerank_score")) if r.get("rerank_score") is not None else None,
                    "meta": meta_dict,
                    "boshqa_boshqalar": sibling_records,
                })

            query_results.append({
                "id": q.get("id", ""),
                "lang": q.get("lang", "uz"),
                "kind": q.get("kind", "answerable"),
                "query": query_text,
                "expected_ids": q.get("expected_ids", []),
                "rerank": res.get("rerank", "off"),
                "rerank_reason": res.get("rerank_reason"),
                "latency_ms": round(latency_ms, 3),
                "top10": top10_records,
            })

        out_path = Path(worker_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_out = out_path.with_suffix(".tmp")
        payload = {
            "config": config,
            "query_results": query_results,
        }
        with open(tmp_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        tmp_out.replace(out_path)
    finally:
        sys.stderr = orig_stderr


def generate_markdown_report(report_data: dict, output_path: Path) -> str:
    """Generate REPORT.md in Uzbek prose containing measured results and recommendations."""
    meta = report_data.get("metadata", {})
    configs_data = report_data.get("configs", {})
    recommended_name = report_data.get("recommended_config", "")
    rec_cfg_info = configs_data.get(recommended_name, {})
    rec_cfg = rec_cfg_info.get("config", {})
    rec_opt = rec_cfg_info.get("threshold_optimization", {})
    rec_lat = rec_cfg_info.get("latency", {})

    table_rows = []
    for cfg_name, c_data in configs_data.items():
        cfg = c_data.get("config", {})
        b_m = c_data.get("metrics_baseline", {})
        opt = c_data.get("threshold_optimization", {})
        lat = c_data.get("latency", {})

        ov = b_m.get("overall", {})
        uz = b_m.get("uz", {})
        en = b_m.get("en", {})

        rerank_txt = "ON" if cfg.get("rerank") else "OFF"
        alpha_val = cfg.get("alpha", 0.5) if cfg.get("rerank") else "-"
        lex_val = cfg.get("lexical_weight", 0.2)
        gate_val = cfg.get("gate", 0.8)

        r1 = ov.get("recall@1", 0.0)
        r5 = ov.get("recall@5", 0.0)
        r10 = ov.get("recall@10", 0.0)
        mrr = ov.get("mrr@10", 0.0)

        uz_r10 = uz.get("recall@10", 0.0)
        uz_mrr = uz.get("mrr@10", 0.0)
        en_r10 = en.get("recall@10", 0.0)
        en_mrr = en.get("mrr@10", 0.0)

        far_nm = ov.get("far_near_miss", 0.0)
        far_un = ov.get("far_unanswerable", 0.0)
        far_tot = ov.get("false_answer_rate", 0.0)

        opt_th = opt.get("optimal_threshold", 0.0)
        opt_r10 = opt.get("recall10_at_threshold", 0.0)
        opt_far = opt.get("far_at_threshold", 0.0)

        p50 = lat.get("p50_latency_ms", 0.0)
        p95 = lat.get("p95_latency_ms", 0.0)

        row = (
            f"| `{cfg_name}` | {rerank_txt} | {alpha_val} | {lex_val} | {gate_val} | "
            f"{r1:.2f} | {r5:.2f} | {r10:.2f} | {mrr:.2f} | "
            f"{uz_r10:.2f} / {uz_mrr:.2f} | {en_r10:.2f} / {en_mrr:.2f} | "
            f"{far_nm:.2f} | {far_un:.2f} | {far_tot:.2f} | "
            f"`{opt_th:.4f}` | {opt_r10:.2f} | {opt_far:.2f} | "
            f"{p50:.1f} | {p95:.1f} |"
        )
        table_rows.append(row)

    verdict_uz = report_data.get("verdict_uz", "")
    verdict_en = report_data.get("verdict_en", "")

    opt_th_val = rec_opt.get("optimal_threshold", 0.0)
    opt_r10_val = rec_opt.get("recall10_at_threshold", 0.0)
    opt_far_val = rec_opt.get("far_at_threshold", 0.0)
    p50_val = rec_lat.get("p50_latency_ms", 0.0)
    p95_val = rec_lat.get("p95_latency_ms", 0.0)

    md = f"""# Graphify v2 Precision va Reranker Baholash Hisoboti

Ushbu hisobot Graphify v2 qidiruv tizimining aniqlik to'plami (precision set) bo'yicha turli konfiguratsiyalarda o'lchangan natijalari va qabul chegaralari asosida tuzilgan.

- **V2 Bazasi:** `{meta.get('v2_db', '')}`
- **So'rovlar to'plami:** `{meta.get('queries_file', '')}` ({meta.get('num_queries', 0)} ta so'rov: {meta.get('counts', {}).get('answerable', 0)} javobli, {meta.get('counts', {}).get('near_miss', 0)} near-miss, {meta.get('counts', {}).get('unanswerable', 0)} javobsiz)
- **O'tkazilgan vaqt:** {meta.get('timestamp', '')}
- **Konfiguratsiyalar soni:** {len(configs_data)}

---

## 📊 Konfiguratsiyalar Natijalari Jadvali

| Konfiguratsiya | Rerank | Alpha | Lex W | Gate | R@1 | R@5 | R@10 | MRR@10 | uz (R@10/MRR) | en (R@10/MRR) | FAR (NM) | FAR (UN) | Jami FAR | Chegara (θ*) | R@10(θ*) | FAR(θ*) | p50 (ms) | p95 (ms) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
""" + "\n".join(table_rows) + f"""

---

## 🏆 Tavsiya Etilgan Konfiguratsiya

O'lchangan ko'rsatkichlarga asoslanib tavsiya etiladigan optimal konfiguratsiya:

- **Konfiguratsiya:** `{recommended_name}`
- **Reranker holati:** `{"ON" if rec_cfg.get("rerank") else "OFF"}`
- **Reranker Alpha (α):** `{rec_cfg.get("alpha") if rec_cfg.get("rerank") else "N/A"}`
- **Leksik og'irlik (Lexical Weight):** `{rec_cfg.get("lexical_weight")}`
- **Gate (`MIN_COVERAGE`):** `{rec_cfg.get("gate")}`
- **Tavsiya etilgan qabul chegarasi (θ*):** `{opt_th_val:.4f}`
- **Chegaradagi Answerable Recall@10:** `{opt_r10_val:.4f}`
- **Chegaradagi False Answer Rate (FAR):** `{opt_far_val:.4f}`
- **Kechikish (Latency):** p50 = `{p50_val:.1f} ms`, p95 = `{p95_val:.1f} ms`

---

## 🌐 O'zbek Tili So'rovlari Bo'yicha Xulosa

{verdict_uz}

---

## 🇬🇧 Ingliz Tili So'rovlari Bo'yicha Xulosa

{verdict_en}
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md.strip() + "\n")

    return md


def compute_language_verdicts(configs_data: dict) -> Tuple[str, str, str]:
    """Compare best rerank ON config vs best rerank OFF config to generate language verdicts and recommendation."""
    rerank_off_configs = [c for c in configs_data.values() if not c["config"].get("rerank")]
    rerank_on_configs = [c for c in configs_data.values() if c["config"].get("rerank")]

    # Sort configs by: high recall10 at optimal threshold, high MRR, low FAR, low p95 latency
    def config_rank_key(c):
        opt = c.get("threshold_optimization", {})
        m = c.get("metrics_baseline", {}).get("overall", {})
        lat = c.get("latency", {})
        return (
            -opt.get("recall10_at_threshold", 0.0),
            -m.get("mrr@10", 0.0),
            opt.get("far_at_threshold", 0.0),
            lat.get("p95_latency_ms", 0.0),
        )

    all_sorted = sorted(configs_data.values(), key=config_rank_key)
    recommended_name = all_sorted[0]["config"]["name"] if all_sorted else ""

    best_off = sorted(rerank_off_configs, key=config_rank_key)[0] if rerank_off_configs else (all_sorted[0] if all_sorted else None)
    best_on = sorted(rerank_on_configs, key=config_rank_key)[0] if rerank_on_configs else None

    if not best_on or not best_off:
        verdict_uz = "Rerank ON va OFF konfiguratsiyalari to'liq mavjud emasligi sababli taqqoslash cheklandi."
        verdict_en = "Rerank ON va OFF konfiguratsiyalari to'liq mavjud emasligi sababli taqqoslash cheklandi."
        return verdict_uz, verdict_en, recommended_name

    uz_off_mrr = best_off["metrics_baseline"]["uz"]["mrr@10"]
    uz_off_r10 = best_off["metrics_baseline"]["uz"]["recall@10"]
    uz_on_mrr = best_on["metrics_baseline"]["uz"]["mrr@10"]
    uz_on_r10 = best_on["metrics_baseline"]["uz"]["recall@10"]

    en_off_mrr = best_off["metrics_baseline"]["en"]["mrr@10"]
    en_off_r10 = best_off["metrics_baseline"]["en"]["recall@10"]
    en_on_mrr = best_on["metrics_baseline"]["en"]["mrr@10"]
    en_on_r10 = best_on["metrics_baseline"]["en"]["recall@10"]

    if uz_on_mrr > uz_off_mrr or (uz_on_mrr == uz_off_mrr and uz_on_r10 > uz_off_r10):
        uz_impact = "ijobiy ta'sir ko'rsatadi (yordam beradi / helps)"
        uz_detail = f"O'zbekcha MRR@10 {uz_off_mrr:.4f} dan {uz_on_mrr:.4f} ga ko'tarildi (+{uz_on_mrr - uz_off_mrr:.4f}), Recall@10 esa {uz_off_r10:.4f} dan {uz_on_r10:.4f} ga o'zgardi."
    elif uz_on_mrr < uz_off_mrr or (uz_on_mrr == uz_off_mrr and uz_on_r10 < uz_off_r10):
        uz_impact = "salbiy ta'sir ko'rsatadi (zarar yetkazadi / hurts)"
        uz_detail = f"O'zbekcha MRR@10 {uz_off_mrr:.4f} dan {uz_on_mrr:.4f} ga tushdi ({uz_on_mrr - uz_off_mrr:.4f}), Recall@10 esa {uz_off_r10:.4f} dan {uz_on_r10:.4f} ga o'zgardi."
    else:
        uz_impact = "natijalarni o'zgartirmadi (neytral)"
        uz_detail = f"O'zbekcha MRR@10 ({uz_on_mrr:.4f}) va Recall@10 ({uz_on_r10:.4f}) o'zgarishsiz qoldi."

    verdict_uz = (
        f"O'lchangan natijalarga ko'ra, bge-reranker-v2-m3 modeli o'zbek tili so'rovlarida **{uz_impact}**. "
        f"{uz_detail} "
        f"Eng yaxshi Rerank ON (`{best_on['config']['name']}`) va Rerank OFF (`{best_off['config']['name']}`) taqqoslandi."
    )

    if en_on_mrr > en_off_mrr or (en_on_mrr == en_off_mrr and en_on_r10 > en_off_r10):
        en_impact = "ijobiy ta'sir ko'rsatadi (yordam beradi / helps)"
        en_detail = f"Inglizcha MRR@10 {en_off_mrr:.4f} dan {en_on_mrr:.4f} ga ko'tarildi (+{en_on_mrr - en_off_mrr:.4f}), Recall@10 esa {en_off_r10:.4f} dan {en_on_r10:.4f} ga o'zgardi."
    elif en_on_mrr < en_off_mrr or (en_on_mrr == en_off_mrr and en_on_r10 < en_off_r10):
        en_impact = "salbiy ta'sir ko'rsatadi (zarar yetkazadi / hurts)"
        en_detail = f"Inglizcha MRR@10 {en_off_mrr:.4f} dan {en_on_mrr:.4f} ga tushdi ({en_on_mrr - en_off_mrr:.4f}), Recall@10 esa {en_off_r10:.4f} dan {en_on_r10:.4f} ga o'zgardi."
    else:
        en_impact = "natijalarni o'zgartirmadi (neytral)"
        en_detail = f"Inglizcha MRR@10 ({en_on_mrr:.4f}) va Recall@10 ({en_on_r10:.4f}) o'zgarishsiz qoldi."

    verdict_en = (
        f"Ingliz tili so'rovlarida reranker **{en_impact}**. "
        f"{en_detail} "
        f"Eng yaxshi Rerank ON (`{best_on['config']['name']}`) va Rerank OFF (`{best_off['config']['name']}`) taqqoslandi."
    )

    return verdict_uz, verdict_en, recommended_name


def run_evaluation(
    v2_db: Path,
    queries_file: Path,
    out_dir: Path,
    grid: Optional[list[dict]] = None,
    worker_timeout: float = 600.0,
) -> dict:
    """Run full precision evaluation harness across grid configs with disposable replica and checkpointing."""
    if not v2_db.exists():
        raise FileNotFoundError(f"v2 ma'lumotlar bazasi topilmadi: {v2_db}")
    if not queries_file.exists():
        raise FileNotFoundError(f"So'rovlar fayli topilmadi: {queries_file}")

    out_dir = Path(out_dir).resolve()
    configs_dir = out_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    configs_to_run = grid if grid is not None else get_default_grid()

    # Read queries to verify counts
    queries = []
    if queries_file.suffix == ".jsonl":
        with open(queries_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    queries.append(json.loads(line))
    else:
        with open(queries_file, "r", encoding="utf-8") as f:
            queries = json.load(f)

    counts = {
        "answerable": sum(1 for q in queries if q.get("kind") == "answerable"),
        "near_miss": sum(1 for q in queries if q.get("kind") == "near_miss"),
        "unanswerable": sum(1 for q in queries if q.get("kind") == "unanswerable"),
        "uz": sum(1 for q in queries if q.get("lang") == "uz"),
        "en": sum(1 for q in queries if q.get("lang") == "en"),
    }

    # Determine which configs are already finished (resumability)
    pending_configs = []
    for cfg in configs_to_run:
        cfg_out_file = configs_dir / f"{cfg['name']}.json"
        is_finished = False
        if cfg_out_file.exists():
            try:
                with open(cfg_out_file, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                    if "query_results" in cached and len(cached["query_results"]) == len(queries):
                        is_finished = True
            except Exception:
                is_finished = False

        if not is_finished:
            pending_configs.append(cfg)

    # If some configs need execution, build disposable replica in temp dir
    if pending_configs:
        with tempfile.TemporaryDirectory(prefix="graphify_precision_replica_", ignore_cleanup_errors=True) as tmpdir:
            tmp_path = Path(tmpdir)
            v2_replica = tmp_path / "kg_db_v2_replica"
            shutil.copytree(v2_db, v2_replica)

            worker_env = os.environ.copy()
            worker_env["GRAPHIFY_STORAGE_MODE"] = "local"

            for cfg in pending_configs:
                cfg_out_file = configs_dir / f"{cfg['name']}.json"
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--db-path", str(v2_replica.resolve()),
                    "--queries-file", str(queries_file.resolve()),
                    "--worker-out", str(cfg_out_file.resolve()),
                    "--config-json", json.dumps(cfg, ensure_ascii=False),
                ]

                try:
                    proc = subprocess.run(
                        cmd,
                        env=worker_env,
                        cwd=str(PROJECT_ROOT),
                        stdin=subprocess.DEVNULL,
                        capture_output=True,
                        text=True,
                        timeout=worker_timeout,
                    )
                except subprocess.TimeoutExpired as e:
                    raise RuntimeError(f"Worker vaqt limiti ({worker_timeout}s) tugadi: {cfg['name']}") from e

                if proc.returncode != 0:
                    raise RuntimeError(
                        f"Worker xatosi ({cfg['name']}, kod {proc.returncode}):\n{proc.stderr}"
                    )

    # Process metrics for all configs (both cached and newly run)
    processed_configs = {}
    for cfg in configs_to_run:
        cfg_out_file = configs_dir / f"{cfg['name']}.json"
        if not cfg_out_file.exists():
            continue

        with open(cfg_out_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        q_res = data.get("query_results", [])
        m_base = compute_metrics_at_threshold(q_res, threshold=0.0)
        th_opt = find_optimal_threshold(q_res, tolerance=0.02)

        latencies = [float(q.get("latency_ms", 0.0)) for q in q_res]
        lat_info = {
            "p50_latency_ms": round(compute_percentile(latencies, 50.0), 2),
            "p95_latency_ms": round(compute_percentile(latencies, 95.0), 2),
        }

        processed_configs[cfg["name"]] = {
            "config": cfg,
            "metrics_baseline": m_base,
            "threshold_optimization": th_opt,
            "latency": lat_info,
        }

    verdict_uz, verdict_en, recommended_name = compute_language_verdicts(processed_configs)

    summary_payload = {
        "metadata": {
            "v2_db": str(v2_db),
            "queries_file": str(queries_file),
            "num_queries": len(queries),
            "counts": counts,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "recommended_config": recommended_name,
        "verdict_uz": verdict_uz,
        "verdict_en": verdict_en,
        "configs": processed_configs,
    }

    # Write precision_eval.json
    json_path = out_dir / "precision_eval.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    # Write REPORT.md
    md_path = out_dir / "REPORT.md"
    generate_markdown_report(summary_payload, md_path)

    return summary_payload


def main():
    parser = argparse.ArgumentParser(description="Evaluate Graphify v2 precision on labelled queries")
    parser.add_argument("--v2-db", default="kg_db_v2", help="Path to v2 database")
    parser.add_argument("--queries", default=None, help="Path to queries file (.jsonl or .json)")
    parser.add_argument("--out-dir", default="migrate_v2/precision_out", help="Output directory")
    parser.add_argument("--grid", default=None, help="Grid specification (small, fast, json, or names)")
    parser.add_argument("--worker-timeout", type=float, default=600.0, help="Worker subprocess timeout in seconds")

    # Hidden worker arguments
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--db-path", help=argparse.SUPPRESS)
    parser.add_argument("--queries-file", help=argparse.SUPPRESS)
    parser.add_argument("--worker-out", help=argparse.SUPPRESS)
    parser.add_argument("--config-json", help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args.worker:
        cfg = json.loads(args.config_json) if args.config_json else {}
        run_worker(
            db_path=args.db_path,
            queries_file=args.queries_file,
            worker_out=args.worker_out,
            config=cfg,
        )
        return

    if not args.queries:
        parser.error("--queries PATH argumenti talab qilinadi")

    grid_configs = parse_grid_arg(args.grid)
    run_evaluation(
        v2_db=Path(args.v2_db),
        queries_file=Path(args.queries),
        out_dir=Path(args.out_dir),
        grid=grid_configs,
        worker_timeout=args.worker_timeout,
    )


if __name__ == "__main__":
    main()
