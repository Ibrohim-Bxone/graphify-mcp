"""Tests for Graphify v2 precision evaluation harness (precision_eval.py).

Verifies metric math on a tiny fixture, threshold selection, report numbers integrity,
resumability, closed stdin, subprocess timeout, and worker execution with mocked retrieve.
No Ollama or live database required.
"""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from migrate_v2.precision_eval import (
    compute_language_verdicts,
    compute_metrics_at_threshold,
    compute_percentile,
    extract_item_ids,
    find_optimal_threshold,
    generate_markdown_report,
    get_default_grid,
    get_parent_prefix,
    is_hit,
    parse_grid_arg,
    run_evaluation,
    run_worker,
)


def test_get_parent_prefix_and_extract_item_ids():
    assert get_parent_prefix("ig:DYuu2v0AorV:0") == "ig:DYuu2v0AorV:"
    assert get_parent_prefix("ig:DYuu2v0AorV:12") == "ig:DYuu2v0AorV:"
    assert get_parent_prefix("ig:DYuu2v0AorV:") == "ig:DYuu2v0AorV:"
    assert get_parent_prefix("simple_id") == "simple_id:"
    assert get_parent_prefix("") == ""

    item = {
        "id": "ig:abc:0",
        "meta": {"id": "ig:abc:0", "parent_id": "ig:abc", "shortcode": "abc"},
    }
    ids = extract_item_ids(item)
    assert "ig:abc:0" in ids
    assert "ig:abc" in ids


def test_is_hit_matching_and_prefix():
    # 1. Exact match
    cand = {"id": "ig:post1:0", "meta": {"id": "ig:post1:0"}}
    assert is_hit(cand, ["ig:post1:0"])

    # 2. Parent prefix match (expected chunk 0, retrieved chunk 1 of same parent)
    cand2 = {"id": "ig:post1:1", "meta": {"id": "ig:post1:1"}}
    assert is_hit(cand2, ["ig:post1:0"])

    # 3. Expected parent id, retrieved chunk
    cand3 = {"id": "ig:post2:3", "meta": {"id": "ig:post2:3"}}
    assert is_hit(cand3, ["ig:post2"])

    # 4. Sibling match in boshqa_boshqalar
    cand4 = {
        "id": "other:0",
        "meta": {"id": "other:0"},
        "boshqa_boshqalar": [
            {"id": "ig:post3:1", "meta": {"id": "ig:post3:1"}}
        ],
    }
    assert is_hit(cand4, ["ig:post3:0"])

    # 5. Non-matching id
    cand5 = {"id": "ig:diff:0", "meta": {"id": "ig:diff:0"}}
    assert not is_hit(cand5, ["ig:post1:0", "ig:post2"])

    # 6. Empty expected ids
    assert not is_hit(cand, [])


def test_percentile_calculation():
    vals = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert compute_percentile(vals, 50.0) == 30.0
    assert compute_percentile(vals, 0.0) == 10.0
    assert compute_percentile(vals, 100.0) == 50.0
    assert compute_percentile([], 50.0) == 0.0
    assert compute_percentile([42.0], 50.0) == 42.0


def test_metric_math_on_tiny_fixture():
    # Tiny fixture: 4 answerable (2 uz, 2 en), 2 near_miss (1 uz, 1 en), 2 unanswerable (1 uz, 1 en)
    query_results = [
        # Answerable 1 (uz): hit at rank 1, score 0.90
        {
            "id": "q1",
            "lang": "uz",
            "kind": "answerable",
            "expected_ids": ["ig:post1:0"],
            "latency_ms": 10.0,
            "top10": [{"id": "ig:post1:0", "score": 0.90}],
        },
        # Answerable 2 (uz): hit at rank 3, score 0.80
        {
            "id": "q2",
            "lang": "uz",
            "kind": "answerable",
            "expected_ids": ["ig:post2:0"],
            "latency_ms": 20.0,
            "top10": [
                {"id": "ig:wrong:1", "score": 0.85},
                {"id": "ig:wrong:2", "score": 0.82},
                {"id": "ig:post2:1", "score": 0.80},  # hit by parent prefix
            ],
        },
        # Answerable 3 (en): hit at rank 6, score 0.70
        {
            "id": "q3",
            "lang": "en",
            "kind": "answerable",
            "expected_ids": ["ig:post3:0"],
            "latency_ms": 30.0,
            "top10": [
                {"id": f"ig:wrong:{i}", "score": 0.75 - i * 0.01}
                for i in range(5)
            ] + [{"id": "ig:post3:0", "score": 0.70}],
        },
        # Answerable 4 (en): no hit in top 10
        {
            "id": "q4",
            "lang": "en",
            "kind": "answerable",
            "expected_ids": ["ig:post4:0"],
            "latency_ms": 40.0,
            "top10": [{"id": "ig:wrong:99", "score": 0.60}],
        },
        # Near-miss 1 (uz): returns result with score 0.75 (false answer)
        {
            "id": "q5",
            "lang": "uz",
            "kind": "near_miss",
            "expected_ids": [],
            "latency_ms": 15.0,
            "top10": [{"id": "ig:nm:1", "score": 0.75}],
        },
        # Near-miss 2 (en): returns no results (correctly rejected)
        {
            "id": "q6",
            "lang": "en",
            "kind": "near_miss",
            "expected_ids": [],
            "latency_ms": 25.0,
            "top10": [],
        },
        # Unanswerable 1 (uz): returns result with score 0.65 (false answer)
        {
            "id": "q7",
            "lang": "uz",
            "kind": "unanswerable",
            "expected_ids": [],
            "latency_ms": 35.0,
            "top10": [{"id": "ig:un:1", "score": 0.65}],
        },
        # Unanswerable 2 (en): returns result with score 0.55 (false answer)
        {
            "id": "q8",
            "lang": "en",
            "kind": "unanswerable",
            "expected_ids": [],
            "latency_ms": 45.0,
            "top10": [{"id": "ig:un:2", "score": 0.55}],
        },
    ]

    m = compute_metrics_at_threshold(query_results, threshold=0.0)

    # Answerable: Q1 (rank 1), Q2 (rank 3), Q3 (rank 6), Q4 (no hit)
    # Recall@1 = 1 / 4 = 0.25
    # Recall@5 = 2 / 4 = 0.50
    # Recall@10 = 3 / 4 = 0.75
    # MRR@10 = (1/1 + 1/3 + 1/6 + 0) / 4 = (1.5) / 4 = 0.375
    assert m["overall"]["recall@1"] == 0.25
    assert m["overall"]["recall@5"] == 0.50
    assert m["overall"]["recall@10"] == 0.75
    assert m["overall"]["mrr@10"] == 0.375

    # Near-miss: 1 false answer out of 2 -> 0.5
    assert m["overall"]["far_near_miss"] == 0.5
    # Unanswerable: 2 false answers out of 2 -> 1.0
    assert m["overall"]["far_unanswerable"] == 1.0
    # Total non-answerable: 3 false answers out of 4 -> 0.75
    assert m["overall"]["false_answer_rate"] == 0.75

    # Per-language:
    # Uzbek answerable: Q1 (rank 1), Q2 (rank 3) -> Recall@10 = 2/2 = 1.0, MRR = (1 + 1/3)/2 = 0.6667
    assert m["uz"]["recall@10"] == 1.0
    assert m["uz"]["mrr@10"] == pytest.approx(0.6667, abs=1e-4)

    # English answerable: Q3 (rank 6), Q4 (no hit) -> Recall@10 = 1/2 = 0.5, MRR = (1/6)/2 = 0.0833
    assert m["en"]["recall@10"] == 0.5
    assert m["en"]["mrr@10"] == pytest.approx(0.0833, abs=1e-4)


def test_threshold_selection_keeps_recall_and_minimises_false_answers():
    # 4 answerable queries with high hit scores (0.80 - 0.95)
    # 4 non-answerable queries with false answers at scores 0.50 - 0.75
    query_results = [
        {
            "id": f"ans_{i}",
            "lang": "uz" if i % 2 == 0 else "en",
            "kind": "answerable",
            "expected_ids": [f"doc_{i}"],
            "latency_ms": 10.0,
            "top10": [{"id": f"doc_{i}", "score": 0.80 + i * 0.05}],
        }
        for i in range(4)
    ]
    # Non-answerable queries (near_miss and unanswerable)
    query_results += [
        {
            "id": "nm_1",
            "lang": "uz",
            "kind": "near_miss",
            "expected_ids": [],
            "latency_ms": 10.0,
            "top10": [{"id": "noise_1", "score": 0.75}],
        },
        {
            "id": "nm_2",
            "lang": "en",
            "kind": "near_miss",
            "expected_ids": [],
            "latency_ms": 10.0,
            "top10": [{"id": "noise_2", "score": 0.70}],
        },
        {
            "id": "un_1",
            "lang": "uz",
            "kind": "unanswerable",
            "expected_ids": [],
            "latency_ms": 10.0,
            "top10": [{"id": "noise_3", "score": 0.65}],
        },
        {
            "id": "un_2",
            "lang": "en",
            "kind": "unanswerable",
            "expected_ids": [],
            "latency_ms": 10.0,
            "top10": [{"id": "noise_4", "score": 0.50}],
        },
    ]

    opt = find_optimal_threshold(query_results, tolerance=0.02)

    # At threshold ~0.78:
    # All 4 answerable have scores >= 0.80, so Recall@10 is 1.0 (exact match with best 1.0).
    # All 4 non-answerable have scores <= 0.75, so FAR is 0.0!
    assert opt["best_recall10"] == 1.0
    assert opt["recall10_at_threshold"] == 1.0
    assert opt["far_at_threshold"] == 0.0
    assert opt["optimal_threshold"] >= 0.75
    assert len(opt["threshold_table"]) > 0


def test_report_generation_contains_only_computed_numbers(tmp_path):
    report_data = {
        "metadata": {
            "v2_db": "test_v2_db",
            "queries_file": "test_queries.jsonl",
            "num_queries": 8,
            "counts": {"answerable": 4, "near_miss": 2, "unanswerable": 2},
            "timestamp": "2026-10-29T12:00:00Z",
        },
        "recommended_config": "rerank_on_a0.5_lex0.2_gate0.8",
        "verdict_uz": "O'lchangan natijalarga ko'ra, bge-reranker-v2-m3 modeli o'zbek tili so'rovlarida **ijobiy ta'sir ko'rsatadi (yordam beradi / helps)**.",
        "verdict_en": "Ingliz tili so'rovlarida reranker **ijobiy ta'sir ko'rsatadi (yordam beradi / helps)**.",
        "configs": {
            "rerank_off_lex0.2_gate0.8": {
                "config": {"name": "rerank_off_lex0.2_gate0.8", "rerank": False, "alpha": 0.5, "lexical_weight": 0.2, "gate": 0.8},
                "metrics_baseline": {
                    "overall": {"recall@1": 0.25, "recall@5": 0.50, "recall@10": 0.75, "mrr@10": 0.35, "far_near_miss": 0.5, "far_unanswerable": 1.0, "false_answer_rate": 0.75},
                    "uz": {"recall@10": 0.80, "mrr@10": 0.40},
                    "en": {"recall@10": 0.70, "mrr@10": 0.30},
                },
                "threshold_optimization": {"optimal_threshold": 0.02, "recall10_at_threshold": 0.75, "far_at_threshold": 0.25},
                "latency": {"p50_latency_ms": 12.3, "p95_latency_ms": 25.4},
            },
            "rerank_on_a0.5_lex0.2_gate0.8": {
                "config": {"name": "rerank_on_a0.5_lex0.2_gate0.8", "rerank": True, "alpha": 0.5, "lexical_weight": 0.2, "gate": 0.8},
                "metrics_baseline": {
                    "overall": {"recall@1": 0.50, "recall@5": 0.75, "recall@10": 1.0, "mrr@10": 0.70, "far_near_miss": 0.0, "far_unanswerable": 0.0, "false_answer_rate": 0.0},
                    "uz": {"recall@10": 1.0, "mrr@10": 0.75},
                    "en": {"recall@10": 1.0, "mrr@10": 0.65},
                },
                "threshold_optimization": {"optimal_threshold": 0.78, "recall10_at_threshold": 1.0, "far_at_threshold": 0.0},
                "latency": {"p50_latency_ms": 18.5, "p95_latency_ms": 32.1},
            },
        },
    }

    out_file = tmp_path / "REPORT.md"
    generate_markdown_report(report_data, out_file)

    content = out_file.read_text(encoding="utf-8")
    # Verify measured numbers exist in the markdown
    assert "rerank_on_a0.5_lex0.2_gate0.8" in content
    assert "rerank_off_lex0.2_gate0.8" in content
    assert "0.7800" in content
    assert "18.5" in content
    assert "32.1" in content
    assert "ijobiy ta'sir ko'rsatadi (yordam beradi / helps)" in content


def test_compute_language_verdicts_positive_and_negative():
    # Case 1: Rerank helps Uzbek
    cfg_data_helps = {
        "off": {
            "config": {"name": "off", "rerank": False},
            "metrics_baseline": {"uz": {"mrr@10": 0.40, "recall@10": 0.70}, "en": {"mrr@10": 0.35, "recall@10": 0.65}, "overall": {"mrr@10": 0.375}},
            "threshold_optimization": {"recall10_at_threshold": 0.70, "far_at_threshold": 0.30},
            "latency": {"p95_latency_ms": 10.0},
        },
        "on": {
            "config": {"name": "on", "rerank": True},
            "metrics_baseline": {"uz": {"mrr@10": 0.60, "recall@10": 0.85}, "en": {"mrr@10": 0.55, "recall@10": 0.80}, "overall": {"mrr@10": 0.575}},
            "threshold_optimization": {"recall10_at_threshold": 0.85, "far_at_threshold": 0.05},
            "latency": {"p95_latency_ms": 20.0},
        },
    }
    uz_v, en_v, rec = compute_language_verdicts(cfg_data_helps)
    assert "yordam beradi" in uz_v
    assert rec == "on"

    # Case 2: Rerank hurts Uzbek
    cfg_data_hurts = {
        "off": {
            "config": {"name": "off", "rerank": False},
            "metrics_baseline": {"uz": {"mrr@10": 0.60, "recall@10": 0.85}, "en": {"mrr@10": 0.55, "recall@10": 0.80}, "overall": {"mrr@10": 0.575}},
            "threshold_optimization": {"recall10_at_threshold": 0.85, "far_at_threshold": 0.05},
            "latency": {"p95_latency_ms": 10.0},
        },
        "on": {
            "config": {"name": "on", "rerank": True},
            "metrics_baseline": {"uz": {"mrr@10": 0.40, "recall@10": 0.70}, "en": {"mrr@10": 0.35, "recall@10": 0.65}, "overall": {"mrr@10": 0.375}},
            "threshold_optimization": {"recall10_at_threshold": 0.70, "far_at_threshold": 0.30},
            "latency": {"p95_latency_ms": 20.0},
        },
    }
    uz_v2, en_v2, rec2 = compute_language_verdicts(cfg_data_hurts)
    assert "zarar yetkazadi" in uz_v2
    assert rec2 == "off"


def test_grid_parsing():
    default_grid = get_default_grid()
    assert len(default_grid) == 16

    fast_grid = parse_grid_arg("fast")
    assert len(fast_grid) == 4

    single_cfg = parse_grid_arg("rerank_off_lex0.2_gate0.8")
    assert len(single_cfg) == 1
    assert single_cfg[0]["name"] == "rerank_off_lex0.2_gate0.8"


def test_resumability_and_subprocess_closed_stdin(tmp_path):
    # Setup mock files
    fake_db = tmp_path / "fake_db"
    fake_db.mkdir()
    (fake_db / "chroma.sqlite3").write_text("fake db", encoding="utf-8")

    queries_file = tmp_path / "queries.jsonl"
    q_data = [
        {"id": "q1", "lang": "uz", "kind": "answerable", "query": "test 1", "expected_ids": ["doc1"]},
        {"id": "q2", "lang": "en", "kind": "near_miss", "query": "test 2", "expected_ids": []},
    ]
    with open(queries_file, "w", encoding="utf-8") as f:
        for q in q_data:
            f.write(json.dumps(q) + "\n")

    out_dir = tmp_path / "out"
    configs_dir = out_dir / "configs"
    configs_dir.mkdir(parents=True)

    # Pre-populate config 1 as completed
    cfg1 = {"name": "cfg1", "rerank": False, "alpha": 0.5, "lexical_weight": 0.2, "gate": 0.8}
    cfg2 = {"name": "cfg2", "rerank": True, "alpha": 0.5, "lexical_weight": 0.2, "gate": 0.8}

    cached_results = {
        "config": cfg1,
        "query_results": [
            {"id": "q1", "lang": "uz", "kind": "answerable", "expected_ids": ["doc1"], "latency_ms": 5.0, "top10": [{"id": "doc1", "score": 0.9}]},
            {"id": "q2", "lang": "en", "kind": "near_miss", "expected_ids": [], "latency_ms": 6.0, "top10": []},
        ],
    }
    with open(configs_dir / "cfg1.json", "w", encoding="utf-8") as f:
        json.dump(cached_results, f)

    def mock_subprocess_run(cmd, **kwargs):
        # Verify closed stdin and timeout
        assert kwargs.get("stdin") == subprocess.DEVNULL
        assert kwargs.get("timeout") == 300.0

        # Worker output file is in cmd
        worker_out_idx = cmd.index("--worker-out") + 1
        worker_out_path = Path(cmd[worker_out_idx])
        # Simulate worker output
        dummy_out = {
            "config": cfg2,
            "query_results": [
                {"id": "q1", "lang": "uz", "kind": "answerable", "expected_ids": ["doc1"], "latency_ms": 8.0, "top10": [{"id": "doc1", "score": 0.95}]},
                {"id": "q2", "lang": "en", "kind": "near_miss", "expected_ids": [], "latency_ms": 7.0, "top10": []},
            ],
        }
        with open(worker_out_path, "w", encoding="utf-8") as f:
            json.dump(dummy_out, f)

        res = Mock()
        res.returncode = 0
        res.stderr = ""
        return res

    with patch("subprocess.run", side_effect=mock_subprocess_run) as mocked_run:
        summary = run_evaluation(
            v2_db=fake_db,
            queries_file=queries_file,
            out_dir=out_dir,
            grid=[cfg1, cfg2],
            worker_timeout=300.0,
        )

        # Assert subprocess.run was called ONLY ONCE (for cfg2, skipping cached cfg1)
        assert mocked_run.call_count == 1
        assert "cfg1" in summary["configs"]
        assert "cfg2" in summary["configs"]
        assert (out_dir / "precision_eval.json").exists()
        assert (out_dir / "REPORT.md").exists()


def test_run_worker_with_mocked_retrieve(tmp_path):
    fake_db = tmp_path / "fake_db"
    fake_db.mkdir()
    queries_file = tmp_path / "queries.jsonl"
    q_data = [
        {"id": "q1", "lang": "uz", "kind": "answerable", "query": "qidiruv", "expected_ids": ["doc_test"]},
    ]
    with open(queries_file, "w", encoding="utf-8") as f:
        for q in q_data:
            f.write(json.dumps(q) + "\n")

    worker_out = tmp_path / "worker_out.json"
    cfg = {"name": "test_cfg", "rerank": True, "alpha": 0.5, "lexical_weight": 0.2, "gate": 0.8}

    mock_chroma = MagicMock()
    mock_chroma.__path__ = []
    chroma_modules = {
        "chromadb": mock_chroma,
        "chromadb.api": MagicMock(),
        "chromadb.api.client": MagicMock(),
    }

    mock_db = Mock()
    mock_kw = Mock()
    mock_retrieve_res = {
        "status": "ok",
        "results": [
            {
                "id": "doc_test",
                "score": 0.92,
                "fused_score": 0.015,
                "rerank_score": 0.98,
                "meta": {"id": "doc_test"},
                "boshqa_boshqalar": [],
            }
        ],
        "rerank": "on",
        "rerank_reason": None,
    }

    with patch.dict(sys.modules, chroma_modules), \
         patch("db.get_collection", return_value=mock_db), \
         patch("keyword_index.get_index", return_value=mock_kw), \
         patch("search_service.retrieve", return_value=mock_retrieve_res):
        run_worker(
            db_path=str(fake_db),
            queries_file=str(queries_file),
            worker_out=str(worker_out),
            config=cfg,
        )

    assert worker_out.exists()
    with open(worker_out, "r", encoding="utf-8") as f:
        saved = json.load(f)

    assert len(saved["query_results"]) == 1
    rec = saved["query_results"][0]
    assert rec["id"] == "q1"
    assert rec["top10"][0]["id"] == "doc_test"
    assert rec["top10"][0]["score"] == 0.92
    assert rec["rerank"] == "on"


def test_run_evaluation_missing_db_or_queries_raises(tmp_path):
    missing_db = tmp_path / "nonexistent_db"
    missing_queries = tmp_path / "nonexistent_queries.jsonl"
    fake_db = tmp_path / "real_db"
    fake_db.mkdir()
    fake_queries = tmp_path / "queries.jsonl"
    fake_queries.write_text("{}\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="v2 ma'lumotlar bazasi topilmadi"):
        run_evaluation(v2_db=missing_db, queries_file=fake_queries, out_dir=tmp_path / "out")

    with pytest.raises(FileNotFoundError, match="So'rovlar fayli topilmadi"):
        run_evaluation(v2_db=fake_db, queries_file=missing_queries, out_dir=tmp_path / "out")


def test_subprocess_timeout_raises_runtime_error(tmp_path):
    fake_db = tmp_path / "db"
    fake_db.mkdir()
    queries_file = tmp_path / "queries.jsonl"
    queries_file.write_text('{"id":"1","query":"q","kind":"answerable"}\n', encoding="utf-8")

    def raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="worker", timeout=1.0)

    cfg = {"name": "test_cfg", "rerank": False, "alpha": 0.5, "lexical_weight": 0.2, "gate": 0.8}
    with patch("subprocess.run", side_effect=raise_timeout):
        with pytest.raises(RuntimeError, match="Worker vaqt limiti"):
            run_evaluation(
                v2_db=fake_db,
                queries_file=queries_file,
                out_dir=tmp_path / "out",
                grid=[cfg],
                worker_timeout=1.0,
            )


def test_subprocess_non_zero_exit_raises_runtime_error(tmp_path):
    fake_db = tmp_path / "db"
    fake_db.mkdir()
    queries_file = tmp_path / "queries.jsonl"
    queries_file.write_text('{"id":"1","query":"q","kind":"answerable"}\n', encoding="utf-8")

    mock_res = Mock()
    mock_res.returncode = 1
    mock_res.stderr = "Fatal error in worker"

    cfg = {"name": "test_cfg", "rerank": False, "alpha": 0.5, "lexical_weight": 0.2, "gate": 0.8}
    with patch("subprocess.run", return_value=mock_res):
        with pytest.raises(RuntimeError, match="Worker xatosi"):
            run_evaluation(
                v2_db=fake_db,
                queries_file=queries_file,
                out_dir=tmp_path / "out",
                grid=[cfg],
            )

