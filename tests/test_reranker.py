import json
import os
import tempfile
import threading
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
import pytest

import reranker
from reranker import Reranker, get_reranker, get_reranker_with_reason, reset_reranker, score_pairs

pytestmark = pytest.mark.reranker


@pytest.fixture(autouse=True)
def clean_reranker_state(monkeypatch):
    reset_reranker()
    monkeypatch.delenv("GRAPHIFY_RERANKER_MODEL", raising=False)
    monkeypatch.delenv("GRAPHIFY_RERANK_BUDGET_S", raising=False)
    monkeypatch.delenv("GRAPHIFY_RERANK", raising=False)
    monkeypatch.delenv("GRAPHIFY_RERANK_DEVICE", raising=False)
    monkeypatch.delenv("GRAPHIFY_RERANK_THREADS", raising=False)
    yield
    reset_reranker()


def test_huggingface_hub_downloads_are_blocked_and_never_reached():
    import huggingface_hub

    def forbidden_download(*args, **kwargs):
        raise AssertionError("huggingface_hub download network function was called!")

    with patch.object(huggingface_hub, "snapshot_download", side_effect=forbidden_download):
        with patch.object(huggingface_hub, "hf_hub_download", side_effect=forbidden_download):
            with patch("huggingface_hub.try_to_load_from_cache", return_value=None):
                inst, reason = get_reranker_with_reason()
                assert inst is None
                assert "model missing" in reason


def test_missing_model_returns_none_and_reports_reason():
    with patch("huggingface_hub.try_to_load_from_cache", return_value=None):
        inst = get_reranker()
        assert inst is None
        inst2, reason = get_reranker_with_reason()
        assert inst2 is None
        assert "model missing" in reason


def test_thread_safe_singleton():
    instances = []

    mock_session = Mock()
    mock_tok = Mock()

    def create_mock_reranker(*args, **kwargs):
        return Reranker(mock_session, mock_tok, pad_id=1, max_length=512), None

    with patch("reranker._load_reranker", side_effect=create_mock_reranker):
        def worker():
            inst = get_reranker()
            instances.append(inst)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(instances) == 10
        first = instances[0]
        assert first is not None
        for inst in instances[1:]:
            assert inst is first


def test_tokenizer_pad_id_and_max_length_read_from_files(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPHIFY_RERANK_DEVICE", "cpu")  # real CUDA on the host must not reach the mocked session
    tok_json = tmp_path / "tokenizer.json"
    tok_cfg = tmp_path / "tokenizer_config.json"
    model_file = tmp_path / "model.onnx"
    model_file.write_bytes(b"dummy onnx")

    tok_cfg_data = {
        "pad_token_id": 99,
        "max_length": 512,
    }
    tok_cfg.write_text(json.dumps(tok_cfg_data), encoding="utf-8")

    tok_json_data = {
        "version": "1.0",
        "truncation": {"max_length": 512},
        "padding": {"pad_id": 99},
    }
    tok_json.write_text(json.dumps(tok_json_data), encoding="utf-8")

    monkeypatch.setenv("GRAPHIFY_RERANKER_MODEL", str(model_file))

    mock_tokenizer_instance = Mock()
    mock_session_instance = Mock()

    with patch("tokenizers.Tokenizer.from_file", return_value=mock_tokenizer_instance):
        with patch("onnxruntime.InferenceSession", return_value=mock_session_instance):
            inst, err = get_reranker_with_reason()
            assert err is None
            assert inst is not None
            assert inst.pad_id == 99
            assert inst.max_length == 512
            mock_tokenizer_instance.enable_truncation.assert_called_with(max_length=512, strategy="only_second")
            mock_tokenizer_instance.enable_padding.assert_called()


def test_batch_scoring_logits_and_sigmoid():
    mock_session = Mock()
    mock_tok = Mock()

    # Create dummy encodings
    class DummyEnc:
        def __init__(self):
            self.ids = [10, 20, 1]
            self.attention_mask = [1, 1, 0]

    mock_tok.encode_batch.side_effect = lambda batch: [DummyEnc() for _ in batch]
    # Logits: 0.0 -> sigmoid = 0.5; 2.197 -> sigmoid ~ 0.9
    mock_session.run.return_value = [np.array([[0.0], [2.1972245773362196]])]

    r = Reranker(session=mock_session, tokenizer=mock_tok, pad_id=1, max_length=512)
    scores, err = r.score_pairs([("q", "d1"), ("q", "d2")])
    assert err is None
    assert scores is not None
    assert len(scores) == 2
    assert np.isclose(scores[0], 0.5, atol=1e-3)
    assert np.isclose(scores[1], 0.9, atol=1e-3)


def test_time_budget_exceeded(monkeypatch):
    mock_session = Mock()
    mock_tok = Mock()

    class DummyEnc:
        def __init__(self):
            self.ids = [1, 1]
            self.attention_mask = [1, 1]

    mock_tok.encode_batch.side_effect = lambda batch: [DummyEnc() for _ in batch]
    mock_session.run.return_value = [np.array([[0.0]])]

    monkeypatch.setenv("GRAPHIFY_RERANK_BUDGET_S", "0.0")

    r = Reranker(session=mock_session, tokenizer=mock_tok, pad_id=1, max_length=512)
    scores, err = r.score_pairs([("q", "doc")])
    assert scores is None
    assert err == "budget exceeded"
    assert r.last_error == "budget exceeded"


def test_load_error_handling(tmp_path, monkeypatch):
    model_file = tmp_path / "model.onnx"
    model_file.write_bytes(b"corrupt")
    tok_json = tmp_path / "tokenizer.json"
    tok_json.write_text("{}", encoding="utf-8")

    monkeypatch.setenv("GRAPHIFY_RERANKER_MODEL", str(model_file))

    with patch("tokenizers.Tokenizer.from_file", side_effect=Exception("Corrupt tokenizer")):
        inst, err = get_reranker_with_reason()
        assert inst is None
        assert "load error" in err


def test_tokenizer_model_max_length_8192_capped_to_512(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPHIFY_RERANK_DEVICE", "cpu")  # real CUDA on the host must not reach the mocked session
    tok_json = tmp_path / "tokenizer.json"
    tok_cfg = tmp_path / "tokenizer_config.json"
    model_file = tmp_path / "model.onnx"
    model_file.write_bytes(b"dummy onnx")

    tok_cfg_data = {
        "pad_token_id": 99,
        "model_max_length": 8192,
    }
    tok_cfg.write_text(json.dumps(tok_cfg_data), encoding="utf-8")

    tok_json_data = {
        "version": "1.0",
        "truncation": None,
        "padding": {"pad_id": 99},
    }
    tok_json.write_text(json.dumps(tok_json_data), encoding="utf-8")

    monkeypatch.setenv("GRAPHIFY_RERANKER_MODEL", str(model_file))

    mock_tokenizer_instance = Mock()
    mock_session_instance = Mock()

    with patch("tokenizers.Tokenizer.from_file", return_value=mock_tokenizer_instance):
        with patch("onnxruntime.InferenceSession", return_value=mock_session_instance):
            inst, err = get_reranker_with_reason()
            assert err is None
            assert inst is not None
            assert inst.max_length == 512
            assert inst.model_max_length == 8192
            mock_tokenizer_instance.enable_truncation.assert_called_with(max_length=512, strategy="only_second")


def test_inference_timeout_when_session_run_is_slow():
    import time
    mock_session = Mock()
    mock_tok = Mock()

    class DummyEnc:
        def __init__(self):
            self.ids = [1, 2]
            self.attention_mask = [1, 1]

    mock_tok.encode_batch.side_effect = lambda batch: [DummyEnc() for _ in batch]

    def slow_run(*args, **kwargs):
        time.sleep(0.15)
        return [np.array([[0.0]])]

    mock_session.run.side_effect = slow_run

    r = Reranker(session=mock_session, tokenizer=mock_tok, pad_id=1, max_length=512)
    t0 = time.monotonic()
    scores, err = r.score_pairs([("q", "doc")], budget_s=0.02)
    elapsed = time.monotonic() - t0

    assert scores is None
    assert err == "budget exceeded"
    assert r.last_error == "budget exceeded"
    assert elapsed < 0.10


def test_negative_or_zero_budget_rejected_immediately():
    mock_session = Mock()
    mock_tok = Mock()
    r = Reranker(session=mock_session, tokenizer=mock_tok, pad_id=1, max_length=512)

    for bad_budget in [0.0, -0.5, -5.0]:
        scores, err = r.score_pairs([("q", "doc")], budget_s=bad_budget)
        assert scores is None
        assert err == "budget exceeded"
        assert r.last_error == "budget exceeded"
        mock_session.run.assert_not_called()


def test_finding1_truncation_only_second_and_oversized_query_unavailable():
    """Finding 1 regression: tokenizer uses strategy='only_second' and oversized queries report unavailable."""
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import Whitespace

    tok = Tokenizer(BPE())
    tok.pre_tokenizer = Whitespace()
    tok.add_tokens(["query", "long", "term", "doc", "word", "[PAD]"])
    tok.enable_truncation(max_length=3, strategy="only_second")
    tok.enable_padding(pad_id=5, pad_token="[PAD]")

    mock_session = Mock()
    r = Reranker(session=mock_session, tokenizer=tok, pad_id=5, max_length=3)

    mock_session.run.return_value = [np.array([[0.0]])]
    scores, err = r.score_pairs([("query", "doc word long term")])
    assert err is None
    assert scores is not None

    scores, err = r.score_pairs([("query long term word", "doc")])
    assert scores is None
    assert "query exceeds max length" in err


def test_finding2_max_one_inflight_inference_semaphore_busy():
    """Finding 2 regression: at most one in-flight inference (semaphore); busy calls return unavailable immediately."""
    import time
    mock_session = Mock()
    mock_tok = Mock()

    class DummyEnc:
        def __init__(self):
            self.ids = [1, 2]
            self.attention_mask = [1, 1]

    mock_tok.encode_batch.side_effect = lambda batch: [DummyEnc() for _ in batch]

    started_evt = threading.Event()
    finish_evt = threading.Event()

    def slow_run(*args, **kwargs):
        started_evt.set()
        finish_evt.wait(timeout=2.0)
        return [np.array([[0.0]])]

    mock_session.run.side_effect = slow_run
    r = Reranker(session=mock_session, tokenizer=mock_tok, pad_id=1, max_length=512)

    res_holder = []

    def t1():
        res = r.score_pairs([("q", "d1")], budget_s=1.0)
        res_holder.append(res)

    thread1 = threading.Thread(target=t1)
    thread1.start()
    started_evt.wait(timeout=1.0)

    scores, err = r.score_pairs([("q", "d2")], budget_s=1.0)
    assert scores is None
    assert "unavailable" in err or "busy" in err

    finish_evt.set()
    thread1.join(timeout=1.0)
    assert len(res_holder) == 1
    assert res_holder[0][0] is not None

    # Also verify that after join timeout and background worker completion, semaphore is released
    started_evt2 = threading.Event()
    finish_evt2 = threading.Event()

    def slow_run2(*args, **kwargs):
        started_evt2.set()
        finish_evt2.wait(timeout=2.0)
        return [np.array([[0.0]])]

    mock_session.run.side_effect = slow_run2
    scores_to, err_to = r.score_pairs([("q", "d3")], budget_s=0.03)
    assert scores_to is None
    assert err_to == "budget exceeded"

    scores_busy, err_busy = r.score_pairs([("q", "d4")], budget_s=1.0)
    assert scores_busy is None
    assert "unavailable" in err_busy

    finish_evt2.set()
    time.sleep(0.05)

    mock_session.run.side_effect = lambda *args, **kwargs: [np.array([[0.0]])]
    scores_ok, err_ok = r.score_pairs([("q", "d5")], budget_s=1.0)
    assert err_ok is None
    assert scores_ok is not None


def test_finding3_tokenization_inside_deadline_checked_chunk_loop():
    """Finding 3 regression: slow tokenization runs inside the deadline-checked chunk loop and respects timeout."""
    import time
    mock_session = Mock()
    mock_tok = Mock()

    def slow_encode_batch(batch):
        time.sleep(0.15)
        class DummyEnc:
            ids = [1, 2]
            attention_mask = [1, 1]
        return [DummyEnc() for _ in batch]

    mock_tok.encode_batch.side_effect = slow_encode_batch
    mock_session.run.return_value = [np.array([[0.0]])]

    r = Reranker(session=mock_session, tokenizer=mock_tok, pad_id=1, max_length=512)
    t0 = time.monotonic()
    scores, err = r.score_pairs([("q", "doc")], budget_s=0.03)
    elapsed = time.monotonic() - t0

    assert scores is None
    assert err == "budget exceeded"
    assert elapsed < 0.12


def test_finding6_score_pairs_returns_tuple_per_request():
    """Finding 6 regression: score_pairs returns (scores, error) tuple per request without shared last_error coupling."""
    mock_session = Mock()
    mock_tok = Mock()

    class DummyEnc:
        def __init__(self):
            self.ids = [1, 2]
            self.attention_mask = [1, 1]

    mock_tok.encode_batch.side_effect = lambda batch: [DummyEnc() for _ in batch]
    mock_session.run.return_value = [np.array([[0.0]])]

    r = Reranker(session=mock_session, tokenizer=mock_tok, pad_id=1, max_length=512)

    scores, err = r.score_pairs([("q", "d")])
    assert isinstance(scores, list)
    assert err is None

    scores_err, err_msg = r.score_pairs([("q", "d")], budget_s=-1.0)
    assert scores_err is None
    assert err_msg == "budget exceeded"


def _setup_dummy_model_files(tmp_path, include_fp16=True):
    tok_json = tmp_path / "tokenizer.json"
    tok_cfg = tmp_path / "tokenizer_config.json"
    tok_cfg.write_text(json.dumps({"pad_token_id": 99, "max_length": 512}), encoding="utf-8")
    tok_json.write_text(json.dumps({"version": "1.0", "truncation": {"max_length": 512}, "padding": {"pad_id": 99}}), encoding="utf-8")
    onnx_dir = tmp_path / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    int8_file = onnx_dir / "model_int8.onnx"
    int8_file.write_bytes(b"dummy int8 onnx")
    fp16_file = onnx_dir / "model_fp16.onnx"
    if include_fp16:
        fp16_file.write_bytes(b"dummy fp16 onnx")
    return int8_file, fp16_file, tok_json, tok_cfg


def test_auto_device_with_cuda_available_loads_fp16_cuda(tmp_path):
    int8_file, fp16_file, tok_json, tok_cfg = _setup_dummy_model_files(tmp_path)

    mock_preload = Mock()
    mock_session = Mock()
    mock_session.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    mock_tok = Mock()

    def mock_hf_cache(repo_id, filename, revision):
        if filename == "onnx/model_fp16.onnx":
            return str(fp16_file)
        if filename == "onnx/model_int8.onnx":
            return str(int8_file)
        if filename == "tokenizer.json":
            return str(tok_json)
        if filename == "tokenizer_config.json":
            return str(tok_cfg)
        return None

    import onnxruntime
    with patch("huggingface_hub.try_to_load_from_cache", side_effect=mock_hf_cache):
        with patch.object(onnxruntime, "get_available_providers", return_value=["CUDAExecutionProvider", "CPUExecutionProvider"]):
            with patch.object(onnxruntime, "preload_dlls", mock_preload, create=True):
                with patch("tokenizers.Tokenizer.from_file", return_value=mock_tok):
                    with patch("onnxruntime.InferenceSession", return_value=mock_session) as mock_inf_cls:
                        inst, err = get_reranker_with_reason()
                        assert err is None
                        assert inst is not None
                        assert inst.device == "cuda"
                        assert inst.fallback_reason is None
                        assert "model_fp16.onnx" in inst.model_path
                        mock_preload.assert_called_once()
                        mock_inf_cls.assert_called_once()
                        mock_session.disable_fallback.assert_called_once()
                        _, kwargs = mock_inf_cls.call_args
                        assert kwargs.get("providers") == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_cuda_listed_but_session_lands_on_cpu_fallback_with_reason(tmp_path):
    int8_file, fp16_file, tok_json, tok_cfg = _setup_dummy_model_files(tmp_path)

    mock_cuda_session = Mock()
    mock_cuda_session.get_providers.return_value = ["CPUExecutionProvider"]

    mock_cpu_session = Mock()
    mock_cpu_session.get_providers.return_value = ["CPUExecutionProvider"]

    mock_tok = Mock()

    def mock_hf_cache(repo_id, filename, revision):
        if filename == "onnx/model_fp16.onnx":
            return str(fp16_file)
        if filename == "onnx/model_int8.onnx":
            return str(int8_file)
        if filename == "tokenizer.json":
            return str(tok_json)
        if filename == "tokenizer_config.json":
            return str(tok_cfg)
        return None

    import onnxruntime
    with patch("huggingface_hub.try_to_load_from_cache", side_effect=mock_hf_cache):
        with patch.object(onnxruntime, "get_available_providers", return_value=["CUDAExecutionProvider", "CPUExecutionProvider"]), patch.object(onnxruntime, "preload_dlls", Mock(), create=True):
            with patch("tokenizers.Tokenizer.from_file", return_value=mock_tok):
                with patch("onnxruntime.InferenceSession", side_effect=[mock_cuda_session, mock_cpu_session]):
                    inst, err = get_reranker_with_reason()
                    assert err is None
                    assert inst is not None
                    assert inst.device == "cpu"
                    assert inst.fallback_reason is not None
                    assert "silently on CPU" in inst.fallback_reason
                    assert "model_int8.onnx" in inst.model_path


def test_fp16_file_missing_falls_back_to_cpu_with_reason(tmp_path):
    int8_file, _, tok_json, tok_cfg = _setup_dummy_model_files(tmp_path, include_fp16=False)

    mock_cpu_session = Mock()
    mock_cpu_session.get_providers.return_value = ["CPUExecutionProvider"]
    mock_tok = Mock()

    def mock_hf_cache(repo_id, filename, revision):
        if filename == "onnx/model_fp16.onnx":
            return None
        if filename == "onnx/model_int8.onnx":
            return str(int8_file)
        if filename == "tokenizer.json":
            return str(tok_json)
        if filename == "tokenizer_config.json":
            return str(tok_cfg)
        return None

    import onnxruntime
    with patch("huggingface_hub.try_to_load_from_cache", side_effect=mock_hf_cache):
        with patch.object(onnxruntime, "get_available_providers", return_value=["CUDAExecutionProvider", "CPUExecutionProvider"]), patch.object(onnxruntime, "preload_dlls", Mock(), create=True):
            with patch("tokenizers.Tokenizer.from_file", return_value=mock_tok):
                with patch("onnxruntime.InferenceSession", return_value=mock_cpu_session):
                    inst, err = get_reranker_with_reason()
                    assert err is None
                    assert inst is not None
                    assert inst.device == "cpu"
                    assert inst.fallback_reason is not None
                    assert "fp16 model missing" in inst.fallback_reason
                    assert "model_int8.onnx" in inst.model_path


def test_device_cpu_forces_cpu_path_without_cuda(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPHIFY_RERANK_DEVICE", "cpu")
    int8_file, fp16_file, tok_json, tok_cfg = _setup_dummy_model_files(tmp_path)

    mock_session = Mock()
    mock_session.get_providers.return_value = ["CPUExecutionProvider"]
    mock_tok = Mock()

    def mock_hf_cache(repo_id, filename, revision):
        if filename == "onnx/model_fp16.onnx":
            return str(fp16_file)
        if filename == "onnx/model_int8.onnx":
            return str(int8_file)
        if filename == "tokenizer.json":
            return str(tok_json)
        if filename == "tokenizer_config.json":
            return str(tok_cfg)
        return None

    import onnxruntime
    with patch("huggingface_hub.try_to_load_from_cache", side_effect=mock_hf_cache):
        with patch.object(onnxruntime, "get_available_providers", return_value=["CUDAExecutionProvider", "CPUExecutionProvider"]):
            with patch("tokenizers.Tokenizer.from_file", return_value=mock_tok):
                with patch("onnxruntime.InferenceSession", return_value=mock_session) as mock_inf_cls:
                    inst, err = get_reranker_with_reason()
                    assert err is None
                    assert inst is not None
                    assert inst.device == "cpu"
                    assert inst.fallback_reason is None
                    assert "model_int8.onnx" in inst.model_path
                    mock_inf_cls.assert_called_once()
                    _, kwargs = mock_inf_cls.call_args
                    assert kwargs.get("providers") == ["CPUExecutionProvider"]


def test_runtime_cuda_error_returns_no_rerank_with_reason():
    mock_session = Mock()
    mock_session.run.side_effect = RuntimeError("CUDA out of memory: tried to allocate 2.00 GiB")
    mock_tok = Mock()

    class DummyEnc:
        def __init__(self):
            self.ids = [1, 2]
            self.attention_mask = [1, 1]

    mock_tok.encode_batch.side_effect = lambda batch: [DummyEnc() for _ in batch]

    r = Reranker(session=mock_session, tokenizer=mock_tok, pad_id=1, max_length=512, device="cuda")
    scores, err = r.score_pairs([("query", "document")])
    assert scores is None
    assert err is not None
    assert "cuda" in err.lower() or "out of memory" in err.lower()
    assert r.last_error == err


def test_threads_env_configures_session_options(tmp_path, monkeypatch):
    int8_file, _, tok_json, tok_cfg = _setup_dummy_model_files(tmp_path)
    mock_tok = Mock()

    def mock_hf_cache(repo_id, filename, revision):
        if filename == "onnx/model_int8.onnx":
            return str(int8_file)
        if filename == "tokenizer.json":
            return str(tok_json)
        if filename == "tokenizer_config.json":
            return str(tok_cfg)
        return None

    captured_options = []

    def mock_session_factory(model_path, sess_options=None, providers=None):
        captured_options.append(sess_options)
        m = Mock()
        m.get_providers.return_value = ["CPUExecutionProvider"]
        return m

    monkeypatch.setenv("GRAPHIFY_RERANK_DEVICE", "cpu")
    monkeypatch.setenv("GRAPHIFY_RERANK_THREADS", "4")

    import onnxruntime
    with patch("huggingface_hub.try_to_load_from_cache", side_effect=mock_hf_cache):
        with patch("tokenizers.Tokenizer.from_file", return_value=mock_tok):
            with patch("onnxruntime.InferenceSession", side_effect=mock_session_factory):
                inst, err = get_reranker_with_reason()
                assert err is None
                assert inst is not None
                assert len(captured_options) == 1
                assert captured_options[0].intra_op_num_threads == 4

    # Default without env set should be 1
    reset_reranker()
    monkeypatch.delenv("GRAPHIFY_RERANK_THREADS", raising=False)
    captured_options.clear()

    with patch("huggingface_hub.try_to_load_from_cache", side_effect=mock_hf_cache):
        with patch("tokenizers.Tokenizer.from_file", return_value=mock_tok):
            with patch("onnxruntime.InferenceSession", side_effect=mock_session_factory):
                inst, err = get_reranker_with_reason()
                assert err is None
                assert inst is not None
                assert len(captured_options) == 1
                assert captured_options[0].intra_op_num_threads == 1


def test_fp16_logits_cast_to_float32_before_sigmoid():
    mock_session = Mock()
    mock_tok = Mock()

    class DummyEnc:
        def __init__(self):
            self.ids = [10, 20, 1]
            self.attention_mask = [1, 1, 0]

    mock_tok.encode_batch.side_effect = lambda batch: [DummyEnc() for _ in batch]
    mock_session.run.return_value = [np.array([[2.1972245773362196]], dtype=np.float16)]

    r = Reranker(session=mock_session, tokenizer=mock_tok, pad_id=1, max_length=512, device="cuda")
    scores, err = r.score_pairs([("q", "doc")])
    assert err is None
    assert scores is not None
    assert len(scores) == 1
    assert np.isclose(scores[0], 0.9, atol=1e-3)


def test_preload_dlls_failure_falls_back_to_cpu(tmp_path):
    int8_file, fp16_file, tok_json, tok_cfg = _setup_dummy_model_files(tmp_path)

    mock_cpu_session = Mock()
    mock_cpu_session.get_providers.return_value = ["CPUExecutionProvider"]
    mock_tok = Mock()

    def mock_hf_cache(repo_id, filename, revision):
        if filename == "onnx/model_fp16.onnx":
            return str(fp16_file)
        if filename == "onnx/model_int8.onnx":
            return str(int8_file)
        if filename == "tokenizer.json":
            return str(tok_json)
        if filename == "tokenizer_config.json":
            return str(tok_cfg)
        return None

    import onnxruntime
    with patch("huggingface_hub.try_to_load_from_cache", side_effect=mock_hf_cache):
        with patch.object(onnxruntime, "get_available_providers", return_value=["CUDAExecutionProvider", "CPUExecutionProvider"]):
            with patch.object(onnxruntime, "preload_dlls", side_effect=RuntimeError("DLL preload failed"), create=True):
                with patch("tokenizers.Tokenizer.from_file", return_value=mock_tok):
                    with patch("onnxruntime.InferenceSession", return_value=mock_cpu_session):
                        inst, err = get_reranker_with_reason()
                        assert err is None
                        assert inst is not None
                        assert inst.device == "cpu"
                        assert inst.fallback_reason is not None
                        assert "preload_dlls" in inst.fallback_reason
                        assert "model_int8.onnx" in inst.model_path


def test_cuda_session_disables_fallback_and_epfail_reaches_error_handler(tmp_path):
    int8_file, fp16_file, tok_json, tok_cfg = _setup_dummy_model_files(tmp_path)

    mock_cuda_session = Mock()
    mock_cuda_session.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    mock_cuda_session.run.side_effect = RuntimeError("EPFail: CUDA kernel execution failed")
    mock_tok = Mock()

    class DummyEnc:
        def __init__(self):
            self.ids = [1, 2]
            self.attention_mask = [1, 1]

    mock_tok.encode_batch.side_effect = lambda batch: [DummyEnc() for _ in batch]

    def mock_hf_cache(repo_id, filename, revision):
        if filename == "onnx/model_fp16.onnx":
            return str(fp16_file)
        if filename == "onnx/model_int8.onnx":
            return str(int8_file)
        if filename == "tokenizer.json":
            return str(tok_json)
        if filename == "tokenizer_config.json":
            return str(tok_cfg)
        return None

    import onnxruntime
    with patch("huggingface_hub.try_to_load_from_cache", side_effect=mock_hf_cache):
        with patch.object(onnxruntime, "get_available_providers", return_value=["CUDAExecutionProvider", "CPUExecutionProvider"]):
            with patch.object(onnxruntime, "preload_dlls", Mock(), create=True):
                with patch("tokenizers.Tokenizer.from_file", return_value=mock_tok):
                    with patch("onnxruntime.InferenceSession", return_value=mock_cuda_session):
                        inst, err = get_reranker_with_reason()
                        assert err is None
                        assert inst is not None
                        assert inst.device == "cuda"
                        mock_cuda_session.disable_fallback.assert_called_once()

                        scores, score_err = inst.score_pairs([("query", "document")])
                        assert scores is None
                        assert score_err is not None
                        assert "epfail" in score_err.lower() or "cuda" in score_err.lower()
                        assert inst.last_error == score_err


def test_disable_fallback_failure_falls_back_to_cpu(tmp_path):
    int8_file, fp16_file, tok_json, tok_cfg = _setup_dummy_model_files(tmp_path)

    mock_cuda_session = Mock()
    mock_cuda_session.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    mock_cuda_session.disable_fallback.side_effect = RuntimeError("disable_fallback failed")

    mock_cpu_session = Mock()
    mock_cpu_session.get_providers.return_value = ["CPUExecutionProvider"]
    mock_tok = Mock()

    def mock_hf_cache(repo_id, filename, revision):
        if filename == "onnx/model_fp16.onnx":
            return str(fp16_file)
        if filename == "onnx/model_int8.onnx":
            return str(int8_file)
        if filename == "tokenizer.json":
            return str(tok_json)
        if filename == "tokenizer_config.json":
            return str(tok_cfg)
        return None

    import onnxruntime
    with patch("huggingface_hub.try_to_load_from_cache", side_effect=mock_hf_cache):
        with patch.object(onnxruntime, "get_available_providers", return_value=["CUDAExecutionProvider", "CPUExecutionProvider"]):
            with patch.object(onnxruntime, "preload_dlls", Mock(), create=True):
                with patch("tokenizers.Tokenizer.from_file", return_value=mock_tok):
                    with patch("onnxruntime.InferenceSession", side_effect=[mock_cuda_session, mock_cpu_session]):
                        inst, err = get_reranker_with_reason()
                        assert err is None
                        assert inst is not None
                        assert inst.device == "cpu"
                        assert inst.fallback_reason is not None
                        assert "disable fallback" in inst.fallback_reason
                        assert "model_int8.onnx" in inst.model_path


def test_cpu_session_constructor_guards_stdout(tmp_path, monkeypatch, capsys):
    """Regression test (Finding 2): CPU ORT session constructor redirects stdout to stderr."""
    int8_file, fp16_file, tok_json, tok_cfg = _setup_dummy_model_files(tmp_path)
    monkeypatch.setenv("GRAPHIFY_RERANK_DEVICE", "cpu")

    mock_cpu_session = Mock()
    mock_cpu_session.get_providers.return_value = ["CPUExecutionProvider"]
    mock_tok = Mock()

    def noisy_session_factory(*args, **kwargs):
        print("ORT CPU initialization notice on stdout")
        return mock_cpu_session

    def mock_hf_cache(repo_id, filename, revision):
        if filename == "onnx/model_int8.onnx":
            return str(int8_file)
        if filename == "tokenizer.json":
            return str(tok_json)
        if filename == "tokenizer_config.json":
            return str(tok_cfg)
        return None

    import onnxruntime
    with patch("huggingface_hub.try_to_load_from_cache", side_effect=mock_hf_cache):
        with patch("tokenizers.Tokenizer.from_file", return_value=mock_tok):
            with patch("onnxruntime.InferenceSession", side_effect=noisy_session_factory):
                inst, err = get_reranker_with_reason()
                assert err is None
                assert inst is not None
                captured = capsys.readouterr()
                assert "ORT CPU initialization notice on stdout" not in captured.out, "CPU ORT session constructor must not write to stdout!"
                assert "ORT CPU initialization notice on stdout" in captured.err, "CPU ORT session constructor output must be redirected to stderr!"


def test_cpu_session_disables_fallback_and_guards_stdout_on_runtime_error(tmp_path, monkeypatch, capsys):
    """Regression test: CPU ORT session disables fallback and guards stdout during runtime errors."""
    int8_file, fp16_file, tok_json, tok_cfg = _setup_dummy_model_files(tmp_path)
    monkeypatch.setenv("GRAPHIFY_RERANK_DEVICE", "cpu")

    mock_cpu_session = Mock()
    mock_cpu_session.get_providers.return_value = ["CPUExecutionProvider"]

    class DummyEnc:
        def __init__(self):
            self.ids = [1, 2]
            self.attention_mask = [1, 1]

    mock_tok = Mock()
    mock_tok.encode_batch.side_effect = lambda batch: [DummyEnc() for _ in batch]

    def noisy_runtime_fail(*args, **kwargs):
        print("ORT CPU runtime EPFail fallback notice on stdout")
        raise RuntimeError("EPFail: execution provider failed")

    mock_cpu_session.run.side_effect = noisy_runtime_fail

    def mock_hf_cache(repo_id, filename, revision):
        if filename == "onnx/model_int8.onnx":
            return str(int8_file)
        if filename == "tokenizer.json":
            return str(tok_json)
        if filename == "tokenizer_config.json":
            return str(tok_cfg)
        return None

    import onnxruntime
    with patch("huggingface_hub.try_to_load_from_cache", side_effect=mock_hf_cache):
        with patch("tokenizers.Tokenizer.from_file", return_value=mock_tok):
            with patch("onnxruntime.InferenceSession", return_value=mock_cpu_session):
                inst, err = get_reranker_with_reason()
                assert err is None
                assert inst is not None
                assert inst.device == "cpu"
                mock_cpu_session.disable_fallback.assert_called_once()

                capsys.readouterr()

                scores, score_err = inst.score_pairs([("query", "document")])
                assert scores is None
                assert score_err is not None
                assert "epfail" in score_err.lower() or "runtime" in score_err.lower() or "cuda" in score_err.lower()

                captured = capsys.readouterr()
                assert "ORT CPU runtime EPFail fallback notice on stdout" not in captured.out, "CPU ORT runtime error must not write to stdout!"
                assert "ORT CPU runtime EPFail fallback notice on stdout" in captured.err, "CPU ORT runtime error output must be redirected to stderr!"


def test_cpu_session_disable_fallback_failure_fails_cleanly(tmp_path, monkeypatch):
    """Regression test: CPU ORT session returns clean error when disable_fallback fails."""
    int8_file, fp16_file, tok_json, tok_cfg = _setup_dummy_model_files(tmp_path)
    monkeypatch.setenv("GRAPHIFY_RERANK_DEVICE", "cpu")

    mock_cpu_session = Mock()
    mock_cpu_session.get_providers.return_value = ["CPUExecutionProvider"]
    mock_cpu_session.disable_fallback.side_effect = RuntimeError("disable_fallback failed on CPU")
    mock_tok = Mock()

    def mock_hf_cache(repo_id, filename, revision):
        if filename == "onnx/model_int8.onnx":
            return str(int8_file)
        if filename == "tokenizer.json":
            return str(tok_json)
        if filename == "tokenizer_config.json":
            return str(tok_cfg)
        return None

    import onnxruntime
    with patch("huggingface_hub.try_to_load_from_cache", side_effect=mock_hf_cache):
        with patch("tokenizers.Tokenizer.from_file", return_value=mock_tok):
            with patch("onnxruntime.InferenceSession", return_value=mock_cpu_session):
                inst, err = get_reranker_with_reason()
                assert inst is None
                assert "failed to disable fallback" in err

