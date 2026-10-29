"""Local cross-encoder reranker using ONNX Runtime and tokenizers."""
import json
import os
import threading
import time
from typing import Optional, List, Tuple
import numpy as np

DEFAULT_REPO_ID = "onnx-community/bge-reranker-v2-m3-ONNX"
DEFAULT_REVISION = "6f5ff65298512715a1e669753bc754d2bc8f367b"
DEFAULT_MODEL_FILENAME = "onnx/model_int8.onnx"
DEFAULT_BUDGET_S = 3.0
DEFAULT_BATCH_SIZE = 16

_singleton_lock = threading.Lock()
_singleton_instance: Optional["Reranker"] = None
_singleton_error: Optional[str] = None
_singleton_attempted: bool = False


class Reranker:
    def __init__(self, session, tokenizer, pad_id: int, max_length: int, model_max_length: Optional[int] = None):
        self.session = session
        self.tokenizer = tokenizer
        self.pad_id = pad_id
        self.max_length = max_length
        self.model_max_length = model_max_length
        self.last_error: Optional[str] = None
        self._inference_sem = threading.Semaphore(1)

    def score_pairs(self, pairs: List[Tuple[str, str]], budget_s: Optional[float] = None) -> Tuple[Optional[List[float]], Optional[str]]:
        if not pairs:
            return [], None

        if budget_s is None:
            env_budget = os.environ.get("GRAPHIFY_RERANK_BUDGET_S")
            if env_budget is not None:
                try:
                    budget_s = float(env_budget)
                except ValueError:
                    budget_s = DEFAULT_BUDGET_S
            else:
                budget_s = DEFAULT_BUDGET_S

        if budget_s <= 0:
            self.last_error = "budget exceeded"
            return None, "budget exceeded"

        if not self._inference_sem.acquire(blocking=False):
            err = "unavailable: inference in progress"
            self.last_error = err
            return None, err

        sem_abandoned_to_worker = False
        try:
            start_time = time.monotonic()
            deadline = start_time + budget_s
            if time.monotonic() >= deadline:
                self.last_error = "budget exceeded"
                return None, "budget exceeded"

            scores: List[float] = []
            batch_size = DEFAULT_BATCH_SIZE

            for i in range(0, len(pairs), batch_size):
                rem = deadline - time.monotonic()
                if rem <= 0:
                    self.last_error = "budget exceeded"
                    return None, "budget exceeded"

                chunk = pairs[i : i + batch_size]
                output_holder = []
                error_holder = []
                chunk_lock = threading.Lock()
                chunk_state = {"status": "RUNNING"}

                def _run_worker():
                    try:
                        encodings = self.tokenizer.encode_batch(chunk)
                        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
                        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
                        res = self.session.run(None, {
                            "input_ids": input_ids,
                            "attention_mask": attention_mask
                        })
                        output_holder.append(res)
                    except BaseException as ex:
                        error_holder.append(ex)
                    finally:
                        with chunk_lock:
                            if chunk_state["status"] == "ABANDONED":
                                self._inference_sem.release()
                            else:
                                chunk_state["status"] = "COMPLETED"

                inf_thread = threading.Thread(target=_run_worker, daemon=True)
                inf_thread.start()
                inf_thread.join(timeout=max(0.0, rem))

                with chunk_lock:
                    if chunk_state["status"] == "COMPLETED":
                        timed_out = False
                    else:
                        chunk_state["status"] = "ABANDONED"
                        sem_abandoned_to_worker = True
                        timed_out = True

                if timed_out:
                    self.last_error = "budget exceeded"
                    return None, "budget exceeded"

                if error_holder:
                    ex = error_holder[0]
                    err_str = str(ex)
                    if "too short" in err_str.lower() or "truncat" in err_str.lower():
                        err_msg = f"query exceeds max length: {ex}"
                    else:
                        err_msg = f"scoring error: {ex}"
                    self.last_error = err_msg
                    return None, err_msg

                if not output_holder:
                    err_msg = "scoring error: no outputs produced"
                    self.last_error = err_msg
                    return None, err_msg

                outputs = output_holder[0]
                logits = np.squeeze(outputs[0])
                if logits.ndim == 0:
                    batch_scores = [float(1.0 / (1.0 + np.exp(-float(logits))))]
                else:
                    batch_scores = [float(1.0 / (1.0 + np.exp(-float(l)))) for l in logits]
                scores.extend(batch_scores)

                if time.monotonic() >= deadline and (i + batch_size < len(pairs)):
                    self.last_error = "budget exceeded"
                    return None, "budget exceeded"

            return scores, None
        finally:
            if not sem_abandoned_to_worker:
                self._inference_sem.release()


def _resolve_model_and_tokenizer() -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Resolves (model_path, tok_json_path, tok_config_path, error_reason).
    Never downloads from the internet.
    """
    env_model = os.environ.get("GRAPHIFY_RERANKER_MODEL")
    if env_model:
        if not os.path.exists(env_model):
            return None, None, None, f"model missing: path '{env_model}' does not exist"
        model_path = env_model
        model_dir = os.path.dirname(model_path) if os.path.isfile(model_path) else model_path

        tok_json = None
        for candidate in [
            os.path.join(model_dir, "tokenizer.json"),
            os.path.join(os.path.dirname(model_dir), "tokenizer.json")
        ]:
            if os.path.exists(candidate):
                tok_json = candidate
                break

        tok_cfg = None
        for candidate in [
            os.path.join(model_dir, "tokenizer_config.json"),
            os.path.join(os.path.dirname(model_dir), "tokenizer_config.json")
        ]:
            if os.path.exists(candidate):
                tok_cfg = candidate
                break

        return model_path, tok_json, tok_cfg, None

    try:
        import huggingface_hub
    except ImportError:
        return None, None, None, "load error: huggingface_hub not installed"

    cached_model = huggingface_hub.try_to_load_from_cache(
        repo_id=DEFAULT_REPO_ID,
        filename=DEFAULT_MODEL_FILENAME,
        revision=DEFAULT_REVISION
    )
    if not isinstance(cached_model, str) or not os.path.exists(cached_model):
        return None, None, None, "model missing: not found in huggingface cache"

    cached_tok_json = huggingface_hub.try_to_load_from_cache(
        repo_id=DEFAULT_REPO_ID,
        filename="tokenizer.json",
        revision=DEFAULT_REVISION
    )
    tok_json = cached_tok_json if isinstance(cached_tok_json, str) and os.path.exists(cached_tok_json) else None

    cached_tok_cfg = huggingface_hub.try_to_load_from_cache(
        repo_id=DEFAULT_REPO_ID,
        filename="tokenizer_config.json",
        revision=DEFAULT_REVISION
    )
    tok_cfg = cached_tok_cfg if isinstance(cached_tok_cfg, str) and os.path.exists(cached_tok_cfg) else None

    if not tok_json:
        model_dir = os.path.dirname(cached_model)
        for candidate in [
            os.path.join(model_dir, "tokenizer.json"),
            os.path.join(os.path.dirname(model_dir), "tokenizer.json")
        ]:
            if os.path.exists(candidate):
                tok_json = candidate
                break

    if not tok_cfg:
        model_dir = os.path.dirname(cached_model)
        for candidate in [
            os.path.join(model_dir, "tokenizer_config.json"),
            os.path.join(os.path.dirname(model_dir), "tokenizer_config.json")
        ]:
            if os.path.exists(candidate):
                tok_cfg = candidate
                break

    return cached_model, tok_json, tok_cfg, None


def _load_reranker() -> Tuple[Optional[Reranker], Optional[str]]:
    """Attempts to instantiate Reranker. Returns (instance, error_message)."""
    model_path, tok_json, tok_cfg, err = _resolve_model_and_tokenizer()
    if err:
        return None, err
    if not model_path:
        return None, "model missing"

    if not tok_json or not os.path.exists(tok_json):
        return None, "load error: tokenizer.json not found"

    try:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(tok_json)
    except Exception as e:
        return None, f"load error: failed to initialize tokenizer from {tok_json}: {e}"

    pad_id = None
    max_length = None
    model_max_length = None

    if tok_cfg and os.path.exists(tok_cfg):
        try:
            with open(tok_cfg, "r", encoding="utf-8") as f:
                cfg_data = json.load(f)
            if "pad_token_id" in cfg_data and cfg_data["pad_token_id"] is not None:
                pad_id = int(cfg_data["pad_token_id"])
            elif "pad_token" in cfg_data and cfg_data["pad_token"] is not None:
                pt = cfg_data["pad_token"]
                pt_str = pt.get("content") if isinstance(pt, dict) else str(pt)
                pad_id = tokenizer.token_to_id(pt_str)

            if "model_max_length" in cfg_data and cfg_data["model_max_length"] is not None:
                model_max_length = int(cfg_data["model_max_length"])

            if "max_length" in cfg_data and cfg_data["max_length"] is not None:
                max_length = int(cfg_data["max_length"])
        except Exception:
            pass

    try:
        with open(tok_json, "r", encoding="utf-8") as f:
            tok_json_data = json.load(f)
        if max_length is None:
            trunc = tok_json_data.get("truncation")
            if isinstance(trunc, dict) and trunc.get("max_length") is not None:
                max_length = int(trunc["max_length"])
        if pad_id is None:
            padding = tok_json_data.get("padding")
            if isinstance(padding, dict) and padding.get("pad_id") is not None:
                pad_id = int(padding["pad_id"])
            elif "added_tokens" in tok_json_data:
                for at in tok_json_data["added_tokens"]:
                    if at.get("content") in ("<pad>", "[PAD]"):
                        pad_id = int(at["id"])
                        break
    except Exception:
        pass

    if pad_id is None:
        for candidate in ("<pad>", "[PAD]"):
            pid = tokenizer.token_to_id(candidate)
            if pid is not None:
                pad_id = pid
                break

    if max_length is None and model_max_length is not None:
        max_length = min(model_max_length, 512)

    if max_length is not None:
        max_length = min(max_length, 512)

    if pad_id is None:
        return None, "load error: pad_id not found in tokenizer files"
    if max_length is None:
        return None, "load error: max_length not found in tokenizer files"

    try:
        tokenizer.enable_truncation(max_length=max_length, strategy="only_second")
        pad_token_str = tokenizer.id_to_token(pad_id) or "<pad>"
        tokenizer.enable_padding(pad_id=pad_id, pad_token=pad_token_str)
    except Exception as e:
        return None, f"load error: failed to configure tokenizer padding/truncation: {e}"

    try:
        import onnxruntime as ort
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        session = ort.InferenceSession(model_path, sess_options=sess_options, providers=["CPUExecutionProvider"])
    except Exception as e:
        return None, f"load error: failed to initialize onnxruntime session: {e}"

    return Reranker(session=session, tokenizer=tokenizer, pad_id=pad_id, max_length=max_length, model_max_length=model_max_length), None


def get_reranker() -> Optional[Reranker]:
    """Lazy, thread-safe singleton returning Reranker instance or None."""
    global _singleton_instance, _singleton_error, _singleton_attempted
    if _singleton_attempted:
        return _singleton_instance

    with _singleton_lock:
        if not _singleton_attempted:
            inst, err = _load_reranker()
            _singleton_instance = inst
            _singleton_error = err
            _singleton_attempted = True
        return _singleton_instance


def get_reranker_with_reason() -> Tuple[Optional[Reranker], Optional[str]]:
    """Returns (instance, error_reason) for the singleton."""
    global _singleton_instance, _singleton_error, _singleton_attempted
    with _singleton_lock:
        if not _singleton_attempted:
            inst, err = _load_reranker()
            _singleton_instance = inst
            _singleton_error = err
            _singleton_attempted = True
        return _singleton_instance, _singleton_error


def score_pairs(pairs: List[Tuple[str, str]], budget_s: Optional[float] = None) -> Tuple[Optional[List[float]], Optional[str]]:
    """Convenience function to score pairs using the singleton."""
    reranker, err = get_reranker_with_reason()
    if reranker is None:
        return None, err or "reranker unavailable"
    return reranker.score_pairs(pairs, budget_s=budget_s)


def reset_reranker():
    """Resets singleton state (for testing)."""
    global _singleton_instance, _singleton_error, _singleton_attempted
    with _singleton_lock:
        _singleton_instance = None
        _singleton_error = None
        _singleton_attempted = False
