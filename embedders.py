"""Graphify embedder backends — pluggable embedding models for the vector DB.

This turns the design sketched in CUSTOM_MODELS.md into real, working code:
each backend implements the `EmbedderBase.encode()` interface, and
`build_chroma_embedding_function()` wraps the active one so it can be passed
straight into `chromadb`'s `get_or_create_collection(embedding_function=...)`.

The active backend is chosen via models_config.json (project root), which the
dashboard's "Modellar" tab writes to and server.py reads at startup. Backends
other than "local" are OPTIONAL — their imports (openai/requests) are lazy,
so the server always starts and the default local embedder always works even
with zero extra packages installed.

IMPORTANT: switching the embedder on a `kg_db/` collection that already has
data is a breaking change (old vectors live in a different embedding space
than the new model produces). See build_chroma_embedding_function() below —
it refuses to silently corrupt an existing collection.
"""

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

CONFIG_PATH = str(Path(__file__).parent / "models_config.json")

DEFAULT_CONFIG = {
    "embedder_type": "local",
    "config": {
        "openai": {"api_key": "", "model": "text-embedding-3-small"},
        "ollama": {"base_url": "http://127.0.0.1:11434", "model": "nomic-embed-text"},
        "custom": {"base_url": "", "api_key": "", "model": ""},
    },
}

# Metadata for the dashboard's "Modellar" tab.
BACKENDS = {
    "local": {
        "label": "Lokal ONNX MiniLM",
        "cost": "0 token / bepul",
        "key_required": False,
        "desc": "sentence-transformers/all-MiniLM-L6-v2, hammasi kompyuteringizda ishlaydi. Kalit yoki internet shart emas.",
    },
    "openai": {
        "label": "OpenAI Embeddings",
        "cost": "~$0.02 / 1M token (text-embedding-3-small)",
        "key_required": True,
        "desc": "OpenAI API orqali. `openai` paketi va API kalit kerak.",
    },
    "ollama": {
        "label": "Ollama (lokal server)",
        "cost": "Bepul, lokal server orqali",
        "key_required": False,
        "desc": "Kompyuteringizda ishlab turgan Ollama serveriga ulanadi (masalan nomic-embed-text modeli).",
    },
    "custom": {
        "label": "Maxsus API",
        "cost": "API shartlariga bog'liq",
        "key_required": False,
        "desc": "O'zingizning embedding API'ingiz: POST {base_url}/embed -> {\"embeddings\": [[...], ...]}",
    },
}


def load_models_config() -> dict:
    """Read models_config.json, filled in with defaults for any missing keys."""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        data = {}
    merged = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy
    merged["embedder_type"] = data.get("embedder_type") or "local"
    for key, sub in (data.get("config") or {}).items():
        merged["config"].setdefault(key, {}).update(sub or {})
    return merged


def save_models_config(cfg: dict) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


class ModelIdentityError(Exception):
    """Raised when database model identity validation fails (fails closed, no fallback)."""
    pass


class EmbedderBase(ABC):
    """Common interface for every embedding backend (see CUSTOM_MODELS.md)."""

    @abstractmethod
    def encode(self, texts: list) -> np.ndarray:
        """Return an (N, dim) array of embeddings for the given texts."""
        raise NotImplementedError


class LocalEmbedder(EmbedderBase):
    """Default backend: local ONNX MiniLM (via chromadb's bundled default
    embedding function). 0 cost, no key, no network needed."""

    def __init__(self):
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        self._fn = DefaultEmbeddingFunction()

    def encode(self, texts):
        return np.array(self._fn(list(texts)))


class OpenAIEmbedder(EmbedderBase):
    """OpenAI embeddings API. Requires the `openai` package + an API key."""

    def __init__(self, api_key: str = "", model: str = "text-embedding-3-small"):
        try:
            import openai
        except ImportError as e:
            raise RuntimeError("`openai` paketi o'rnatilmagan: pip install openai") from e
        if not api_key:
            raise RuntimeError("OpenAI embedder uchun api_key kerak (Modellar bo'limida kiriting)")
        self._client = openai.OpenAI(api_key=api_key)
        self._model = model

    def encode(self, texts):
        resp = self._client.embeddings.create(model=self._model, input=list(texts))
        return np.array([d.embedding for d in resp.data])


class OllamaEmbedder(EmbedderBase):
    """Local Ollama server. Requires `requests` + Ollama running with an
    embedding-capable model already pulled (e.g. `ollama pull nomic-embed-text`)."""

    def __init__(self, base_url: str = "http://127.0.0.1:11434", model: str = "nomic-embed-text", normalize: bool = False):
        try:
            import requests  # noqa: F401
        except ImportError as e:
            raise RuntimeError("`requests` paketi o'rnatilmagan: pip install requests") from e
        self._base_url = (base_url or "http://127.0.0.1:11434").rstrip("/")
        self._model = model or "nomic-embed-text"
        self._normalize = bool(normalize)
        self.normalize = self._normalize

    def encode(self, texts):
        import re
        import sys
        import requests
        out = []
        for text in texts:
            current_text = text
            retries = 0
            while True:
                r = requests.post(
                    f"{self._base_url}/api/embeddings",
                    json={"model": self._model, "prompt": current_text},
                    timeout=60,
                )
                if r.status_code == 500 and ("too large" in r.text.lower() or "exceeds the context length" in r.text.lower()):
                    if retries < 4:
                        retries += 1
                        orig_len = len(current_text)
                        m = re.search(r"input \((\d+) tokens\) is too large", r.text, re.IGNORECASE)
                        if m:
                            n_tokens = int(m.group(1))
                            new_len = int(orig_len * 8000 / n_tokens * 0.9)
                            if new_len >= orig_len:
                                new_len = orig_len // 2
                            n_val = n_tokens
                        else:
                            new_len = orig_len // 2
                            n_val = None
                        new_len = max(0, new_len)
                        print(
                            f"[OllamaEmbedder] input too large, matn qisqartirildi: asl={orig_len}, yangi={new_len}, N={n_val}",
                            file=sys.stderr,
                            flush=True,
                        )
                        current_text = current_text[:new_len]
                        continue
                    else:
                        raise RuntimeError(
                            f"Ollama input too large: matn 4 marta qisqartirilsa ham sig'madi (oxirgi javob: {r.text})"
                        )
                r.raise_for_status()
                out.append(r.json()["embedding"])
                break
        arr = np.array(out, dtype=np.float32)
        if self._normalize and arr.size > 0:
            if arr.ndim == 1:
                norm = float(np.linalg.norm(arr))
                if norm == 0.0 or np.isnan(norm) or np.isinf(norm):
                    raise ModelIdentityError("Nol vektor normallashtirib bo'lmaydi")
                arr = arr / norm
            elif arr.ndim == 2:
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                if np.any(norms == 0.0) or np.any(np.isnan(norms)) or np.any(np.isinf(norms)):
                    raise ModelIdentityError("Nol vektor normallashtirib bo'lmaydi")
                arr = arr / norms
        return arr


class CustomAPIEmbedder(EmbedderBase):
    """Generic REST embedder: POST {"texts": [...], "model": ...} -> {"embeddings": [...]}."""

    def __init__(self, base_url: str = "", api_key: str = "", model: str = ""):
        try:
            import requests  # noqa: F401
        except ImportError as e:
            raise RuntimeError("`requests` paketi o'rnatilmagan: pip install requests") from e
        if not base_url:
            raise RuntimeError("Maxsus API embedder uchun base_url kerak (Modellar bo'limida kiriting)")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model

    def encode(self, texts):
        import requests
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        r = requests.post(
            f"{self._base_url}/embed",
            json={"texts": list(texts), "model": self._model},
            headers=headers,
            timeout=60,
        )
        r.raise_for_status()
        return np.array(r.json()["embeddings"])


def build_embedder(cfg: dict = None) -> EmbedderBase:
    """Instantiate the EmbedderBase implementation selected in models_config.json."""
    cfg = cfg or load_models_config()
    kind = cfg.get("embedder_type", "local")
    sub = (cfg.get("config") or {}).get(kind, {})
    if kind == "openai":
        return OpenAIEmbedder(api_key=sub.get("api_key", ""), model=sub.get("model") or "text-embedding-3-small")
    if kind == "ollama":
        norm = sub.get("normalize", False) or sub.get("normalization", False) or cfg.get("normalization", False)
        return OllamaEmbedder(base_url=sub.get("base_url") or "http://127.0.0.1:11434", model=sub.get("model") or "nomic-embed-text", normalize=bool(norm))
    if kind == "custom":
        return CustomAPIEmbedder(base_url=sub.get("base_url", ""), api_key=sub.get("api_key", ""), model=sub.get("model", ""))
    return LocalEmbedder()


class _ChromaEmbeddingFunctionAdapter:
    """Wraps an EmbedderBase so it satisfies chromadb's EmbeddingFunction
    protocol (a callable with __call__ + name())."""

    def __init__(self, embedder: EmbedderBase, backend_name: str):
        self._embedder = embedder
        self._backend_name = backend_name

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        return self._embedder.encode(list(input)).tolist()

    def embed_query(self, input):
        return self(input)

    def embed_documents(self, input):
        return self(input)

    def name(self):
        return f"graphify-{self._backend_name}"

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> list:
        return ["cosine", "l2", "ip"]

    def get_config(self) -> dict:
        return {"backend_name": self._backend_name}

    @staticmethod
    def build_from_config(config: dict):
        return NotImplemented

    def is_legacy(self) -> bool:
        return True


def get_ollama_model_digest(base_url: str = "http://127.0.0.1:11434", model: str = "bge-m3") -> str:
    """Fetch model digest from Ollama /api/tags or /api/show."""
    import urllib.request
    
    url_base = (base_url or "http://127.0.0.1:11434").rstrip("/")
    target_tag = model if ":" in model else f"{model}:latest"
    try:
        req = urllib.request.Request(f"{url_base}/api/tags", headers={"User-Agent": "graphify"})
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        for m in data.get("models", []):
            m_name = m.get("name", "")
            m_model = m.get("model", "")
            m_name_norm = m_name if ":" in m_name else (f"{m_name}:latest" if m_name else "")
            m_model_norm = m_model if ":" in m_model else (f"{m_model}:latest" if m_model else "")
            if m_name_norm == target_tag or m_model_norm == target_tag:
                digest = m.get("digest")
                if digest:
                    return digest
    except Exception as e:
        try:
            body = json.dumps({"model": target_tag}).encode("utf-8")
            req = urllib.request.Request(
                f"{url_base}/api/show",
                data=body,
                headers={"Content-Type": "application/json", "User-Agent": "graphify"},
            )
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            digest = data.get("details", {}).get("digest") or data.get("digest")
            if digest:
                return digest
        except Exception:
            raise ModelIdentityError(f"Ollama serveriga ulanib bo'lmadi yoki model topilmadi ({model}): {e}") from e
        raise ModelIdentityError(f"Ollama model digest topilmadi: {model}") from e
    raise ModelIdentityError(f"Ollama model '{model}' /api/tags da topilmadi")


def build_identity_embedder(identity: dict, base_url: str = None) -> EmbedderBase:
    """Instantiate and strictly validate the embedder required by database identity.

    Fails closed: on mismatch (model name, digest or dimension), missing Ollama,
    or a failed 1-query self-test -> raises ModelIdentityError, never falls back.
    """
    backend = identity.get("backend") or identity.get("graphify:backend")
    model = identity.get("model") or identity.get("graphify:model") or ""
    expected_digest = identity.get("model_digest") or identity.get("graphify:model_digest")
    expected_dim = identity.get("dimension") or identity.get("graphify:dimension")

    if not backend:
        raise ModelIdentityError("Identifikatsiyada backend ko'rsatilmagan")
    if not model:
        raise ModelIdentityError("Identifikatsiyada model nomi ko'rsatilmagan")
    if expected_dim is None:
        raise ModelIdentityError("Identifikatsiyada model dimension ko'rsatilmagan")

    raw_norm = identity.get("normalization")
    if raw_norm is None:
        raw_norm = identity.get("graphify:normalization")
    normalize_flag = False
    if raw_norm is not None:
        from identity_utils import _parse_bool
        try:
            normalize_flag = _parse_bool(raw_norm)
        except Exception as e:
            raise ModelIdentityError(f"Identifikatsiyada normalization boolean emas: {raw_norm}") from e

    if backend == "ollama":
        if not expected_digest:
            raise ModelIdentityError("Identifikatsiyada model_digest ko'rsatilmagan")
        ollama_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        actual_digest = get_ollama_model_digest(ollama_url, model)
        if actual_digest != expected_digest:
            raise ModelIdentityError(
                f"Model digest mos kelmadi: kutilgan={expected_digest}, haqiqiy={actual_digest}"
            )
        try:
            embedder = OllamaEmbedder(base_url=ollama_url, model=model, normalize=normalize_flag)
        except Exception as e:
            raise ModelIdentityError(f"Ollama embedder yaratib bo'lmadi: {e}") from e
    elif backend == "local":
        supported_local = {"all-minilm-l6-v2", "sentence-transformers/all-minilm-l6-v2"}
        if model.strip().lower() not in supported_local:
            raise ModelIdentityError(
                f"Qo'llab-quvvatlanmagan lokal model: '{model}'. "
                f"Faqat quyidagilar qo'llab-quvvatlanadi: {', '.join(sorted(supported_local))}"
            )
        embedder = LocalEmbedder()
    elif backend == "openai":
        cfg = load_models_config()
        sub = (cfg.get("config") or {}).get("openai", {})
        embedder = OpenAIEmbedder(api_key=sub.get("api_key", ""), model=model or sub.get("model", "text-embedding-3-small"))
    elif backend == "custom":
        cfg = load_models_config()
        sub = (cfg.get("config") or {}).get("custom", {})
        embedder = CustomAPIEmbedder(base_url=sub.get("base_url", ""), api_key=sub.get("api_key", ""), model=model or sub.get("model", ""))
    else:
        raise ModelIdentityError(f"Noma'lum embedder backend: {backend}")

    # 1-query self-test
    try:
        test_vec = embedder.encode(["self-test"])
        if hasattr(test_vec, "tolist"):
            test_vec = test_vec.tolist()
        elif not isinstance(test_vec, list):
            test_vec = list(test_vec)
    except Exception as e:
        raise ModelIdentityError(f"Embedder 1-query self-test xatoligi: {e}") from e

    if not test_vec or not hasattr(test_vec[0], "__len__"):
        raise ModelIdentityError("Embedder 1-query self-test natijasi noto'g'ri shaklda")

    actual_dim = len(test_vec[0])
    if actual_dim != int(expected_dim):
        raise ModelIdentityError(
            f"Model o'lchami mos kelmadi: kutilgan={expected_dim}, haqiqiy={actual_dim}"
        )

    return embedder



def build_chroma_embedding_function(cfg: dict = None, identity: dict = None):
    """Return a chromadb-compatible embedding_function for get_or_create_collection().

    If identity is provided, builds the exact embedder required and validates it
    strictly (fails closed by raising ModelIdentityError, never falls back).

    If identity is None, preserves today's models_config.json behavior:
    returns None for "local" or falls back to None on startup initialization errors.
    """
    if identity:
        backend = identity.get("backend") or identity.get("graphify:backend") or "ollama"
        embedder = build_identity_embedder(identity)
        return _ChromaEmbeddingFunctionAdapter(embedder, backend)

    cfg = cfg or load_models_config()
    kind = cfg.get("embedder_type", "local")
    if kind == "local":
        return None
    try:
        return _ChromaEmbeddingFunctionAdapter(build_embedder(cfg), kind)
    except Exception as e:  # noqa: BLE001 - never let a bad model config break startup
        import sys
        print(f"[graphify] Ogohlantirish: '{kind}' embedder ishga tushmadi ({e}); lokal embedderga qaytildi.", file=sys.stderr)
        return None
