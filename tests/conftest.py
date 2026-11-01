import os
import tempfile
from pathlib import Path
import pytest
import reranker

# Module-level isolation: test modules import db / keyword_index during collection.
# Always overwrite GRAPHIFY_DB_PATH so external environment never pollutes tests.
_TEST_DB_PATH = tempfile.mkdtemp(prefix="graphify-test-")
os.environ["GRAPHIFY_DB_PATH"] = _TEST_DB_PATH

# Track checkout kg_db presence before test collection starts
_CHECKOUT_ROOT = Path(__file__).resolve().parents[1]
_CHECKOUT_KG_DB = _CHECKOUT_ROOT / "kg_db"
_INITIAL_KG_DB_EXISTS = _CHECKOUT_KG_DB.exists()


def pytest_configure(config):
    config.addinivalue_line("markers", "reranker: mark test as opting in to reranker / model loading")


def pytest_sessionfinish(session, exitstatus):
    if not _INITIAL_KG_DB_EXISTS:
        assert not _CHECKOUT_KG_DB.exists(), (
            f"Test to'plami tugagach checkout ichida kg_db/ yaratilgan bo'lmasligi kerak: {_CHECKOUT_KG_DB}"
        )


@pytest.fixture
def enable_reranker():
    """Explicit opt-in fixture for tests that need reranker functionality."""
    return True


@pytest.fixture(autouse=True)
def _default_test_environment(request, monkeypatch):
    """Autouse fixture that sets GRAPHIFY_RERANK=0 and blocks model loading,
    with explicit opt-in for reranker tests via marker or fixture.
    """
    is_opted_in = bool(
        request.node.get_closest_marker("reranker")
        or "enable_reranker" in getattr(request, "fixturenames", [])
    )

    if is_opted_in:
        # Reranker sukut bo'yicha o'chiq; opt-in testlar uni aniq yoqadi
        monkeypatch.setenv("GRAPHIFY_RERANK", "1")
    else:
        monkeypatch.setenv("GRAPHIFY_RERANK", "0")

        def _blocked_load_reranker(*args, **kwargs):
            raise RuntimeError(
                "Model loading is blocked in tests without explicit opt-in. "
                "Use @pytest.mark.reranker or enable_reranker fixture."
            )

        monkeypatch.setattr(reranker, "_load_reranker", _blocked_load_reranker)

    yield
