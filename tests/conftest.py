import os
import pytest
import reranker


def pytest_configure(config):
    config.addinivalue_line("markers", "reranker: mark test as opting in to reranker / model loading")


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
