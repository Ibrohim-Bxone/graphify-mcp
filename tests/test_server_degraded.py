"""Tests for degraded search handling in server.py (Requirements G2 and G3)."""

from unittest.mock import MagicMock
import pytest
import search_core
import server


@pytest.fixture
def mock_server_collection(monkeypatch):
    """Fixture providing a mock Chroma collection for server._col."""
    mock_col = MagicMock()
    mock_col.count.return_value = 5
    monkeypatch.setattr(server, "_col", mock_col)
    return mock_col


def test_server_degraded_whitespace_error_fallback(mock_server_collection, monkeypatch):
    """G2: server.py handles degraded_error with only whitespace without raising IndexError,
    and falls back to 'vektor indeksi xatosi'.
    """
    def mock_hybrid(query, **kwargs):
        return {
            "results": [{"text": "Dasturlash asoslari", "meta": {"id": "1", "title": "Asoslar"}}],
            "degraded": True,
            "degraded_error": "   \n\t   ",
        }

    monkeypatch.setattr(search_core, "hybrid_search", mock_hybrid)

    res = server.search_knowledge("Python")
    assert "⚠️ Vektor qidiruv ishlamadi (vektor indeksi xatosi), natija faqat kalit so'z bo'yicha." in res
    assert "Dasturlash asoslari" in res


def test_server_degraded_partial_warning(mock_server_collection, monkeypatch):
    """G3: When searching multiple query variants and only a subset is degraded,
    the warning states 'vektor qidiruvning bir qismi ishlamadi' and does NOT say
    'faqu kalit so'z bo'ycha'.
    """
    call_count = 0

    def mock_hybrid(query, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {
                "results": [{"text": "Xotira haqida", "meta": {"id": "1", "title": "Xotira"}}],
                "degraded": True,
                "degraded_error": "Vektor bazasi vaqtincha band",
            }
        return {
            "results": [{"text": "Memory details", "meta": {"id": "2", "title": "Memory"}}],
            "degraded": False,
        }

    monkeypatch.setattr(search_core, "hybrid_search", mock_hybrid)

    # 'xotira' expands to multiple variants (['xotira', 'memory'])
    res = server.search_knowledge("xotira")
    assert "⚠️ Vektor qidiruvning bir qismi ishlamadi (Vektor bazasi vaqtincha band)." in res
    assert "faqat kalit so'z bo'yicha" not in res
    assert "Xotira haqida" in res or "Memory details" in res


def test_server_degraded_full_warning(mock_server_collection, monkeypatch):
    """G3: When all query variants are degraded, the original warning header is returned."""
    def mock_hybrid(query, **kwargs):
        return {
            "results": [{"text": "Hamma natija", "meta": {"id": "1", "title": "Natija"}}],
            "degraded": True,
            "degraded_error": "Chroma ulanish xatosi",
        }

    monkeypatch.setattr(search_core, "hybrid_search", mock_hybrid)

    res = server.search_knowledge("xotira")
    assert "⚠️ Vektor qidiruv ishlamadi (Chroma ulanish xatosi), natija faqat kalit so'z bo'yicha." in res
    assert "Hamma natija" in res


def test_server_degraded_multiline_error_truncation(mock_server_collection, monkeypatch):
    """G2: server.py truncates long error messages exceeding 120 chars and takes first non-empty line."""
    long_err = "A" * 150 + "\nSecond line details"

    def mock_hybrid(query, **kwargs):
        return {
            "results": [{"text": "Doc text", "meta": {"id": "1", "title": "Doc"}}],
            "degraded": True,
            "degraded_error": long_err,
        }

    monkeypatch.setattr(search_core, "hybrid_search", mock_hybrid)

    res = server.search_knowledge("Python")
    expected_truncated = "A" * 117 + "..."
    assert f"({expected_truncated})" in res
    assert "Second line details" not in res
