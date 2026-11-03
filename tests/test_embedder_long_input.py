"""Tests for OllamaEmbedder handling long inputs exceeding context length (E1, E2, E3, E4).

Verifies that OllamaEmbedder truncates texts deterministically on Ollama 500
context length errors (`input (N tokens) is too large` and `exceeds the context length`),
logs the truncation to stderr, and retries up to 4 times before raising RuntimeError.
All tests use monkeypatching/mocking to ensure real Ollama is never called.
"""

import json

import numpy as np
import pytest
import requests
from embedders import OllamaEmbedder


class MockResponse:
    """Mock requests.Response for Ollama /api/embeddings endpoint."""

    def __init__(self, status_code: int, text: str = "", json_data: dict = None):
        self.status_code = status_code
        self.text = text
        self._json_data = json_data

    def json(self):
        if self._json_data is not None:
            return self._json_data
        return json.loads(self.text)

    def raise_for_status(self):
        if 400 <= self.status_code < 600:
            raise requests.exceptions.HTTPError(
                f"{self.status_code} Error: {self.text}",
                response=self,
            )


@pytest.fixture(autouse=True)
def assert_real_network_blocked(monkeypatch):
    """Enforce that real requests.post is replaced and network calls are blocked."""
    def guarded_post(*args, **kwargs):
        raise AssertionError("Real requests.post must never be called in test suite!")

    monkeypatch.setattr(requests, "post", guarded_post)


def test_e1_long_input_truncated_and_succeeds_on_retry(monkeypatch):
    """E1: Long text causing 500 'too large' is truncated with formula and succeeds on 2nd attempt."""
    calls = []

    def mock_post(url, json=None, timeout=None):
        calls.append({"url": url, "json": json, "timeout": timeout})
        if len(calls) == 1:
            return MockResponse(
                500,
                text='{"error": "input (9520 tokens) is too large to process. increase the physical batch size (current batch size: 2048)"}',
            )
        return MockResponse(200, json_data={"embedding": [0.1] * 1024})

    monkeypatch.setattr(requests, "post", mock_post)

    long_text = "A" * 10000
    embedder = OllamaEmbedder(base_url="http://mock-ollama:11434", model="bge-m3")
    result = embedder.encode([long_text])

    # Exactly 2 calls made
    assert len(calls) == 2
    assert "/api/embeddings" in calls[0]["url"]
    assert calls[0]["json"]["model"] == "bge-m3"
    assert calls[0]["json"]["prompt"] == long_text

    # 2nd call prompt is shorter according to formula: int(10000 * 8000 / 9520 * 0.9) = 7563
    expected_len = int(10000 * 8000 / 9520 * 0.9)
    assert expected_len == 7563
    assert len(calls[1]["json"]["prompt"]) == expected_len
    assert len(calls[1]["json"]["prompt"]) < len(calls[0]["json"]["prompt"])
    assert calls[1]["json"]["prompt"] == long_text[:expected_len]

    # Result returned as ndarray
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 1024)


def test_e1_consecutive_five_too_large_raises_runtime_error(monkeypatch):
    """E1: 5 consecutive 'too large' responses (4 retries) raise RuntimeError."""
    calls = []

    def mock_post(url, json=None, timeout=None):
        calls.append({"url": url, "json": json, "timeout": timeout})
        return MockResponse(
            500,
            text='{"error": "input (9520 tokens) is too large to process"}',
        )

    monkeypatch.setattr(requests, "post", mock_post)

    long_text = "B" * 10000
    embedder = OllamaEmbedder(base_url="http://mock-ollama:11434")

    with pytest.raises(RuntimeError, match="too large"):
        embedder.encode([long_text])

    # Initial attempt + 4 retries = exactly 5 requests
    assert len(calls) == 5

    # Each retry successively shortened the prompt
    lengths = [len(c["json"]["prompt"]) for c in calls]
    for i in range(len(lengths) - 1):
        assert lengths[i] > lengths[i + 1], f"Step {i} ({lengths[i]}) not > {i+1} ({lengths[i+1]})"


def test_e2_other_500_error_raises_without_truncation(monkeypatch):
    """E2: HTTP 500 for reasons other than 'too large' raises HTTPError without retrying."""
    calls = []

    def mock_post(url, json=None, timeout=None):
        calls.append({"url": url, "json": json, "timeout": timeout})
        return MockResponse(500, text='{"error": "Internal server error: CUDA out of memory"}')

    monkeypatch.setattr(requests, "post", mock_post)

    embedder = OllamaEmbedder(base_url="http://mock-ollama:11434")

    with pytest.raises(requests.exceptions.HTTPError):
        embedder.encode(["some text"])

    # Must NOT retry: exactly 1 call
    assert len(calls) == 1
    assert calls[0]["json"]["prompt"] == "some text"


def test_e1_short_input_single_request_prompt_unchanged(monkeypatch):
    """E1: Short text produces a single request with unchanged prompt."""
    calls = []

    def mock_post(url, json=None, timeout=None):
        calls.append({"url": url, "json": json, "timeout": timeout})
        return MockResponse(200, json_data={"embedding": [0.42] * 1024})

    monkeypatch.setattr(requests, "post", mock_post)

    text = "Short text for embedding"
    embedder = OllamaEmbedder(base_url="http://mock-ollama:11434", model="nomic-embed-text")
    result = embedder.encode([text])

    assert len(calls) == 1
    assert calls[0]["json"]["prompt"] == text
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 1024)
    assert result[0][0] == np.float32(0.42)


def test_e2_truncation_logs_to_stderr_with_details(monkeypatch, capsys):
    """E2: Truncating prints single line to stderr containing original length, new length, and N."""
    calls = []

    def mock_post(url, json=None, timeout=None):
        calls.append({"url": url, "json": json, "timeout": timeout})
        if len(calls) == 1:
            return MockResponse(
                500,
                text='{"error": "input (9520 tokens) is too large to process"}',
            )
        return MockResponse(200, json_data={"embedding": [0.1] * 1024})

    monkeypatch.setattr(requests, "post", mock_post)

    long_text = "C" * 10000
    embedder = OllamaEmbedder(base_url="http://mock-ollama:11434")
    embedder.encode([long_text])

    captured = capsys.readouterr()
    err_output = captured.err.strip()
    assert err_output != "", "Stderr must not be empty on truncation"

    # Must contain original chars (10000), new chars (7563), and token count N (9520)
    lines = err_output.splitlines()
    assert len(lines) == 1, f"Expected exactly 1 stderr line, got {len(lines)}: {lines}"
    line = lines[0]
    assert "10000" in line, f"Original length missing in: {line}"
    assert "7563" in line, f"New length missing in: {line}"
    assert "9520" in line, f"Token count N missing in: {line}"


def test_e1_truncation_without_token_count_halves_length(monkeypatch):
    """E1: When N cannot be extracted from 500 context error, text length is halved."""
    calls = []

    def mock_post(url, json=None, timeout=None):
        calls.append({"url": url, "json": json, "timeout": timeout})
        if len(calls) == 1:
            return MockResponse(500, text='{"error": "the input length exceeds the context length"}')
        return MockResponse(200, json_data={"embedding": [0.2] * 1024})

    monkeypatch.setattr(requests, "post", mock_post)

    long_text = "D" * 10000
    embedder = OllamaEmbedder(base_url="http://mock-ollama:11434")
    embedder.encode([long_text])

    assert len(calls) == 2
    assert len(calls[0]["json"]["prompt"]) == 10000
    # Halved length: 10000 // 2 = 5000
    assert len(calls[1]["json"]["prompt"]) == 5000
    assert calls[1]["json"]["prompt"] == long_text[:5000]


def test_e3_normalization_preserved_after_truncation(monkeypatch):
    """E3: Normalization logic is preserved after truncation occurs."""
    calls = []

    def mock_post(url, json=None, timeout=None):
        calls.append({"url": url, "json": json, "timeout": timeout})
        if len(calls) == 1:
            return MockResponse(500, text='{"error": "input (9520 tokens) is too large"}')
        # Return unnormalized vector [3.0, 4.0, 0.0, ...] with norm 5.0
        return MockResponse(200, json_data={"embedding": [3.0, 4.0] + [0.0] * 1022})

    monkeypatch.setattr(requests, "post", mock_post)

    long_text = "E" * 10000
    embedder = OllamaEmbedder(base_url="http://mock-ollama:11434", normalize=True)
    result = embedder.encode([long_text])

    assert len(calls) == 2
    norm = float(np.linalg.norm(result[0]))
    assert np.isclose(norm, 1.0, atol=1e-5)
    assert np.isclose(result[0][0], 0.6, atol=1e-5)
    assert np.isclose(result[0][1], 0.8, atol=1e-5)


def test_other_http_errors_raise_immediately(monkeypatch):
    """E2: HTTP 404 or connection failures raise immediately without retry."""
    calls = []

    def mock_post(url, json=None, timeout=None):
        calls.append({"url": url, "json": json, "timeout": timeout})
        return MockResponse(404, text='{"error": "model not found"}')

    monkeypatch.setattr(requests, "post", mock_post)

    embedder = OllamaEmbedder(base_url="http://mock-ollama:11434")

    with pytest.raises(requests.exceptions.HTTPError):
        embedder.encode(["test text"])

    assert len(calls) == 1

def test_e1_multiple_texts_batch_handling(monkeypatch):
    """E1: Batch of texts where one is normal and one exceeds context length."""
    calls = []

    def mock_post(url, json=None, timeout=None):
        calls.append({"url": url, "json": json, "timeout": timeout})
        prompt = json.get("prompt", "")
        if len(prompt) > 5000 and len(calls) == 2:
            return MockResponse(
                500,
                text='{"error": "input (9520 tokens) is too large"}',
            )
        return MockResponse(200, json_data={"embedding": [0.3] * 1024})

    monkeypatch.setattr(requests, "post", mock_post)

    embedder = OllamaEmbedder(base_url="http://mock-ollama:11434")
    texts = ["normal text", "Z" * 10000]
    result = embedder.encode(texts)

    # 1 call for "normal text", 2 calls for "Z" * 10000 = total 3 calls
    assert len(calls) == 3
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 1024)


def test_e4_real_requests_blocked_by_default(assert_real_network_blocked):
    """E4: Verify safety harness blocks any un-mocked requests.post calls."""
    with pytest.raises(AssertionError, match="Real requests.post must never be called"):
        requests.post("http://127.0.0.1:11434/api/embeddings")



def test_context_length_error_body_from_real_ollama_halves_input(monkeypatch):
    """Ollama 0.x real 500 body (2026-11-02) has no token count: input is halved and retried."""
    calls = []

    def mock_post(url, json=None, timeout=None):
        calls.append(json["prompt"])
        if len(calls) == 1:
            return MockResponse(500, text='{"error":"the input length exceeds the context length"}')
        return MockResponse(200, json_data={"embedding": [0.1] * 1024})

    monkeypatch.setattr(requests, "post", mock_post)
    OllamaEmbedder(base_url="http://mock-ollama:11434", model="bge-m3").encode(["B" * 10000])
    assert [len(p) for p in calls] == [10000, 5000]


def test_n1_memory_error_raises_without_truncation(monkeypatch):
    """N1: HTTP 500 'model is too large for available memory' raises HTTPError without truncation."""
    calls = []

    def mock_post(url, json=None, timeout=None):
        calls.append({"url": url, "json": json, "timeout": timeout})
        return MockResponse(500, text='{"error": "model is too large for available memory"}')

    monkeypatch.setattr(requests, "post", mock_post)

    embedder = OllamaEmbedder(base_url="http://mock-ollama:11434")

    with pytest.raises(requests.exceptions.HTTPError):
        embedder.encode(["some text to embed"])

    # Must raise immediately without retry: exactly 1 call
    assert len(calls) == 1
    assert calls[0]["json"]["prompt"] == "some text to embed"


def test_n1_error_field_mismatch_with_context_pattern_in_raw_text_raises_without_truncation(monkeypatch):
    """N1: HTTP 500 where JSON 'error' does not match context error but raw text contains pattern raises HTTPError without truncation."""
    calls = []

    def mock_post(url, json=None, timeout=None):
        calls.append({"url": url, "json": json, "timeout": timeout})
        return MockResponse(
            500,
            text='{"error": "unrelated server failure", "raw_log": "input (9520 tokens) is too large to process"}',
        )

    monkeypatch.setattr(requests, "post", mock_post)

    embedder = OllamaEmbedder(base_url="http://mock-ollama:11434")

    with pytest.raises(requests.exceptions.HTTPError):
        embedder.encode(["some text to embed"])

    # Must raise immediately without retry: exactly 1 call
    assert len(calls) == 1
    assert calls[0]["json"]["prompt"] == "some text to embed"


def test_n2_single_char_context_error_raises_runtime_error_no_empty_prompt(monkeypatch):
    """N2: Single-char input with context length error raises RuntimeError immediately without sending empty prompt."""
    calls = []

    def mock_post(url, json=None, timeout=None):
        calls.append({"url": url, "json": json, "timeout": timeout})
        return MockResponse(500, text='{"error": "the input length exceeds the context length"}')

    monkeypatch.setattr(requests, "post", mock_post)

    embedder = OllamaEmbedder(base_url="http://mock-ollama:11434")

    with pytest.raises(RuntimeError, match="qisqartirish xatosi"):
        embedder.encode(["A"])

    # Exactly 1 call made, no empty prompt ever sent
    assert len(calls) == 1
    assert calls[0]["json"]["prompt"] == "A"
    for call in calls:
        assert call["json"]["prompt"] != "", "Empty prompt must never be sent to Ollama"


@pytest.mark.parametrize("normalize", [False, True])
def test_n2_empty_embedding_response_raises_error(monkeypatch, normalize):
    """N2: Empty embedding response `[]` raises RuntimeError regardless of normalize flag."""
    calls = []

    def mock_post(url, json=None, timeout=None):
        calls.append({"url": url, "json": json, "timeout": timeout})
        return MockResponse(200, json_data={"embedding": []})

    monkeypatch.setattr(requests, "post", mock_post)

    embedder = OllamaEmbedder(base_url="http://mock-ollama:11434", normalize=normalize)

    with pytest.raises(RuntimeError, match="bo'sh embedding"):
        embedder.encode(["valid text"])

    assert len(calls) == 1



def test_non_json_context_error_body_uses_raw_text(monkeypatch):
    """N1: a non-JSON 500 body is matched on raw text; non-context raw text still raises."""
    calls = []

    def mock_post(url, json=None, timeout=None):
        calls.append(json["prompt"])
        if len(calls) == 1:
            return MockResponse(500, text="the input length exceeds the context length")
        return MockResponse(200, json_data={"embedding": [0.1] * 1024})

    monkeypatch.setattr(requests, "post", mock_post)
    OllamaEmbedder(base_url="http://mock-ollama:11434", model="bge-m3").encode(["C" * 1000])
    assert [len(p) for p in calls] == [1000, 500]


def test_empty_text_list_returns_empty_array(monkeypatch):
    """encode([]) keeps returning an empty array without any request."""
    def no_post(*a, **k):
        raise AssertionError("request must not be sent")

    monkeypatch.setattr(requests, "post", no_post)
    assert OllamaEmbedder(base_url="http://mock-ollama:11434", model="bge-m3").encode([]).size == 0
