"""Graphify MCP server — local knowledge base for Claude Code.

Tools:
  search_knowledge  - semantic search over the local vector DB (0 API tokens)
  save_memory       - store a decision/summary/note into the DB
  list_memories     - list recently saved memories
  delete_memory     - remove an outdated memory by id

Everything runs locally (ChromaDB + ONNX MiniLM embeddings). No API keys needed.
"""

import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from mcp.server.fastmcp import FastMCP

from embedders import build_chroma_embedding_function

DB_PATH = str(Path(__file__).parent / "kg_db")
COLLECTION = "graphify"
# Running cumulative token-savings ESTIMATE (see search_knowledge/save_memory
# below). Kept as a small JSON file next to kg_db/ rather than in Chroma's
# collection metadata, so it never interferes with hnsw config.
STATS_PATH = str(Path(__file__).parent / "token_stats.json")
# Claude Code starts the MCP server in the session's working directory,
# so the folder name identifies which project a memory belongs to.
PROJECT = Path(os.getcwd()).name or "unknown"

mcp = FastMCP(
    "graphify",
    instructions=(
        "Graphify is the user's persistent project knowledge base. "
        "At the START of a task, call search_knowledge with the task topic to recover "
        "prior decisions, prompts and session summaries instead of asking the user or "
        "re-reading many files. At the END of a substantial session, call save_memory "
        "with a short summary of what was done and decided. "
        "Ignore results with similarity below 0.3. By DEFAULT search is scoped to "
        "the CURRENT project only (isolation) — never surfaces other projects' "
        "memories unless the user explicitly asks, in which case pass project='all'."
    ),
)


def _est_tokens(text: str) -> int:
    """Rough chars/4 heuristic for token count. Not a real tokenizer — this is
    only ever used for the dashboard's "taxminiy" (estimated) token-savings stat."""
    return max(1, len(text) // 4)


def _load_token_stats() -> dict:
    try:
        with open(STATS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "would_have_tokens": int(data.get("would_have_tokens", 0)),
            "actual_tokens": int(data.get("actual_tokens", 0)),
        }
    except (OSError, ValueError):
        return {"would_have_tokens": 0, "actual_tokens": 0}


def _save_token_stats(stats: dict) -> None:
    try:
        with open(STATS_PATH, "w", encoding="utf-8") as f:
            json.dump(stats, f)
    except OSError:
        pass


def _record_search_savings(where, returned_tokens: int) -> None:
    """Update the running would-have-vs-actual token counters (ESTIMATE only).

    "would_have" = cost of reading every candidate in the searched scope from
    scratch (the naive alternative to semantic search: grep/open every file
    that could be relevant). "actual" = tokens of what search_knowledge
    actually returned this call. The gap between the two is the savings
    surfaced on the dashboard.
    """
    pool = _col.get(where=where, include=["documents", "metadatas"])
    would_have = sum(
        (meta or {}).get("est_tokens") or _est_tokens(doc)
        for doc, meta in zip(pool["documents"], pool["metadatas"])
    )
    if would_have <= 0:
        return
    stats = _load_token_stats()
    stats["would_have_tokens"] += would_have
    stats["actual_tokens"] += returned_tokens
    _save_token_stats(stats)


def _open_collection():
    """Open (or create) the collection with the currently configured embedder.

    Falls back to the collection's existing (persisted) embedder if the
    configured one conflicts with it — e.g. the collection already has data
    embedded with a different model, which chromadb itself refuses to mix —
    or if it fails to initialize (missing dependency/API key). A bad
    models_config.json must never stop the MCP server from starting.
    """
    embed_fn = build_chroma_embedding_function()
    kwargs = {"metadata": {"hnsw:space": "cosine"}}
    if embed_fn is not None:
        kwargs["embedding_function"] = embed_fn
    try:
        return _client.get_or_create_collection(COLLECTION, **kwargs)
    except ValueError as e:
        if embed_fn is not None:
            print(
                f"[graphify] Ogohlantirish: models_config.json'dagi embedder mavjud "
                f"bazadagi bilan mos kelmadi ({e}); saqlangan embedderga qaytildi. "
                f"Almashtirish uchun kg_db/ ni tozalab qaytadan indekslang.",
                file=sys.stderr,
            )
            return _client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
        raise


_client = chromadb.PersistentClient(path=DB_PATH)
_col = _open_collection()


@mcp.tool()
def search_knowledge(query: str, top_k: int = 5, kind: str = "", project: str = "current") -> str:
    """Semantic search over the project knowledge base (decisions, prompts, docs, session summaries).

    Args:
        query: what to look for, in natural language (any language).
        top_k: number of results to return (1-10).
        kind: optional filter: "decision", "summary", "note", "prompt" or "doc".
        project: which project to search. Default "current" = ONLY this project.
            Isolation is the default on purpose: one project's memories never leak
            into another. Pass "all" (or "*") to search every project, or an exact
            project name to target a specific one. Cross-project search is opt-in.
    """
    top_k = max(1, min(int(top_k), 10))
    filters = []
    if kind:
        filters.append({"kind": kind})
    # Project isolation is the DEFAULT. Only an explicit "all"/"*" disables the
    # project filter; "" and "current" both resolve to the current project.
    if project not in ("all", "*"):
        filters.append({"project": PROJECT if project in ("current", "") else project})
    where = filters[0] if len(filters) == 1 else ({"$and": filters} if filters else None)
    if _col.count() == 0:
        return "Knowledge base is empty. Nothing indexed yet."
    res = _col.query(
        query_texts=[query],
        n_results=min(top_k, _col.count()),
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    out = []
    returned_tokens = 0
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        sim = 1.0 - dist
        if sim < 0.15:
            continue
        meta = meta or {}
        returned_tokens += _est_tokens(doc)
        out.append(
            f"[{meta.get('kind', 'doc')}] {meta.get('title', '(untitled)')} "
            f"| project: {meta.get('project', '?')} | source: {meta.get('source', '?')} "
            f"| date: {meta.get('date', '?')} "
            f"| similarity: {sim:.2f} | id: {meta.get('id', '?')}\n{doc}"
        )
    _record_search_savings(where, returned_tokens)
    if not out:
        return "No relevant results found."
    return "\n\n---\n\n".join(out)


@mcp.tool()
def save_memory(content: str, title: str, kind: str = "note", tags: str = "") -> str:
    """Save a memory (decision, session summary, note or important prompt) to the knowledge base.

    Args:
        content: the text to remember. Keep it self-contained and concise.
        title: short title for the memory.
        kind: one of "decision", "summary", "note", "prompt".
        tags: optional comma-separated tags.
    """
    if kind not in ("decision", "summary", "note", "prompt", "doc"):
        kind = "note"
    mem_id = f"mem_{uuid.uuid4().hex[:12]}"
    _col.add(
        ids=[mem_id],
        documents=[content],
        metadatas=[{
            "id": mem_id,
            "title": title,
            "kind": kind,
            "tags": tags,
            "source": "save_memory",
            "project": PROJECT,
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "est_tokens": _est_tokens(content),
        }],
    )
    return f"Saved as {mem_id} ({kind}: {title}, project: {PROJECT})"


@mcp.tool()
def list_memories(limit: int = 20) -> str:
    """List the most recently saved memories (not indexed files), newest first."""
    res = _col.get(where={"source": "save_memory"}, include=["metadatas"])
    metas = sorted(res["metadatas"], key=lambda m: m.get("date", ""), reverse=True)
    if not metas:
        return "No saved memories yet."
    lines = [
        f"{m.get('id')} | {m.get('date')} | {m.get('project', '?')} | [{m.get('kind')}] {m.get('title')}"
        for m in metas[: max(1, int(limit))]
    ]
    return "\n".join(lines)


@mcp.tool()
def delete_memory(memory_id: str) -> str:
    """Delete an outdated or wrong memory by its id (as shown by search_knowledge/list_memories)."""
    _col.delete(ids=[memory_id])
    return f"Deleted {memory_id}"


@mcp.tool()
def token_stats() -> str:
    """Show the running token-savings stats for this Graphify knowledge base: how many
    tokens search_knowledge calls have saved versus Claude re-reading full matched content
    from scratch each time. This is an ESTIMATE (chars/4 heuristic), not an exact measurement.
    Same numbers shown on the dashboard's topbar/sidebar stat card."""
    stats = _load_token_stats()
    would = stats["would_have_tokens"]
    actual = stats["actual_tokens"]
    if would <= 0:
        return "Hali statistika yo'q — search_knowledge hali hech chaqirilmagan."
    pct = round((1 - actual / would) * 100)
    return (
        f"Token tejaldi: ~{pct}% (taxminiy)\n"
        f"Agar Claude har safar mos kelgan kontentni to'liq qayta o'qiganda ketadigan token: ~{would}\n"
        f"search_knowledge orqali haqiqatda ishlatilgan token: ~{actual}"
    )


if __name__ == "__main__":
    mcp.run()
