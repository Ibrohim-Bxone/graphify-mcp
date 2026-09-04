"""Graphify MCP server — local knowledge base for Claude Code.

Tools:
  search_knowledge     - semantic search over the local vector DB (0 API tokens)
  save_memory          - store a decision/summary/note into the DB
  list_memories        - list recently saved memories
  delete_memory        - remove an outdated memory by id
  delete_by_shortcode  - delete all chunks of an Instagram post by shortcode

Everything runs locally (ChromaDB + ONNX MiniLM embeddings). No API keys needed.
"""

import functools
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from db import (
    DB_PATH,
    COLLECTION,
    col as _col,
    client as _client,
    reopen as db_reopen,
    db_retry,
    _open_collection as db_open_collection,
)
# Running cumulative token-savings ESTIMATE (see search_knowledge/save_memory
# below). Kept as a small JSON file next to kg_db/ rather than in Chroma's
# collection metadata, so it never interferes with hnsw config.
STATS_PATH = str(Path(__file__).parent / "token_stats.json")
# Hard floor for search results. The FastMCP instructions below quote this same
# constant, so the tool description can never drift from what the code does.
MIN_SIMILARITY = 0.15
# Claude Code starts the MCP server in the session's working directory,
# so the folder name identifies which project a memory belongs to.
PROJECT = Path(os.getcwd()).name or "unknown"
ALWAYS_OPEN = (
    tuple(x.strip() for x in os.environ["GRAPHIFY_ALWAYS_OPEN"].split(",") if x.strip())
    if "GRAPHIFY_ALWAYS_OPEN" in os.environ
    else ("shared", "Promtlarim")
)

mcp = FastMCP(
    "graphify",
    instructions=(
        "Graphify is the user's persistent project knowledge base. "
        "At the START of a task, call search_knowledge with the task topic to recover "
        "prior decisions, prompts and session summaries instead of asking the user or "
        "re-reading many files. At the END of a substantial session, call save_memory "
        "with a short summary of what was done and decided. "
        f"Ignore results with similarity below {MIN_SIMILARITY}. By DEFAULT search is scoped to "
        "the CURRENT project PLUS the always-open projects (" + ", ".join(ALWAYS_OPEN) + "): "
        "other projects' private memories never leak in, but the shared knowledge base "
        "is reachable from everywhere. Pass project='all' to search every project, or an "
        "exact project name to target one (always-open projects are NOT added in that case). "
        "When saving memories, provide author (or configure GRAPHIFY_AUTHOR) to record who contributed the entry."
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
    # Atomic replace: the bot and the MCP server can both be running, and a
    # half-written file was silently resetting the counters to zero on read.
    tmp = f"{STATS_PATH}.{uuid.uuid4().hex}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(stats, f)
        os.replace(tmp, STATS_PATH)
    except OSError:
        try:
            os.unlink(tmp)
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
    """Kolleksiyani mos embedder bilan ochadi yoki yaratadi (db.py orqali)."""
    return db_open_collection(_client)


def _reopen():
    """Tashqi jarayon bazaga yozganda yoki kesh eskirganda klient va kolleksiyani qayta ochish (db.py orqali)."""
    return db_reopen()


def _db_retry(fn):
    """Kolleksiya eskirib xato berganda klient va kolleksiyani qayta ochib bir marta qayta urinish.

    So'rovdan oldin mtime bo'yicha kesh yangiligini tekshiradi; xatolik yuz berganda zaxira sifatida qayta ochib urinadi (db.py orqali).
    """
    return db_retry(fn)


@mcp.tool()
@_db_retry
def search_knowledge(query: str, top_k: int = 5, kind: str = "", project: str = "current") -> str:
    """Semantic search over the project knowledge base (decisions, prompts, docs, session summaries).

    Args:
        query: what to look for, in natural language (any language).
        top_k: number of results to return (1-10).
        kind: optional filter: "decision", "summary", "note", "prompt" or "doc".
        project: which project to search. Default "current" = this project PLUS the
            always-open projects (ALWAYS_OPEN, default "shared" and "Promtlarim";
            override with the GRAPHIFY_ALWAYS_OPEN env var). Isolation still holds
            for everything else: another project's private memories never leak in,
            while the shared knowledge base stays reachable from every project.
            Pass "all" (or "*") to search every project, or an exact project name to
            target one — an explicit name is taken literally, so the always-open
            projects are NOT added to it.

    Returns:
        Matched entries formatted with kind, title, project, source, date, author (if available),
        similarity score, and snippet.
    """
    top_k = max(1, min(int(top_k), 10))
    filters = []
    if kind:
        filters.append({"kind": kind})
    # Project isolation is the DEFAULT. Only an explicit "all"/"*" disables the
    # project filter; "" and "current" resolve to current project + ALWAYS_OPEN.
    if project not in ("all", "*"):
        if project in ("current", ""):
            projects = [PROJECT] + [p for p in ALWAYS_OPEN if p != PROJECT]
            filters.append({"project": {"$in": projects}})
        else:
            filters.append({"project": project})
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
        if sim < MIN_SIMILARITY:
            continue
        meta = meta or {}
        returned_tokens += _est_tokens(doc)
        author_part = f" | author: {meta.get('author')}" if meta.get("author") else ""
        out.append(
            f"[{meta.get('kind', 'doc')}] {meta.get('title', '(untitled)')} "
            f"| project: {meta.get('project', '?')} | source: {meta.get('source', '?')} "
            f"| date: {meta.get('date', '?')}{author_part} "
            f"| similarity: {sim:.2f} | id: {meta.get('id', '?')}\n{doc}"
        )
    try:
        _record_search_savings(where, returned_tokens)
    except Exception as err:
        print(f"[graphify] Statistika xatosi ({err}), o'tkazib yuborildi.", file=sys.stderr)
    if not out:
        return "No relevant results found."
    return "\n\n---\n\n".join(out)


@mcp.tool()
@_db_retry
def save_memory(content: str, title: str, kind: str = "note", tags: str = "",
                 shared: bool = False, author: str = "") -> str:
    """Save a memory (decision, session summary, note or important prompt) to the knowledge base.

    Args:
        content: the text to remember. Keep it self-contained and concise.
        title: short title for the memory.
        kind: one of "decision", "summary", "note", "prompt".
        tags: optional comma-separated tags.
        shared: pass True when this is reusable knowledge that belongs to no single
            project (a tool choice, a config value, a "best X for Y" fact) rather
            than a project-specific decision. Saves under the first ALWAYS_OPEN
            project (default "shared") so every project's default search finds it
            without needing project='all', instead of the current project — where
            it would only surface for someone who already knew to look here.
        author: optional author/contributor identifier. If empty, falls back to the
            GRAPHIFY_AUTHOR environment variable (or empty string if unset).
    """
    author = str(author or os.environ.get("GRAPHIFY_AUTHOR", "") or "").strip()
    if kind not in ("decision", "summary", "note", "prompt", "doc"):
        kind = "note"
    project = ALWAYS_OPEN[0] if shared and ALWAYS_OPEN else PROJECT
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
            "project": project,
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "author": author,
            "est_tokens": _est_tokens(content),
        }],
    )
    return f"Saved as {mem_id} ({kind}: {title}, project: {project})"


@mcp.tool()
@_db_retry
def list_memories(limit: int = 20) -> str:
    """List the most recently saved memories (not indexed files), newest first."""
    res = _col.get(where={"source": "save_memory"}, include=["metadatas"])
    metas = sorted(res["metadatas"], key=lambda m: (m or {}).get("date", ""), reverse=True)
    if not metas:
        return "No saved memories yet."
    lines = []
    for m in metas[: max(1, int(limit))]:
        m = m or {}
        author_part = f" | author: {m.get('author')}" if m.get("author") else ""
        lines.append(
            f"{m.get('id')} | {m.get('date')} | {m.get('project', '?')}{author_part} | [{m.get('kind')}] {m.get('title')}"
        )
    return "\n".join(lines)


@mcp.tool()
@_db_retry
def delete_memory(memory_id: str) -> str:
    """Delete an outdated or wrong memory by its id (as shown by search_knowledge/list_memories).

    NOTE: If memory_id starts with 'ig:' (e.g. ig:{shortcode}:summary or ig:{shortcode}:item:{i}),
    this deletes ONLY that single chunk. To delete an entire Instagram post and all its chunks,
    use delete_by_shortcode(shortcode) instead.
    """
    _col.delete(ids=[memory_id])
    return f"Deleted {memory_id}"


@mcp.tool()
@_db_retry
def delete_by_shortcode(shortcode: str) -> str:
    """Delete all knowledge base entries associated with an Instagram shortcode (summary and all items).

    Use this tool when you want to completely remove an entire Instagram post from the database.
    Do NOT use delete_memory() for this purpose, as delete_memory() only removes a single chunk
    (e.g., ig:{shortcode}:summary or ig:{shortcode}:item:{i}) and leaves orphaned chunks behind.
    Conversely, use delete_memory() when removing standalone memories (e.g. mem_...) or a specific chunk.

    Args:
        shortcode: The Instagram post shortcode (e.g. 'C123abc' or 'ZZTEST00').

    Returns:
        Confirmation message with the count of deleted entries and a reminder about the archive file.
    """
    shortcode = str(shortcode or "").strip()
    if not shortcode:
        return "Error: shortcode cannot be empty."
    existing = _col.get(where={"shortcode": shortcode}, include=[])
    ids = existing.get("ids", []) if existing else []
    count = len(ids)
    if count == 0:
        return f"No entries found for shortcode '{shortcode}'."
    _col.delete(where={"shortcode": shortcode})
    return (
        f"Deleted {count} entries for shortcode '{shortcode}'. "
        f"Note: Markdown archive file ({shortcode}.md) was not deleted; delete it on the bot side if needed."
    )


@mcp.tool()
@_db_retry
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
