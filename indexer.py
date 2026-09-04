"""Graphify indexer — index documents/notes into the local knowledge base.

Usage:
    python indexer.py <file-or-folder> [more paths...] [--kind doc]

Indexes .md, .txt, .rst files (chunked). Re-running on the same file updates it
(chunks are keyed by path + chunk number). Everything is local, 0 API tokens.
"""

import argparse
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import chromadb
from db import flush_index

DB_PATH = str(Path(__file__).parent / "kg_db")
COLLECTION = "graphify"
EXTS = {".md", ".txt", ".rst"}
CHUNK_SIZE = 1500
OVERLAP = 150


def chunk_text(text: str):
    text = text.strip()
    if not text:
        return
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        # try to break at a paragraph or sentence boundary
        if end < len(text):
            for sep in ("\n\n", "\n", ". "):
                cut = text.rfind(sep, start + CHUNK_SIZE // 2, end)
                if cut != -1:
                    end = cut + len(sep)
                    break
        yield text[start:end].strip()
        if end >= len(text):
            break
        start = end - OVERLAP


def collect_files(paths):
    for p in paths:
        p = Path(p)
        if p.is_file() and p.suffix.lower() in EXTS:
            yield p
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and f.suffix.lower() in EXTS and ".venv" not in f.parts and "kg_db" not in f.parts:
                    yield f
        else:
            print(f"skip (not found or unsupported type): {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="files or folders to index")
    ap.add_argument("--kind", default="doc", help="kind label: doc/prompt/note (default: doc)")
    ap.add_argument("--project", default="shared", help="project label (default: shared = visible everywhere)")
    args = ap.parse_args()

    client = chromadb.PersistentClient(path=DB_PATH)
    col = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    total_chunks = 0
    total_files = 0
    for f in collect_files(args.paths):
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            print(f"skip {f}: {e}")
            continue
        ids, docs, metas = [], [], []
        for i, chunk in enumerate(chunk_text(text)):
            cid = "file_" + hashlib.sha1(f"{f.resolve()}#{i}".encode()).hexdigest()[:16]
            ids.append(cid)
            docs.append(chunk)
            metas.append({
                "id": cid,
                "title": f"{f.name} (part {i + 1})",
                "kind": args.kind,
                "project": args.project,
                "source": str(f),
                "date": today,
                "tags": "",
                "est_tokens": max(1, len(chunk) // 4),  # chars/4 heuristic, see dashboard token-savings stat
            })
        if ids:
            col.upsert(ids=ids, documents=docs, metadatas=metas)
            total_chunks += len(ids)
            total_files += 1
            print(f"indexed: {f} ({len(ids)} chunks)")

    if total_chunks > 0:
        flush_index(col)

    print(f"\nDone. {total_files} files, {total_chunks} chunks. DB now holds {col.count()} chunks total.")


if __name__ == "__main__":
    sys.exit(main())
