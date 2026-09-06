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


import re
from chunking import chunk_by_tokens

def has_shortcode_frontmatter(text: str) -> bool:
    if not text.startswith("---"):
        return False
    end = text.find("\n---", 3)
    if end == -1:
        return False
    frontmatter = text[3:end]
    return bool(re.search(r'^shortcode:', frontmatter, re.MULTILINE))

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
    ap.add_argument("--dry-run", action="store_true", help="do not write to db, just count")
    args = ap.parse_args()

    client = chromadb.PersistentClient(path=DB_PATH)
    col = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    total_chunks = 0
    total_files = 0
    skipped_archive_files = 0
    long_chunks_count = 0
    
    for f in collect_files(args.paths):
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            print(f"skip {f}: {e}")
            continue
            
        if has_shortcode_frontmatter(text):
            skipped_archive_files += 1
            print(f"arxiv — o'tkazib yuborildi: {f}")
            continue
            
        ids, docs, metas = [], [], []
        parent_id = str(f.resolve())
        chunks = chunk_by_tokens(text, parent_id=parent_id)
        
        for c in chunks:
            chunk = c["text"]
            i = c["chunk_index"]
            cid = "file_" + hashlib.sha1(f"{parent_id}#{i}".encode()).hexdigest()[:16]
            ids.append(cid)
            docs.append(chunk)
            
            if c["token_count"] > 256:
                long_chunks_count += 1
                
            metas.append({
                "id": cid,
                "title": f"{f.name} (part {i + 1})",
                "kind": args.kind,
                "project": args.project,
                "source": str(f),
                "date": today,
                "tags": "",
                "est_tokens": max(1, len(chunk) // 4),  # chars/4 heuristic, see dashboard token-savings stat
                "parent_id": parent_id,
                "chunk_index": i,
                "chunk_total": c["chunk_total"],
                "token_count": c["token_count"]
            })
            
        if ids:
            if not args.dry_run:
                # 1. Get existing IDs for this file
                res = col.get(where={"source": str(f)}, include=[])
                existing_ids = res.get("ids", []) if res else []
                
                # 2. Upsert new/updated chunks
                col.upsert(ids=ids, documents=docs, metadatas=metas)
                
                import keyword_index
                keyword_index.get_index().upsert(ids=ids, texts=docs, metadatas=metas)
                
                # 3. Delete orphans
                expected_ids = set(ids)
                to_delete = [i for i in existing_ids if i not in expected_ids]
                if to_delete:
                    col.delete(ids=to_delete)
                    keyword_index.get_index().delete(ids=to_delete)
                    
            total_chunks += len(ids)
            total_files += 1
            print(f"indexed: {f} ({len(ids)} chunks)")

    if total_chunks > 0 and not args.dry_run:
        flush_index(col)

    print(f"\nDone. Indekslangan fayl: {total_files}, O'tkazib yuborilgan arxiv fayl: {skipped_archive_files}, Jami chunk: {total_chunks}, 256 tokendan uzun chunk soni: {long_chunks_count}.")
    if not args.dry_run:
        print(f"DB now holds {col.count()} chunks total.")


if __name__ == "__main__":
    sys.exit(main())
