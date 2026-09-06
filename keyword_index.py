import sqlite3
import re
import sys
import math
from pathlib import Path
from collections import Counter
import argparse

DB_PATH = Path(__file__).parent / "kg_db" / "keyword_index.sqlite3"

def normalize_text(text: str) -> str:
    text = text.lower()
    # Normalize different apostrophes to a single one
    text = re.sub(r"['`’ʻʼ]", "'", text)
    return text

# Light, conservative Uzbek suffix list for keyword-index tokenization only
# (does not touch the semantic/embedding path). Ordered longest-first so a
# token like "kitobning" strips "ning" and not a shorter false match.
_UZ_SUFFIXES = sorted(
    ["lar", "ning", "ni", "ga", "ka", "qa", "da", "ta", "dan", "tan"],
    key=len,
    reverse=True,
)
_UZ_SUFFIX_MIN_ROOT = 4

def _strip_uz_suffix(token: str) -> str:
    # Cuts at most one (the longest matching) suffix, and only if the
    # remaining root stays at least _UZ_SUFFIX_MIN_ROOT chars, so short
    # words ("bor", "bir", "van") are never mistaken for suffixed forms.
    for suf in _UZ_SUFFIXES:
        if token.endswith(suf) and len(token) - len(suf) >= _UZ_SUFFIX_MIN_ROOT:
            return token[: -len(suf)]
    return token

def tokenize(text: str) -> list[str]:
    text = normalize_text(text)
    # Match alphanumeric including unicode (kirill, latin) and apostrophe inside words
    # Python's \w includes most unicode alphanumeric characters
    # We want to allow apostrophe in the middle of words
    tokens = re.findall(r"\b[\w']+\b", text)
    # Remove leading/trailing apostrophes from tokens if any
    tokens = [t.strip("'") for t in tokens if t.strip("'")]
    return [_strip_uz_suffix(t) for t in tokens]

def _check_fts5(conn) -> bool:
    try:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS _fts5_test USING fts5(x)")
        conn.execute("DROP TABLE _fts5_test")
        return True
    except sqlite3.OperationalError:
        return False

def _get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False, timeout=15.0)
    conn.row_factory = sqlite3.Row
    return conn

class KeywordIndex:
    def __init__(self):
        self.conn = _get_conn()
        self.use_fts5 = _check_fts5(self.conn)
        print(f"[keyword_index] {'FTS5' if self.use_fts5 else 'Pure Python BM25'} ishlatilmoqda", file=sys.stderr)
        self._init_db()

    def _init_db(self):
        with self.conn:
            # Metadata table to store id, project, kind, parent_id
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS docs (
                    id TEXT PRIMARY KEY,
                    project TEXT,
                    kind TEXT,
                    parent_id TEXT,
                    token_count INTEGER,
                    text TEXT
                )
            """)
            if self.use_fts5:
                # the content='docs', content_rowid='id' feature is useful, but id is TEXT.
                # So we just store a separate FTS5 table with id.
                self.conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS fts_idx USING fts5(
                        id UNINDEXED,
                        text,
                        tokenize="unicode61 remove_diacritics 0"
                    )
                """)
            else:
                self.conn.execute("""
                    CREATE TABLE IF NOT EXISTS py_term_freqs (
                        doc_id TEXT,
                        term TEXT,
                        count INTEGER,
                        FOREIGN KEY(doc_id) REFERENCES docs(id) ON DELETE CASCADE
                    )
                """)
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_py_term ON py_term_freqs(term)")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_py_doc ON py_term_freqs(doc_id)")

    def upsert(self, ids: list[str], texts: list[str], metadatas: list[dict]):
        if not ids:
            return
        
        with self.conn:
            for doc_id, text, meta in zip(ids, texts, metadatas):
                meta = meta or {}
                project = meta.get("project", "")
                kind = meta.get("kind", "")
                parent_id = meta.get("parent_id", "")
                if not parent_id and "shortcode" in meta:
                    parent_id = meta["shortcode"]
                if not parent_id and "source" in meta:
                    parent_id = meta["source"]
                
                # Delete existing
                self.conn.execute("DELETE FROM docs WHERE id = ?", (doc_id,))
                if self.use_fts5:
                    self.conn.execute("DELETE FROM fts_idx WHERE id = ?", (doc_id,))
                else:
                    self.conn.execute("DELETE FROM py_term_freqs WHERE doc_id = ?", (doc_id,))

                # Insert new
                tokens = tokenize(text)
                token_count = len(tokens)
                
                self.conn.execute(
                    "INSERT INTO docs (id, project, kind, parent_id, token_count, text) VALUES (?, ?, ?, ?, ?, ?)",
                    (doc_id, project, kind, parent_id, token_count, text)
                )

                if self.use_fts5:
                    # Feed normalized tokens space separated
                    normalized_text = " ".join(tokens)
                    self.conn.execute(
                        "INSERT INTO fts_idx (id, text) VALUES (?, ?)",
                        (doc_id, normalized_text)
                    )
                else:
                    freqs = Counter(tokens)
                    self.conn.executemany(
                        "INSERT INTO py_term_freqs (doc_id, term, count) VALUES (?, ?, ?)",
                        [(doc_id, term, count) for term, count in freqs.items()]
                    )

    def delete(self, ids: list[str]):
        if not ids:
            return
        with self.conn:
            for doc_id in ids:
                self.conn.execute("DELETE FROM docs WHERE id = ?", (doc_id,))
                if self.use_fts5:
                    self.conn.execute("DELETE FROM fts_idx WHERE id = ?", (doc_id,))
                else:
                    self.conn.execute("DELETE FROM py_term_freqs WHERE doc_id = ?", (doc_id,))

    def search(self, query: str, limit: int = 50, filters: dict = None) -> list[dict]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        total_docs = self.conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
        if total_docs == 0:
            print("[keyword_index] Ogohlantirish: indeks bo'sh (0 hujjat), kalit so'z qidiruvi natija qaytarmaydi.", file=sys.stderr)
            return []

        filters = filters or {}
        project = filters.get("project")
        kind = filters.get("kind")
        
        # Build filter SQL
        filter_sql = ""
        filter_params = []
        if project and project not in ("all", "*"):
            # Graphify/server.py handles `project` filtering as $in or exact.
            if isinstance(project, dict) and "$in" in project:
                placeholders = ",".join("?" * len(project["$in"]))
                filter_sql += f" AND d.project IN ({placeholders})"
                filter_params.extend(project["$in"])
            else:
                filter_sql += " AND d.project = ?"
                filter_params.append(project)
                
        if kind:
            filter_sql += " AND d.kind = ?"
            filter_params.append(kind)
            
        if self.use_fts5:
            # Query fts_idx directly. OR (not the FTS5 default AND) so results
            # match the pure-Python BM25 path, which scores any doc containing
            # at least one query term.
            fts_query = " OR ".join(f'"{t}"' for t in query_tokens)
            sql = f"""
                SELECT d.id, d.text, d.parent_id, d.project, d.kind, f.rank as score
                FROM fts_idx f
                JOIN docs d ON d.id = f.id
                WHERE f.text MATCH ? {filter_sql}
                ORDER BY f.rank ASC
                LIMIT ?
            """
            params = [fts_query] + filter_params + [limit]
            
            try:
                rows = self.conn.execute(sql, params).fetchall()
            except sqlite3.OperationalError:
                # Fallback if match syntax is invalid
                return []
            
            results = []
            for row in rows:
                results.append({
                    "id": row["id"],
                    "text": row["text"],
                    "parent_id": row["parent_id"],
                    "project": row["project"],
                    "kind": row["kind"],
                    "score": row["score"]  # Note: FTS5 bm25 is negative, so smaller is better.
                })
            return results
        else:
            # Pure Python BM25
            N = self.conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
            if N == 0:
                return []
                
            avgdl = self.conn.execute("SELECT AVG(token_count) FROM docs").fetchone()[0] or 1.0
            
            k1 = 1.2
            b = 0.75
            
            # Fetch doc frequencies for query terms
            doc_freqs = {}
            for t in set(query_tokens):
                c = self.conn.execute("SELECT COUNT(DISTINCT doc_id) FROM py_term_freqs WHERE term = ?", (t,)).fetchone()[0]
                doc_freqs[t] = c
                
            scores = Counter()
            
            for t in set(query_tokens):
                n_t = doc_freqs.get(t, 0)
                if n_t == 0:
                    continue
                # IDF
                idf = math.log((N - n_t + 0.5) / (n_t + 0.5) + 1)
                
                # Fetch all docs containing term that match filters
                # We do filtering after fetching due to simplicity, or we can do it in SQL
                sql = f"""
                    SELECT d.id, d.token_count, p.count
                    FROM py_term_freqs p
                    JOIN docs d ON d.id = p.doc_id
                    WHERE p.term = ? {filter_sql}
                """
                params = [t] + filter_params
                
                for row in self.conn.execute(sql, params):
                    doc_id = row["id"]
                    dl = row["token_count"]
                    tf = row["count"]
                    
                    term_score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (dl / avgdl)))
                    # To match FTS5 "smaller is better" behavior, we negate the positive score
                    scores[doc_id] -= term_score
                    
            if not scores:
                return []
                
            top_ids = [doc_id for doc_id, _ in scores.most_common()[:-limit-1:-1]] # most_common but smallest first
            # but most_common gives largest first, we want smallest first (most negative)
            sorted_scores = sorted(scores.items(), key=lambda x: x[1])[:limit]
            
            results = []
            for doc_id, score in sorted_scores:
                row = self.conn.execute("SELECT id, text, parent_id, project, kind FROM docs WHERE id = ?", (doc_id,)).fetchone()
                results.append({
                    "id": row["id"],
                    "text": row["text"],
                    "parent_id": row["parent_id"],
                    "project": row["project"],
                    "kind": row["kind"],
                    "score": score
                })
            return results

_index_singleton = None

def get_index():
    global _index_singleton
    if _index_singleton is None:
        _index_singleton = KeywordIndex()
    return _index_singleton

def rebuild():
    print("Kalit so'z indeksini qayta qurish (bu biroz vaqt olishi mumkin)...")
    import chromadb
    client = chromadb.PersistentClient(path=str(DB_PATH.parent.parent / "kg_db"))
    col = client.get_collection("graphify")
    
    idx = KeywordIndex()
    with idx.conn:
        idx.conn.execute("DROP TABLE IF EXISTS docs")
        idx.conn.execute("DROP TABLE IF EXISTS fts_idx")
        idx.conn.execute("DROP TABLE IF EXISTS py_term_freqs")
    idx._init_db()
    
    # Process in batches
    data = col.get(include=["documents", "metadatas"])
    ids = data["ids"]
    docs = data["documents"]
    metas = data["metadatas"]
    
    batch_size = 500
    for i in range(0, len(ids), batch_size):
        idx.upsert(
            ids[i:i+batch_size], 
            docs[i:i+batch_size], 
            metas[i:i+batch_size]
        )
        print(f"{i + len(ids[i:i+batch_size])}/{len(ids)} indekslandi...")
    print("Tayyor.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qayta-qur", action="store_true")
    args = parser.parse_args()
    if args.qayta_qur:
        rebuild()
    else:
        parser.print_help()
