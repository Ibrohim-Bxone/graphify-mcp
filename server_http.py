"""Graphify — remote MCP entrypoint (streamable HTTP).

Bu fayl server.py dagi asboblarni O'ZGARTIRMAYDI: aynan o'sha FastMCP
namunasini oladi va uni stdio o'rniga HTTP orqali chiqaradi. ChatGPT
(Developer mode custom connector) faqat public HTTPS + streamable HTTP
bilan ishlaydi, shuning uchun stdio varianti u yerda yaramaydi.

Xavfsizlik: tunnel ochiq internetda turadi va ChatGPT "No authentication"
rejimida hech qanday header yubormaydi. Shuning uchun sir URL yo'lining
o'zida bo'ladi: /mcp/<token>. Token .http-token faylida saqlanadi
(git'ga tushmasin).

Ishga tushirish:
    .venv\\Scripts\\python.exe server_http.py            # 127.0.0.1:8787
    .venv\\Scripts\\python.exe server_http.py --port 9000
"""
import argparse
import json
import os
import secrets
from pathlib import Path

HERE = Path(__file__).parent
TOKEN_FILE = HERE / ".http-token"


def _token() -> str:
    tok = os.environ.get("GRAPHIFY_HTTP_TOKEN", "").strip()
    if tok:
        return tok
    if TOKEN_FILE.exists():
        tok = TOKEN_FILE.read_text(encoding="utf-8").strip()
        if tok:
            return tok
    tok = secrets.token_urlsafe(24)
    TOKEN_FILE.write_text(tok, encoding="utf-8")
    return tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8787)
    args = ap.parse_args()

    import server  # server.py dagi mcp namunasi, 6 ta asbob, va similarity moduli chaqiruvi

    mcp = server.mcp
    token = _token()

    # ChatGPT ulanishida loyiha konteksti yo'q — cwd'dan aniqlanadigan
    # PROJECT bu yerda ma'nosiz. Shuning uchun search/fetch aliaslari
    # doim project="all" bilan ishlaydi. Bu ataylab qilingan, sababi 
    # global qidiruv orqali foydalanuvchining barcha loyihalaridagi 
    # bilimlari chatda mavjud bo'lishi kerak.
    # (search/fetch nomlari ataylab: ChatGPT connector'lari aynan shu ikkitasini kutadi.)
    @mcp.tool()
    def search(query: str) -> str:
        """Search the user's Graphify knowledge base across ALL projects.
        Returns matched entries with id, title, project, date and snippet.
        Note: The project parameter is intentionally hardcoded to 'all' to 
        allow global retrieval across the user's entire knowledge base."""
        return server.search_knowledge(query=query, top_k=8, project="all")

    @mcp.tool()
    def fetch(id: str) -> str:
        """Fetch the full text of one Graphify entry by the id returned by search."""
        from db import get_collection
        res = get_collection().get(ids=[id], include=["documents", "metadatas"])
        docs = res.get("documents") or []
        if not docs:
            return f"Topilmadi: {id}"
        meta = (res.get("metadatas") or [{}])[0] or {}
        return json.dumps(
            {"id": id, "title": meta.get("title", ""), "project": meta.get("project", ""),
             "date": meta.get("date", ""), "text": docs[0]},
            ensure_ascii=False, indent=1,
        )

    mcp.settings.host = args.host
    mcp.settings.port = args.port
    mcp.settings.streamable_http_path = f"/mcp/{token}"

    print(f"Graphify MCP (streamable-http): http://{args.host}:{args.port}/mcp/{token}")
    print(f"Token fayl: {TOKEN_FILE}")
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
