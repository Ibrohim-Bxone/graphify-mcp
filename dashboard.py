"""Graphify Web Dashboard — ro'yxat ko'rinishi + semantik graf ko'rinishi.

Graf ko'rinishdagi qirralar (bog'lanishlar) qo'lda chizilmaydi — ular embedding
o'xshashligidan avtomatik hisoblanadi, shu jumladan turli loyihalar orasidagi
kutilmagan bog'lanishlar ham ko'rinadi.
"""

import html as html_lib
import json
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, redirect, request
import chromadb

from embedders import BACKENDS, load_models_config, save_models_config

DB_PATH = str(Path(__file__).parent / "kg_db")
COLLECTION = "graphify"
# Running token-savings counters written by server.py's search_knowledge (see
# there for the estimation method). Read-only here.
STATS_PATH = str(Path(__file__).parent / "token_stats.json")

# Har tugundan nechta eng yaqin qo'shnisi qirra sifatida ko'rsatiladi (shovqinni cheklash uchun)
TOP_K_NEIGHBORS = 6
# Bundan past o'xshashlikdagi qirralar chizilmaydi
MIN_EDGE_SIMILARITY = 0.15

KIND_LABELS = {'decision': 'Qaror', 'summary': 'Xulosa', 'note': 'Eslatma', 'doc': 'Hujjat'}
KIND_COLORS = {'decision': '28a745', 'summary': '17a2b8', 'note': 'ffc107', 'doc': '6c757d'}
KIND_BG = {'decision': 'd4edda', 'summary': 'd1ecf1', 'note': 'fff3cd', 'doc': 'e2e3e5'}
KIND_TEXT = {'decision': '155724', 'summary': '0c5460', 'note': '856404', 'doc': '383d41'}

# Loyihalarni graf ko'rinishida ajratish uchun rang palitrasi (kind ranglaridan mustaqil).
PROJECT_PALETTE = ['e83e8c', 'fd7e14', '6610f2', '20c997', '0d6efd', 'd63384', '198754', 'cfa100']

app = Flask(__name__)
client = chromadb.PersistentClient(path=DB_PATH)
col = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def esc(value) -> str:
    """Foydalanuvchi kiritgan matnni xavfsiz HTML uchun escape qiladi."""
    return html_lib.escape(str(value), quote=True)


def safe_kind(kind) -> str:
    """CSS klass/attribute nomi sifatida ishlatish uchun kind qiymatini cheklaydi."""
    return kind if kind in KIND_LABELS else 'note'


def get_all_memories():
    res = col.get(include=['metadatas', 'documents'])
    memories = []
    for id_, meta, doc in zip(res['ids'], res['metadatas'], res['documents']):
        memories.append({
            'id': id_,
            'title': meta.get('title', '(untitled)'),
            'kind': meta.get('kind', 'note'),
            'project': meta.get('project', '?'),
            'date': meta.get('date', '?'),
            'tags': meta.get('tags', ''),
            'content': doc,
        })
    return sorted(memories, key=lambda m: m['date'], reverse=True)


def get_projects(memories):
    return sorted(set(m['project'] for m in memories))


def load_token_stats():
    """Read the running token-savings counters written by server.py (ESTIMATE only)."""
    try:
        with open(STATS_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (OSError, ValueError):
        data = {}
    would = int(data.get('would_have_tokens', 0))
    actual = int(data.get('actual_tokens', 0))
    pct = round((1 - actual / would) * 100) if would > 0 else None
    return {'would': would, 'actual': actual, 'pct': pct}


@app.route('/')
def index():
    memories = get_all_memories()
    projects = get_projects(memories)
    project_filter = request.args.get('project', '')
    search_query = request.args.get('q', '')
    kind_filter = request.args.get('kind', '')
    token_stats = load_token_stats()
    token_pct_label = f"~{token_stats['pct']}%" if token_stats['pct'] is not None else '—'

    filtered = memories
    if project_filter:
        filtered = [m for m in filtered if m['project'] == project_filter]
    if search_query:
        q_lower = search_query.lower()
        filtered = [m for m in filtered if q_lower in m['title'].lower() or q_lower in m['content'].lower()]
    if kind_filter:
        filtered = [m for m in filtered if m['kind'] == kind_filter]

    selected = None
    if request.args.get('id'):
        selected = next((m for m in memories if m['id'] == request.args.get('id')), None)

    project_opts = ''.join(
        f'<option value="{esc(p)}" {"selected" if p == project_filter else ""}>{esc(p)}</option>' for p in projects
    )
    kind_opts = ''.join(
        f'<option value="{k}" {"selected" if k == kind_filter else ""}>{esc(v)}</option>'
        for k, v in KIND_LABELS.items()
    )

    if filtered:
        memory_list = ''.join(
            f'<a href="?{"project=" + esc(project_filter) + "&" if project_filter else ""}id={esc(m["id"])}" '
            f'style="text-decoration:none;color:inherit;">'
            f'<div class="memory-card {"active" if selected and selected["id"] == m["id"] else ""}">'
            f'<div class="memory-title"><span class="badge badge-{esc(safe_kind(m["kind"]))}">{esc(KIND_LABELS.get(m["kind"], m["kind"]))}</span>{esc(m["title"])}</div>'
            f'<div class="memory-meta"><strong>{esc(m["project"])}</strong> • {esc(m["date"])}</div>'
            f'</div></a>'
            for m in filtered
        )
    else:
        memory_list = '<div class="empty">Hech narsa topilmadi</div>'

    if selected:
        detail_html = f'''<div class="detail">
            <h2>{esc(selected["title"])}</h2>
            <div class="detail-meta">
                <span class="badge badge-{esc(safe_kind(selected["kind"]))}">{esc(KIND_LABELS.get(selected["kind"], selected["kind"]))}</span>
                {esc(selected["project"])} • {esc(selected["date"])}
                {" • " + esc(selected["tags"]) if selected["tags"] else ""}
            </div>
            <div class="detail-content">{esc(selected["content"])}</div>
            <form method="post" action="/delete" style="display:inline;">
                <input type="hidden" name="id" value="{esc(selected["id"])}">
                <button class="btn-delete" type="submit" onclick="return confirm('Haqiqatdan ham o\\'chirmisiz?')">O'chirish</button>
            </form>
        </div>'''
    else:
        detail_html = '<div class="empty">Xotirini tanlang</div>'

    badge_css = '\n'.join(
        f'.badge-{k} {{ background: #{KIND_BG[k]}; color: #{KIND_TEXT[k]}; }}' for k in KIND_LABELS
    )
    legend_items = ''.join(
        f'<span class="legend-item"><span class="legend-dot" style="background:#{KIND_COLORS[k]}"></span>{esc(v)}</span>'
        for k, v in KIND_LABELS.items()
    )

    project_colors = {p: PROJECT_PALETTE[i % len(PROJECT_PALETTE)] for i, p in enumerate(projects)}
    project_filter_items = ''.join(
        f'<label class="project-chip"><input type="checkbox" class="project-toggle" value="{esc(p)}" checked>'
        f'<span class="project-dot" style="background:#{project_colors[p]}"></span>{esc(p)}</label>'
        for p in projects
    )
    project_colors_json = json.dumps(project_colors)

    # --- Modellar tab ---
    models_cfg = load_models_config()
    active_backend = models_cfg['embedder_type'] if models_cfg['embedder_type'] in BACKENDS else 'local'
    doc_count = col.count()
    models_saved = request.args.get('saved') == '1'

    def backend_option_html(key, info):
        checked = 'checked' if key == active_backend else ''
        sub = models_cfg['config'].get(key, {})
        fields_html = ''
        if key == 'openai':
            fields_html = (
                f'<input type="password" name="openai_api_key" placeholder="OpenAI API key" value="{esc(sub.get("api_key", ""))}">'
                f'<input type="text" name="openai_model" placeholder="model (masalan text-embedding-3-small)" value="{esc(sub.get("model", "text-embedding-3-small"))}">'
            )
        elif key == 'ollama':
            fields_html = (
                f'<input type="text" name="ollama_base_url" placeholder="http://localhost:11434" value="{esc(sub.get("base_url", "http://localhost:11434"))}">'
                f'<input type="text" name="ollama_model" placeholder="model (masalan nomic-embed-text)" value="{esc(sub.get("model", "nomic-embed-text"))}">'
            )
        elif key == 'custom':
            fields_html = (
                f'<input type="text" name="custom_base_url" placeholder="https://api.masalan.com" value="{esc(sub.get("base_url", ""))}">'
                f'<input type="password" name="custom_api_key" placeholder="API key (ixtiyoriy)" value="{esc(sub.get("api_key", ""))}">'
                f'<input type="text" name="custom_model" placeholder="model nomi (ixtiyoriy)" value="{esc(sub.get("model", ""))}">'
            )
        active_tag = ' <span class="model-active-badge">faol</span>' if key == active_backend else ''
        show_fields = 'block' if (fields_html and key == active_backend) else 'none'
        return f'''<label class="model-option">
            <div class="model-option-head">
                <input type="radio" name="embedder_type" value="{key}" {checked} onchange="toggleModelFields()">
                <strong>{esc(info["label"])}</strong>{active_tag}
                <span class="model-cost">{esc(info["cost"])}</span>
            </div>
            <div class="model-desc">{esc(info["desc"])}</div>
            <div class="model-fields" data-key="{key}" style="display:{show_fields};">{fields_html}</div>
        </label>'''

    model_options_html = ''.join(backend_option_html(k, v) for k, v in BACKENDS.items())

    html = f"""<!DOCTYPE html>
<html lang="uz">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Graphify Dashboard</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .app {{
            max-width: 1280px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .topbar {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 22px 30px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 14px;
        }}
        .brand {{ display: flex; align-items: baseline; gap: 10px; }}
        .brand h1 {{ font-size: 1.7em; }}
        .brand span {{ opacity: 0.85; font-size: 0.9em; }}
        .tabs {{ display: flex; gap: 6px; background: rgba(255,255,255,0.15); padding: 4px; border-radius: 8px; }}
        .tab-btn {{
            background: transparent;
            border: none;
            color: white;
            padding: 8px 18px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9em;
            font-weight: 500;
            width: auto;
            transition: 0.2s;
        }}
        .tab-btn:hover {{ background: rgba(255,255,255,0.12); }}
        .tab-btn.active {{ background: white; color: #667eea; }}
        .topbar-stats {{ font-size: 0.85em; opacity: 0.9; }}
        .view-list {{
            display: grid;
            grid-template-columns: 350px 1fr;
            min-height: 620px;
        }}
        @media (max-width: 900px) {{
            .view-list {{ grid-template-columns: 1fr; }}
        }}
        .sidebar {{
            background: #f8f9fa;
            border-right: 1px solid #dee2e6;
            padding: 20px;
            overflow-y: auto;
            max-height: 620px;
        }}
        .main {{ padding: 20px; overflow-y: auto; max-height: 620px; }}
        .sidebar h2 {{ font-size: 1.05em; margin-bottom: 12px; color: #495057; }}
        input, select {{
            width: 100%;
            padding: 10px;
            margin-bottom: 10px;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            font-size: 0.95em;
        }}
        button {{
            width: 100%;
            padding: 10px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.95em;
            transition: 0.3s;
        }}
        button:hover {{ background: #764ba2; }}
        .memory-card {{
            background: white;
            border-left: 4px solid #667eea;
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 4px;
            cursor: pointer;
            transition: 0.2s;
        }}
        .memory-card:hover {{ box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .memory-card.active {{ background: #e7e5ff; border-left-color: #764ba2; }}
        .memory-title {{ font-weight: bold; color: #212529; font-size: 0.95em; }}
        .memory-meta {{ font-size: 0.8em; color: #6c757d; margin-top: 3px; }}
        .badge {{ display: inline-block; padding: 2px 6px; border-radius: 10px; font-size: 0.7em; margin-right: 4px; }}
        {badge_css}
        .detail {{ background: white; border: 1px solid #dee2e6; border-radius: 6px; padding: 20px; }}
        .detail h2 {{ margin-bottom: 10px; }}
        .detail-meta {{ font-size: 0.9em; color: #6c757d; margin-bottom: 15px; padding-bottom: 10px; border-bottom: 1px solid #e9ecef; }}
        .detail-content {{ white-space: pre-wrap; line-height: 1.6; font-size: 0.95em; }}
        .btn-delete {{ background: #dc3545; width: auto; padding: 8px 16px; margin-top: 15px; }}
        .btn-delete:hover {{ background: #c82333; }}
        .empty {{ color: #6c757d; text-align: center; padding: 40px 20px; }}
        .stats {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 15px; }}
        .stat {{ background: white; border: 1px solid #dee2e6; border-radius: 6px; padding: 10px; text-align: center; }}
        .stat-num {{ font-size: 1.5em; font-weight: bold; color: #667eea; }}
        .stat-label {{ font-size: 0.8em; color: #6c757d; }}
        form {{ display: flex; gap: 5px; }}
        form input {{ flex: 1; margin-bottom: 0; }}
        form button {{ width: auto; padding: 10px 15px; }}

        /* Graf ko'rinish */
        .view-graph {{ display: none; }}
        .graph-toolbar {{
            display: flex;
            align-items: center;
            gap: 20px;
            padding: 14px 20px;
            border-bottom: 1px solid #dee2e6;
            background: #f8f9fa;
            flex-wrap: wrap;
        }}
        .graph-toolbar label {{ font-size: 0.85em; color: #495057; display: flex; align-items: center; gap: 8px; }}
        .graph-toolbar input[type=range] {{ width: 140px; margin: 0; }}
        .legend {{ display: flex; gap: 14px; flex-wrap: wrap; }}
        .legend-item {{ display: flex; align-items: center; gap: 6px; font-size: 0.8em; color: #495057; }}
        .legend-dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
        .project-filter {{ display: flex; gap: 12px; flex-wrap: wrap; margin-left: auto; }}
        .project-chip {{ display: flex; align-items: center; gap: 5px; font-size: 0.8em; color: #495057; cursor: pointer; user-select: none; }}
        .project-chip input[type=checkbox] {{ width: auto; margin: 0; }}
        .project-dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
        .cross-legend {{ display: flex; align-items: center; gap: 6px; font-size: 0.8em; color: #495057; }}
        .cross-line {{ width: 18px; height: 2px; background: #fd7e14; display: inline-block; }}
        .graph-body {{ display: grid; grid-template-columns: 1fr 300px; }}
        @media (max-width: 900px) {{ .graph-body {{ grid-template-columns: 1fr; }} }}
        #graph-canvas-wrap {{ position: relative; height: 560px; background: #fbfbfd; }}
        #graph-canvas {{ display: block; width: 100%; height: 100%; cursor: grab; }}
        .graph-side {{ padding: 16px; border-left: 1px solid #dee2e6; max-height: 560px; overflow-y: auto; }}
        .graph-empty-hint {{ position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%); color: #adb5bd; text-align: center; }}
        .btn-link {{ display: inline-block; margin-top: 10px; color: #667eea; font-size: 0.85em; text-decoration: none; font-weight: 500; }}
        .btn-link:hover {{ text-decoration: underline; }}

        /* Modellar ko'rinish */
        .view-models {{ display: none; padding: 24px 20px; max-height: 620px; overflow-y: auto; }}
        .models-wrap {{ max-width: 640px; margin: 0 auto; }}
        .models-wrap h2 {{ margin-bottom: 6px; color: #212529; }}
        .models-hint {{ color: #6c757d; font-size: 0.9em; margin-bottom: 18px; }}
        .warning-box {{ background: #fff3cd; border: 1px solid #ffe69c; color: #856404; border-radius: 6px; padding: 12px 14px; margin-bottom: 18px; font-size: 0.88em; line-height: 1.5; }}
        .success-box {{ background: #d4edda; border: 1px solid #c3e6cb; color: #155724; border-radius: 6px; padding: 10px 14px; margin-bottom: 18px; font-size: 0.88em; }}
        .model-option {{ display: block; border: 1px solid #dee2e6; border-radius: 6px; padding: 12px 14px; margin-bottom: 10px; cursor: pointer; }}
        .model-option-head {{ display: flex; align-items: center; gap: 8px; }}
        .model-option-head input[type=radio] {{ width: auto; margin: 0; }}
        .model-cost {{ margin-left: auto; font-size: 0.8em; color: #6c757d; }}
        .model-active-badge {{ background: #d4edda; color: #155724; font-size: 0.7em; padding: 2px 6px; border-radius: 10px; }}
        .model-desc {{ font-size: 0.82em; color: #6c757d; margin-top: 4px; }}
        .model-fields {{ margin-top: 10px; }}
        .model-fields input {{ margin-bottom: 8px; }}
        .models-note {{ font-size: 0.8em; color: #6c757d; margin-top: 10px; }}
    </style>
</head>
<body>
    <div class="app">
        <div class="topbar">
            <div class="brand">
                <h1>Graphify</h1>
                <span>Claude Code uchun doimiy xotira</span>
            </div>
            <div class="tabs">
                <button class="tab-btn active" data-view="list" type="button">Ro'yxat</button>
                <button class="tab-btn" data-view="graph" type="button">Graf</button>
                <button class="tab-btn" data-view="models" type="button">Modellar</button>
            </div>
            <div class="topbar-stats">{len(memories)} xotira • {len(projects)} loyiha • token tejaldi: {token_pct_label} (taxminiy)</div>
        </div>

        <div class="view-list" id="view-list">
            <div class="sidebar">
                <div class="stats">
                    <div class="stat"><div class="stat-num">{len(memories)}</div><div class="stat-label">Jami</div></div>
                    <div class="stat"><div class="stat-num">{len(filtered)}</div><div class="stat-label">Tanlangan</div></div>
                    <div class="stat" style="grid-column: 1 / -1;"><div class="stat-num">{token_pct_label}</div><div class="stat-label">Token tejaldi (taxminiy)</div></div>
                </div>

                <h2>Loyiha</h2>
                <select onchange="window.location='?project='+encodeURIComponent(this.value)">
                    <option value="">Hamma</option>
                    {project_opts}
                </select>

                <h2>Tur</h2>
                <select onchange="window.location='?kind='+encodeURIComponent(this.value)">
                    <option value="">Hamma</option>
                    {kind_opts}
                </select>

                <h2>Qidirish</h2>
                <form method="get">
                    <input type="text" name="q" placeholder="Qidiring..." value="{esc(search_query)}">
                    <button type="submit">OK</button>
                </form>

                <h2 style="margin-top:20px;">Xotiralari</h2>
                {memory_list}
            </div>
            <div class="main">{detail_html}</div>
        </div>

        <div class="view-graph" id="view-graph">
            <div class="graph-toolbar">
                <label>Sezuvchanlik chegarasi
                    <input type="range" id="sim-threshold" min="0.05" max="0.8" step="0.05" value="0.15">
                    <span id="sim-threshold-label">0.15</span>
                </label>
                <div class="project-filter">{project_filter_items}</div>
                <div class="legend">{legend_items}</div>
                <div class="cross-legend"><span class="cross-line"></span> boshqa loyihadan bog'lanish</div>
            </div>
            <div class="graph-body">
                <div id="graph-canvas-wrap">
                    <canvas id="graph-canvas"></canvas>
                    <div class="graph-empty-hint" id="graph-empty-hint" style="display:none;">Graf uchun kamida 2 ta xotira kerak</div>
                </div>
                <div class="graph-side">
                    <div id="graph-detail"><div class="empty">Tugunni bosing</div></div>
                </div>
            </div>
        </div>

        <div class="view-models" id="view-models">
            <div class="models-wrap">
                <h2>Embedding modeli</h2>
                <p class="models-hint">Qidiruv va saqlashda matnni vektorga aylantiruvchi model. Standart — kalit yoki internet talab qilmaydigan lokal model.</p>

                {'<div class="success-box">Saqlandi. O\'zgarish keyingi safar Claude Code / MCP server qayta ishga tushganda kuchga kiradi.</div>' if models_saved else ''}

                {f'<div class="warning-box"><strong>Ogohlantirish:</strong> bazada allaqachon {doc_count} ta yozuv bor. Embedderni almashtirish — buzuvchi (breaking) o\'zgarish: eski vektorlar yangi model fazosiga mos kelmaydi va qidiruv natijalari noto\'g\'ri bo\'lib qoladi. Chromadb mavjud, boshqa embedder bilan yozilgan bazani avtomatik qayta yozishga yo\'l qo\'ymaydi — server shunchaki eski (saqlangan) embedderga qaytadi. Haqiqiy almashtirish uchun: <code>kg_db/</code> papkasini bo\'shatib (yoki backup qilib), <code>indexer.py</code> bilan hujjatlarni qaytadan indekslang.</div>' if doc_count else ''}

                <form method="post" action="/models/save" onsubmit="return confirmModelSwitch()">
                    {model_options_html}
                    <button type="submit">Saqlash</button>
                </form>
                <div class="models-note">Konfiguratsiya <code>models_config.json</code> fayliga yoziladi; uni <code>server.py</code> ishga tushganda o'qiydi. API kalitlar shu faylda ochiq matn sifatida saqlanadi — uni git'ga qo'shmang.</div>
            </div>
        </div>
    </div>

    <script>
        // --- Tab almashtirish ---
        document.querySelectorAll('.tab-btn').forEach(btn => {{
            btn.addEventListener('click', () => {{
                document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                const view = btn.dataset.view;
                document.getElementById('view-list').style.display = view === 'list' ? 'grid' : 'none';
                document.getElementById('view-graph').style.display = view === 'graph' ? 'block' : 'none';
                document.getElementById('view-models').style.display = view === 'models' ? 'block' : 'none';
                if (view === 'graph') initGraphView();
            }});
        }});

        const initialTab = new URLSearchParams(location.search).get('tab');
        if (initialTab) {{
            const initialBtn = document.querySelector(`.tab-btn[data-view="${{initialTab}}"]`);
            if (initialBtn) initialBtn.click();
        }}

        // --- Modellar: forma ---
        function toggleModelFields() {{
            document.querySelectorAll('input[name="embedder_type"]').forEach(r => {{
                const box = document.querySelector(`.model-fields[data-key="${{r.value}}"]`);
                if (box) box.style.display = (r.checked && box.innerHTML.trim()) ? 'block' : 'none';
            }});
        }}
        toggleModelFields();

        function confirmModelSwitch() {{
            const selected = document.querySelector('input[name="embedder_type"]:checked');
            if (!selected) return true;
            const active = '{active_backend}';
            const docCount = {doc_count};
            if (selected.value !== active && docCount > 0) {{
                return confirm('Diqqat: bazada ' + docCount + " ta yozuv bor. Embedderni almashtirish qidiruvni buzishi mumkin (eski vektorlar yangisiga mos kelmaydi). Baribir davom etasizmi?");
            }}
            return true;
        }}

        // --- Graf: state ---
        const KIND_COLORS = {{'decision': '#28a745', 'summary': '#17a2b8', 'note': '#ffc107', 'doc': '#6c757d'}};
        const PROJECT_COLORS = {project_colors_json};
        let graphLoaded = false;
        let simNodes = [], simEdges = [];
        let canvas, ctx, W, H;
        let dragNode = null, hoverNode = null;
        let minSimThreshold = 0.15;
        let hiddenProjects = new Set();

        document.querySelectorAll('.project-toggle').forEach(cb => {{
            cb.addEventListener('change', () => {{
                if (cb.checked) hiddenProjects.delete(cb.value);
                else hiddenProjects.add(cb.value);
            }});
        }});

        function escHtml(s) {{
            const d = document.createElement('div');
            d.textContent = s;
            return d.innerHTML;
        }}

        function initGraphView() {{
            if (graphLoaded) return;
            graphLoaded = true;
            fetch('/api/graph-data').then(r => r.json()).then(setupGraph);
        }}

        function setupGraph(data) {{
            canvas = document.getElementById('graph-canvas');
            ctx = canvas.getContext('2d');
            resizeCanvas();
            window.addEventListener('resize', resizeCanvas);

            if (data.nodes.length < 2) {{
                document.getElementById('graph-empty-hint').style.display = 'block';
            }}

            const idToIndex = {{}};
            simNodes = data.nodes.map((n, i) => {{
                idToIndex[n.id] = i;
                const angle = (i / Math.max(1, data.nodes.length)) * Math.PI * 2;
                return Object.assign({{}}, n, {{
                    x: W / 2 + Math.cos(angle) * 120,
                    y: H / 2 + Math.sin(angle) * 120,
                    vx: 0, vy: 0, degree: 0,
                }});
            }});
            simEdges = data.edges.map(e => ({{ a: idToIndex[e.source], b: idToIndex[e.target], weight: e.weight }}));
            simEdges.forEach(e => {{ simNodes[e.a].degree++; simNodes[e.b].degree++; }});

            attachCanvasEvents();
            requestAnimationFrame(tick);

            document.getElementById('sim-threshold').addEventListener('input', e => {{
                minSimThreshold = parseFloat(e.target.value);
                document.getElementById('sim-threshold-label').textContent = minSimThreshold.toFixed(2);
            }});
        }}

        function resizeCanvas() {{
            const wrap = document.getElementById('graph-canvas-wrap');
            W = canvas.width = wrap.clientWidth;
            H = canvas.height = wrap.clientHeight;
        }}

        function tick() {{
            step();
            draw();
            requestAnimationFrame(tick);
        }}

        function step() {{
            const REPEL = 2400, SPRING = 0.02, CENTER = 0.002, DAMP = 0.85;
            for (let i = 0; i < simNodes.length; i++) {{
                const a = simNodes[i];
                if (a === dragNode) continue;
                let fx = 0, fy = 0;
                for (let j = 0; j < simNodes.length; j++) {{
                    if (i === j) continue;
                    const b = simNodes[j];
                    const dx = a.x - b.x, dy = a.y - b.y;
                    const distSq = dx * dx + dy * dy || 0.01;
                    const dist = Math.sqrt(distSq);
                    const force = REPEL / distSq;
                    fx += (dx / dist) * force;
                    fy += (dy / dist) * force;
                }}
                fx += (W / 2 - a.x) * CENTER;
                fy += (H / 2 - a.y) * CENTER;
                a.vx = (a.vx + fx) * DAMP;
                a.vy = (a.vy + fy) * DAMP;
            }}
            simEdges.forEach(e => {{
                if (e.weight < minSimThreshold) return;
                const a = simNodes[e.a], b = simNodes[e.b];
                const dx = b.x - a.x, dy = b.y - a.y;
                const dist = Math.sqrt(dx * dx + dy * dy) || 0.01;
                const targetDist = 150 - e.weight * 90;
                const force = (dist - targetDist) * SPRING * e.weight;
                const fx = (dx / dist) * force, fy = (dy / dist) * force;
                if (a !== dragNode) {{ a.vx += fx; a.vy += fy; }}
                if (b !== dragNode) {{ b.vx -= fx; b.vy -= fy; }}
            }});
            simNodes.forEach(n => {{
                if (n === dragNode) return;
                n.x += n.vx * 0.15;
                n.y += n.vy * 0.15;
                n.x = Math.max(20, Math.min(W - 20, n.x));
                n.y = Math.max(20, Math.min(H - 20, n.y));
            }});
        }}

        function draw() {{
            if (!ctx) return;
            ctx.clearRect(0, 0, W, H);
            simEdges.forEach(e => {{
                if (e.weight < minSimThreshold) return;
                const a = simNodes[e.a], b = simNodes[e.b];
                if (hiddenProjects.has(a.project) || hiddenProjects.has(b.project)) return;
                const crossProject = a.project !== b.project;
                ctx.strokeStyle = crossProject
                    ? `rgba(253,126,20,${{Math.min(0.75, e.weight + 0.25)}})`
                    : `rgba(102,126,234,${{Math.min(0.6, e.weight)}})`;
                ctx.lineWidth = crossProject ? 2 : 1;
                ctx.beginPath();
                ctx.moveTo(a.x, a.y);
                ctx.lineTo(b.x, b.y);
                ctx.stroke();
            }});
            simNodes.forEach(n => {{
                if (hiddenProjects.has(n.project)) return;
                const r = 6 + Math.min(n.degree, 8) * 1.2;
                ctx.beginPath();
                ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
                ctx.fillStyle = KIND_COLORS[n.kind] || '#667eea';
                ctx.fill();
                ctx.lineWidth = 2.5;
                ctx.strokeStyle = PROJECT_COLORS[n.project] || '#adb5bd';
                ctx.stroke();
                if (n === hoverNode || n === dragNode) {{
                    ctx.lineWidth = 2;
                    ctx.strokeStyle = '#212529';
                    ctx.stroke();
                }}
                if (r > 8 || n === hoverNode) {{
                    ctx.fillStyle = '#212529';
                    ctx.font = '11px sans-serif';
                    const label = n.title.length > 24 ? n.title.slice(0, 24) + '…' : n.title;
                    ctx.fillText(label, n.x + r + 4, n.y + 4);
                }}
            }});
        }}

        function nodeAt(x, y) {{
            for (let i = simNodes.length - 1; i >= 0; i--) {{
                const n = simNodes[i];
                if (hiddenProjects.has(n.project)) continue;
                const r = 6 + Math.min(n.degree, 8) * 1.2;
                const dx = x - n.x, dy = y - n.y;
                if (dx * dx + dy * dy <= (r + 4) * (r + 4)) return n;
            }}
            return null;
        }}

        function canvasPos(e) {{
            const rect = canvas.getBoundingClientRect();
            return {{ x: e.clientX - rect.left, y: e.clientY - rect.top }};
        }}

        function attachCanvasEvents() {{
            canvas.addEventListener('mousedown', e => {{
                const {{ x, y }} = canvasPos(e);
                dragNode = nodeAt(x, y);
                canvas.style.cursor = dragNode ? 'grabbing' : 'grab';
            }});
            window.addEventListener('mouseup', () => {{
                if (dragNode) showNodeDetail(dragNode);
                dragNode = null;
                canvas.style.cursor = 'grab';
            }});
            canvas.addEventListener('mousemove', e => {{
                const {{ x, y }} = canvasPos(e);
                if (dragNode) {{
                    dragNode.x = x; dragNode.y = y; dragNode.vx = 0; dragNode.vy = 0;
                }} else {{
                    hoverNode = nodeAt(x, y);
                    canvas.style.cursor = hoverNode ? 'pointer' : 'grab';
                }}
            }});
        }}

        function safeKind(k) {{
            return Object.prototype.hasOwnProperty.call(KIND_COLORS, k) ? k : 'note';
        }}

        function showNodeDetail(n) {{
            const panel = document.getElementById('graph-detail');
            const sk = safeKind(n.kind);
            panel.innerHTML = `
                <span class="badge badge-${{sk}}" style="background:${{KIND_COLORS[sk]}}22;color:${{KIND_COLORS[sk]}};">${{escHtml(n.kind)}}</span>
                <h3 style="margin:8px 0 4px;font-size:1em;">${{escHtml(n.title)}}</h3>
                <div style="font-size:0.8em;color:#6c757d;margin-bottom:10px;">${{escHtml(n.project)}} • ${{escHtml(n.date)}} • ${{n.degree}} ta bog'lanish</div>
                <div style="font-size:0.88em;line-height:1.5;white-space:pre-wrap;max-height:280px;overflow-y:auto;">${{escHtml(n.content)}}</div>
                <a class="btn-link" href="?id=${{encodeURIComponent(n.id)}}">To'liq ko'rish / o'chirish →</a>
            `;
        }}
    </script>
</body>
</html>"""
    return html


@app.route('/api/graph-data')
def graph_data():
    res = col.get(include=['metadatas', 'embeddings', 'documents'])
    ids = res['ids']
    metas = res['metadatas']
    docs = res['documents']
    embeddings = res['embeddings']

    def to_node(i):
        content = docs[i]
        return {
            'id': ids[i],
            'title': metas[i].get('title', '(untitled)'),
            'kind': metas[i].get('kind', 'note'),
            'project': metas[i].get('project', '?'),
            'date': metas[i].get('date', '?'),
            'content': (content[:400] + '…') if len(content) > 400 else content,
        }

    n = len(ids)
    nodes = [to_node(i) for i in range(n)]

    if n < 2 or embeddings is None:
        return jsonify({'nodes': nodes, 'edges': []})

    emb = np.array(embeddings)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    normed = emb / norms
    similarity = normed @ normed.T
    np.fill_diagonal(similarity, -1.0)

    edge_weight = {}
    for i in range(n):
        neighbors = np.argsort(-similarity[i])[:TOP_K_NEIGHBORS]
        for j in neighbors:
            j = int(j)
            score = float(similarity[i][j])
            if score < MIN_EDGE_SIMILARITY:
                continue
            key = (i, j) if i < j else (j, i)
            if key not in edge_weight or edge_weight[key] < score:
                edge_weight[key] = score

    edges = [
        {'source': ids[i], 'target': ids[j], 'weight': round(w, 3)}
        for (i, j), w in edge_weight.items()
    ]

    return jsonify({'nodes': nodes, 'edges': edges})


@app.route('/delete', methods=['POST'])
def delete_memory():
    mem_id = request.form.get('id')
    if mem_id:
        col.delete(ids=[mem_id])
    return redirect('/')


@app.route('/models/save', methods=['POST'])
def save_models():
    """Persist the selected embedder to models_config.json. This is a config/
    selection UI only — it does NOT re-embed existing data. server.py picks
    the new setting up the next time it starts."""
    cfg = load_models_config()
    embedder_type = request.form.get('embedder_type', 'local')
    cfg['embedder_type'] = embedder_type if embedder_type in BACKENDS else 'local'
    for backend_key in ('openai', 'ollama', 'custom'):
        sub = cfg['config'].setdefault(backend_key, {})
        for field in list(sub.keys()):
            val = request.form.get(f'{backend_key}_{field}')
            if val is not None:
                sub[field] = val
    save_models_config(cfg)
    return redirect('/?tab=models&saved=1')


if __name__ == '__main__':
    print("Dashboard: http://localhost:5000")
    app.run(debug=False, port=5000)
