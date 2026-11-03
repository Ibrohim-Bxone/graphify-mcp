"""Local retrieval policy: query normalization, canonical sources, live archive quarantine."""
import functools
import re
from pathlib import Path

ARCHIVE_DIR = Path(__file__).resolve().parent.parent / "Promtlarim" / "instagram"
_SC = re.compile(r"[A-Za-z0-9_-]+\Z")
_GENERIC = {"save_memory", "upload", "web_search", "web_page"}
_FILLER = set("salom hello hi iltimos please menga manga man bizga ber bering ayt aytib haqida batafsil ma'lumot ma’lumot tushuntir tushuntirib kerak qanday nima qaysi va bilan uchun bu shu shuni yordamida yordam menga manashu what how why which where when who does do is are used use using about tell me explain show the a an and for of to in it can you i".split())
_TRANSLATE = {
    'shaxsiy':'personal', 'yordamchi':'assistant', 'bosqich':'step', "to'qqiz":'9',
    "yig'ish":'build', 'yaratish':'build', 'xotira':'memory', 'doimiy':'persistent',
    'foydalanuvchi':'user', 'xohish':'preferences', 'bilim':'knowledge', 'grafig':'graph',
    'graf':'graph', 'kod':'code', 'hujjat':'document', 'qidirish':'search', 'qidiruv':'search',
    'vektorlarsiz':'vectorless', 'vektorsiz':'vectorless', 'mantiqiy':'reasoning',
    'katta':'large', 'spetsifikatsiya':'specification', 'reja':'plan', 'vazifa':'task',
    'ajratish':'breakdown', 'mustaqil':'independent', 'tekshiruvchi':'audits',
    'yozuvchi':'writes', 'ikki':'dual', 'muvaffaqiyatli':'successful', 'yechim':'solution',
    'saqlash':'store', "ko'nikma":'skill', 'yangi':'new', 'guruh':'group', 'bosh':'lead',
    'boshqarish':'orchestration', 'eslatma':'notes', 'kontekst':'context', 'uzoq':'long',
    'muddatli':'term', 'xavfsizlik':'security', 'fayl':'file', 'oy':'month', 'hafta':'week',
    'sof':'net', 'foyda':'profit', 'tushum':'revenue', 'daromad':'income',
}


def query_variants(query):
    """At most two loss-limited variants. Never drop quoted constraints or numeric anchors."""
    text = re.sub(r"['`’ʻʼ]", "'", str(query))
    _phrases = (
        r"bilimlar\s+bazasidagi\s+manbalarga\s+asoslanib|"
        r"bilimlar\s+bazasidagi\s+manbaga\s+tayanib|"
        r"bazadagi\s+manbalarga\s+asoslanib|"
        r"bazadagi\s+manbaga\s+tayanib"
    )
    text = re.sub(r'("[^"]+")|\b(?:' + _phrases + r')\b', lambda m: m.group(1) if m.group(1) is not None else '', text, flags=re.I)
    tokens = re.findall(r'"[^"]+"|\$?\d+(?:[.,]\d+)*(?:[kKmM])?\+?|[\w]+(?:[\x27-][\w]+)*', text)
    words = [w for w in tokens if w.casefold() not in _FILLER]
    if len(words) > 40:
        # Bound context while keeping dates/money/model versions, even in the middle.
        words = words[:20] + [w for w in words[20:-20] if any(c.isdigit() for c in w) or w.startswith('"')] + words[-20:]
    cleaned = ' '.join(words)
    if not cleaned or len(cleaned) > 1400:
        return []
    import keyword_index
    translated = re.sub(r'("[^"]+")|\bkod\s+baza\w*',
                        lambda m: m.group(1) if m.group(1) is not None else 'codebase', cleaned, flags=re.I)
    result = []
    for token in re.findall(r'"[^"]+"|\S+', translated):
        if token.startswith('"') or any(c.isdigit() for c in token):
            result.append(token)
            continue
        normalized = keyword_index.tokenize(token)
        word = normalized[0] if len(normalized) == 1 else token.casefold()
        replacement = _TRANSLATE.get(word)
        if replacement is None:
            replacement = next((value for root, value in _TRANSLATE.items()
                                if len(root) >= 6 and word.startswith(root)), token)
        result.append(replacement)
    expanded = ' '.join(result)
    return list(dict.fromkeys((cleaned, expanded)))


def shortcode(meta):
    for value in (meta.get('shortcode'),):
        if isinstance(value, str) and _SC.fullmatch(value):
            return value
    for value in (meta.get('id'), meta.get('parent_id'), meta.get('canonical_id')):
        if isinstance(value, str) and value.startswith('ig:'):
            candidate = value.split(':', 2)[1]
            if _SC.fullmatch(candidate):
                return candidate
    source = str(meta.get('source') or '').replace('\\', '/')
    match = re.search(r'(?:instagram\.com/(?:reel|p|tv)/|instagram-file:)([A-Za-z0-9_-]+)', source, re.I)
    if match:
        return match.group(1)
    # Legacy MD imports have no shortcode metadata. Resolve only the known archive layout.
    for value in (source, str(meta.get('parent_id') or '').replace('\\', '/')):
        match = re.search(r'/Promtlarim/instagram/([A-Za-z0-9_-]+)\.md$', value, re.I)
        if match:
            return match.group(1)
    return None


def source_key(meta):
    sc = shortcode(meta)
    if sc:
        return 'ig:' + sc
    pid = meta.get('parent_id') or meta.get('canonical_id') or meta.get('source')
    return meta.get('id') if not pid or pid in _GENERIC else pid


@functools.lru_cache(maxsize=4096)
def _archive_status(path, mtime_ns, size):
    try:
        with Path(path).open(encoding='utf-8-sig', errors='replace') as stream:
            header = stream.read(8192)
    except OSError:
        return False
    if not header.startswith('---'):
        return None
    header = header.split('\n---', 1)[0]
    found = re.search(r'(?mi)^usable:\s*["\x27]?(false|true|0|1|no|yes)\b', header)
    return found.group(1).lower() not in ('false', '0', 'no') if found else None


def usable(meta, text=''):
    if str(meta.get('usable', '')).strip().lower() in ('false', '0', 'no'):
        return False
    sc = shortcode(meta)
    if sc:
        path = ARCHIVE_DIR / (sc + '.md')
        try:
            stat = path.stat()
            return _archive_status(str(path), stat.st_mtime_ns, stat.st_size) is not False
        except FileNotFoundError:
            return True  # New in-memory/test imports may have no archive yet.
        except OSError:
            return False  # An existing archive whose state cannot be read is not evidence.
    return True


def matches_where(meta, where):
    """Re-check authoritative Chroma metadata; stale FTS metadata must not widen access."""
    if not where:
        return True
    for key, expected in where.items():
        if key in ('$and', '$or'):
            checks = [matches_where(meta, condition) for condition in expected]
            if not (all(checks) if key == '$and' else any(checks)):
                return False
        elif isinstance(expected, dict):
            for op, value in expected.items():
                actual = meta.get(key)
                if op == '$in' and actual in value:
                    continue
                if op == '$eq' and actual == value:
                    continue
                if op == '$ne' and actual != value:
                    continue
                if op == '$nin' and actual not in value:
                    continue
                return False
        elif meta.get(key) != expected:
            return False
    return True
