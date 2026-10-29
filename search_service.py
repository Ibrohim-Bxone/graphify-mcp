"""Shared retrieval for MCP and the Telegram CLI; expansion never changes the access scope."""
import json
from collections import defaultdict
import search_core
import search_policy


def retrieve(query, *, top_k=5, projects=None, where=None, collection, kw_index, min_similarity=.1):
    if projects is not None:
        if not projects or any(p in ('all', '*', 'current', '') for p in projects):
            raise ValueError('Explicit, non-reserved project namespaces required')
        if where is not None:
            raise ValueError('Use projects or where, not both')
        where = {'project': projects[0]} if len(projects) == 1 else {'project': {'$in': list(dict.fromkeys(projects))}}
    top_k = max(1, min(int(top_k), 30))
    variants = search_policy.query_variants(query)
    scores = defaultdict(float); hits = {}; related = {}
    primary_res = None
    all_results = []
    for variant in variants:
        res = search_core.hybrid_search(query=variant, top_k=top_k, where=where,
                                       collection=collection, kw_index=kw_index, min_similarity=min_similarity)
        if primary_res is None:
            primary_res = res
        all_results.append(res)
        for rank, item in enumerate(res['results']):
            key = search_policy.source_key(item.get('meta') or {})
            # Rewrites of one question are alternatives, not independent votes.
            score = 1 / (60 + rank + 1)
            # Preserve one complete evidence-bearing chunk; don't synthesize text across sources.
            if key not in hits or score > scores[key]:
                hits[key] = dict(item, matched_query=variant, source_key=key)
            scores[key] = max(scores[key], score)
        for item in res.get('related_candidates', []):
            key = search_policy.source_key(item.get('meta') or {})
            related.setdefault(key, dict(item, source_key=key))
    ranked = sorted(hits, key=lambda key: scores[key], reverse=True)
    chosen = ranked[:top_k]
    candidates = [hits[key] for key in ranked[top_k:]] + [item for key, item in related.items() if key not in hits]
    rerank_statuses = [r.get('rerank', 'off') for r in all_results]
    if 'unavailable' in rerank_statuses:
        final_rerank = 'unavailable'
        reasons = [r.get('rerank_reason') for r in all_results if r.get('rerank') == 'unavailable' and r.get('rerank_reason')]
        if not reasons:
            reasons = [r.get('rerank_reason') for r in all_results if r.get('rerank_reason')]
        final_reason = '; '.join(dict.fromkeys(reasons)) if reasons else 'rerank unavailable'
    elif 'on' in rerank_statuses:
        final_rerank = 'on'
        final_reason = None
    elif 'off' in rerank_statuses:
        final_rerank = 'off'
        reasons = [r.get('rerank_reason') for r in all_results if r.get('rerank_reason')]
        final_reason = reasons[0] if reasons else None
    else:
        final_rerank = 'off'
        final_reason = None
    return dict(status='ok' if chosen else 'insufficient_evidence', queries=variants,
                results=[hits[key] for key in chosen], related_candidates=candidates,
                rerank=final_rerank, rerank_reason=final_reason)


def factory_context(query, project, *, collection, kw_index, top_k=5, max_chars=10000):
    if project in ('all', '*', 'current', ''):
        raise ValueError('Factory project namespace reserved')
    projects = list(dict.fromkeys((project, 'shared', 'Promtlarim')))
    result = retrieve(query, top_k=top_k, projects=projects, collection=collection, kw_index=kw_index)
    sources = {name: {'project': name, 'status': 'searched', 'hits': [], 'candidates': []} for name in projects}
    payload = dict(status=result['status'], query=query[:1600], queries=result['queries'], sources=list(sources.values()),
                   coverage='Joriy loyiha + shared + Promtlarim; boshqa private loyihalar qidirilmadi.', truncated=False)

    def record(item, evidence=True):
        meta = item.get('meta') or {}
        value = {key: meta.get(key, '') for key in ('id', 'title', 'project', 'source', 'date')}
        value['id'] = item.get('id') or value['id']
        value['source_key'] = item.get('source_key') or search_policy.source_key(meta)
        if evidence:
            value.update(text=item['text'], matched_query=item.get('matched_query', query))
        return value

    for item in result['results']:
        value = record(item)
        destination = sources.get(value['project'])
        if destination is None:
            raise ValueError('Search result escaped factory namespace')
        destination['hits'].append(value)
        if len(json.dumps(payload, ensure_ascii=False)) > max_chars:
            destination['hits'].pop()
            payload['truncated'] = True
    for item in result.get('related_candidates', [])[:3]:
        value = record(item, evidence=False)
        destination = sources.get(value['project'])
        if destination is not None:
            destination['candidates'].append(value)
            if len(json.dumps(payload, ensure_ascii=False)) > max_chars:
                destination['candidates'].pop()
                payload['truncated'] = True
    if not any(s['hits'] for s in sources.values()):
        payload['status'] = 'insufficient_evidence'
    return payload
