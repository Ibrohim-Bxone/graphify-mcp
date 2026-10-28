import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import keyword_index
import search_core
import search_policy
import search_service


class Collection:
    metadata = {'hnsw:space':'cosine'}

    def __init__(self, records, vector_ids=()):
        self.records=records
        self.vector_ids=vector_ids

    def count(self): return len(self.records)

    def get(self, ids, include):
        keys=[key for key in ids if key in self.records]
        return {'ids':keys,'documents':[self.records[key][0] for key in keys],
                'metadatas':[self.records[key][1] for key in keys]}

    def query(self, **kwargs):
        keys=list(self.vector_ids)
        return {'ids':[keys],'documents':[[self.records[k][0] for k in keys]],
                'metadatas':[[self.records[k][1] for k in keys]],'distances':[[.05+i*.2 for i in range(len(keys))]]}


@pytest.fixture
def index(tmp_path,monkeypatch):
    monkeypatch.setattr(keyword_index,'DB_PATH',tmp_path/'kw.sqlite3')
    monkeypatch.setattr(search_policy,'ARCHIVE_DIR',tmp_path/'archive')
    search_policy.ARCHIVE_DIR.mkdir()
    value=keyword_index.KeywordIndex()
    yield value
    value.conn.close()


def search(index,records,query,where=None,vector_ids=()):
    index.upsert(list(records),[v[0] for v in records.values()],[v[1] for v in records.values()])
    return search_core.hybrid_search(query=query,top_k=5,where=where,collection=Collection(records,vector_ids),kw_index=index)


def test_legacy_and_typed_quarantine_excluded_from_results_and_candidates(index):
    (search_policy.ARCHIVE_DIR/'bad123.md').write_text('---\nusable: false\n---\narchived body','utf-8')
    records={
      'ig:bad123:item:0':('Quarantine content',{'id':'ig:bad123:item:0','title':'Quarantine','project':'shared','shortcode':'bad123'}),
      'file_old':('Quarantine content sibling without frontmatter',{'id':'file_old','title':'Quarantine','project':'shared','source':r'D:\x\Promtlarim\instagram\bad123.md'}),
    }
    result=search(index,records,'Quarantine',vector_ids=list(records))
    assert result['results']==[] and result['related_candidates']==[]


def test_archive_status_change_rechecked_without_index_mutation(index):
    p=search_policy.ARCHIVE_DIR/'change.md';p.write_text('---\nusable: true\n---','utf-8')
    meta={'id':'ig:change:item:0'}
    assert search_policy.usable(meta)
    p.write_text('---\nusable: false\n---','utf-8')
    assert not search_policy.usable(meta)


def test_canonical_group_uses_passing_sibling_and_collapses_legacy_alias(index):
    records={
      'ig:same:summary':('A generic description',{'id':'ig:same:summary','title':'Generic','shortcode':'same','parent_id':'ig:same:summary'}),
      'ig:same:item:0':('needle phrase is here',{'id':'ig:same:item:0','title':'Exact','shortcode':'same','parent_id':'ig:same:item:0'}),
      'file_legacy':('needle phrase legacy',{'id':'file_legacy','source':r'D:\x\Promtlarim\instagram\same.md'}),
    }
    result=search(index,records,'needle phrase',vector_ids=['ig:same:summary','ig:same:item:0'])
    assert len(result['results'])==1 and 'needle phrase' in result['results'][0]['text']
    assert search_policy.source_key(result['results'][0]['meta'])=='ig:same'
    assert len({search_policy.source_key(r['meta']) for r in result['results']+result['related_candidates']})==1


def test_stale_keyword_project_cannot_leak_authoritative_private_metadata(index):
    records={'private':('private needle',{'id':'private','project':'PRIVATE','title':'Needle'})}
    index.upsert(['private'],['private needle'],[{'id':'private','project':'shared','title':'Needle'}])
    result=search_core.hybrid_search(query='needle',top_k=5,where={'project':'shared'},collection=Collection(records,['private']),kw_index=index)
    assert result['results']==[] and result['related_candidates']==[]


def test_memory_rows_do_not_collapse_to_generic_source():
    assert search_policy.source_key({'id':'mem_1','parent_id':'save_memory'})=='mem_1'
    assert search_policy.source_key({'id':'mem_2','source':'save_memory'})=='mem_2'


def test_query_rewrites_cannot_double_vote_unrelated_source(monkeypatch):
    monkeypatch.setattr(search_policy, 'query_variants', lambda q: ['original','translated'])
    calls=[]
    def core(**kw):
        calls.append(kw)
        keys=['a','b','c','d'] if kw['query']=='original' else ['x','y','d','a']
        return {'results':[{'meta':{'id':key},'text':kw['query']+key} for key in keys]}
    monkeypatch.setattr(search_core,'hybrid_search',core)
    result=search_service.retrieve('question',top_k=4,projects=['shared'],collection=None,kw_index=None)
    assert [item['source_key'] for item in result['results']]==['a','x','b','y']
    assert all(call['where']=={'project':'shared'} for call in calls)
    assert result['results'][0]['matched_query']=='original'
    candidate=next(item for item in result['related_candidates'] if item['source_key']=='d')
    assert candidate['matched_query']=='translated' and candidate['text']=='translatedd'


def test_query_variants_preserve_money_versions_and_quoted_constraints():
    variants=search_policy.query_variants('Salom, menga shaxsiy yordamchi $10k+ sof foyda gpt-6 "exact model" haqida ma’lumot ber')
    assert 1<=len(variants)<=2
    for value in variants:
        assert '$10k+' in value and 'gpt-6' in value and '"exact model"' in value
    assert any('net profit' in value for value in variants)


@pytest.mark.parametrize('phrase', ['bazadagi manbaga tayanib','bazadagi manbalarga asoslanib',
    'bilimlar bazasidagi manbaga tayanib','bilimlar bazasidagi manbalarga asoslanib'])
@pytest.mark.parametrize('uppercase', [False,True])
def test_source_request_framing_does_not_hide_named_tool(phrase,uppercase):
    text=phrase.upper() if uppercase else phrase
    assert search_policy.query_variants('Codebase Memory MCP haqida '+text+' tushuntir')==['Codebase Memory MCP']


def test_source_request_framing_preserves_quotes_and_numeric_constraints():
    query='Tool 2026 $10k+ sof foyda gpt-6 "bazadagi manbaga tayanib" bazadagi manbalarga asoslanib'
    variants=search_policy.query_variants(query)
    for value in variants:
        assert all(token in value for token in ('2026','$10k+','gpt-6','"bazadagi manbaga tayanib"'))
        assert 'bazadagi manbalarga asoslanib' not in value


def test_source_request_framing_does_not_remove_substantive_source_terms():
    assert search_policy.query_variants('manbaga hujum xavfsizlik tekshiruvi')[0]=='manbaga hujum xavfsizlik tekshiruvi'


def test_source_request_without_topic_does_not_create_search_terms():
    assert search_policy.query_variants('bazadagi manbaga tayanib tushuntir')==[]


@pytest.mark.parametrize('quoted', ['"bazadagi manbaga\ntayanib"','"bilimlar bazasidagi\nmanbalarga asoslanib"'])
def test_source_request_framing_preserves_multiline_quoted_constraints(quoted):
    variants=search_policy.query_variants('Hindsight 2026 '+quoted+' bazadagi manbaga tayanib')
    assert variants
    assert all(quoted in value for value in variants)


@pytest.mark.parametrize('quoted', ['"kod baza"','"kod\nbaza"'])
def test_codebase_translation_never_rewrites_exact_quoted_constraint(quoted):
    variants=search_policy.query_variants('kod bazasini 2026 '+quoted)
    assert len(variants)==2
    assert all(quoted in value for value in variants)
    assert variants[1].startswith('codebase 2026')


def test_year_metadata_is_searchable_but_wrong_year_stays_rejected(index):
    records={'date':('harness architecture',{'id':'date','title':'Harness','date':'2026-10-25'})}
    assert search(index,records,'2026 harness architecture')['results']
    assert search(index,records,'2099 harness architecture')['results']==[]


def test_factory_context_scope_and_complete_chunk_budget(index):
    text='needle full evidence '+('content '*100)
    records={
      'shared_doc':(text,{'id':'shared_doc','title':'Needle','project':'Promtlarim','source':'book'}),
      'private_doc':('needle private',{'id':'private_doc','title':'Needle','project':'PRIVATE','source':'private'}),
    }
    index.upsert(list(records),[v[0] for v in records.values()],[v[1] for v in records.values()])
    result=search_service.factory_context('needle','test-project',collection=Collection(records,list(records)),kw_index=index)
    assert [s['project'] for s in result['sources']]==['test-project','shared','Promtlarim']
    assert result['sources'][2]['hits'][0]['text']==text
    assert 'private_doc' not in json.dumps(result)
    assert len(json.dumps(result,ensure_ascii=False))<=10000
    tiny=search_service.factory_context('needle','test-project',max_chars=600,collection=Collection(records,list(records)),kw_index=index)
    assert tiny['truncated'] and tiny['status']=='insufficient_evidence'
    assert all(not s['hits'] for s in tiny['sources'])
    with pytest.raises(ValueError):
        search_service.factory_context('needle','all',collection=Collection(records),kw_index=index)


def test_server_exposes_labeled_candidates_when_no_accepted_evidence(monkeypatch):
    source=Path(search_core.__file__).with_name('server.py')
    tree=ast.parse(source.read_text('utf-8'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='search_knowledge')
    fn.decorator_list=[]
    res=dict(status='insufficient_evidence',results=[],related_candidates=[{'meta':{'id':'candidate','title':'Related title'},'similarity':.4}])
    monkeypatch.setattr(search_service,'retrieve',lambda **kw:res)
    monkeypatch.setattr(keyword_index,'get_index',lambda:None)
    namespace={'_col':SimpleNamespace(count=lambda:1),'PROJECT':'test','ALWAYS_OPEN':('shared','Promtlarim'),
               'MIN_SIMILARITY':.1,'_record_search_savings':lambda *a:None,'_est_tokens':lambda t:1}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),namespace)
    result=namespace['search_knowledge']('needle')
    assert 'yetarli mos dalil topilmadi' in result and 'faqat nomzod' in result
    assert 'Related title' in result and 'candidate' in result
