import pytest
import numpy as np
from similarity import distance_to_similarity, make_scorer

class MockCollection:
    def __init__(self, space="l2", embeddings=None):
        self.metadata = {"hnsw:space": space}
        self.embeddings = embeddings

    def get(self, limit=5, include=None):
        return {"embeddings": self.embeddings}

def test_l2_unit_vector():
    # Masofa = 1.0 -> sim = 1 - 1/2 = 0.5
    sim, raw_sim = distance_to_similarity(1.0, "l2")
    assert np.isclose(raw_sim, 0.5)
    assert np.isclose(sim, 0.5)
    
def test_l2_non_unit_vector():
    # Katta vektorlar: sim = dot(u,v) / (norm(u)*norm(v))
    # query_vec = [2, 0], doc_vec = [0, 2] -> ortogonal -> sim = 0.0
    sim, raw_sim = distance_to_similarity(4.0, "l2", query_vec=np.array([2, 0]), doc_vec=np.array([0, 2]))
    assert np.isclose(raw_sim, 0.0)
    assert np.isclose(sim, 0.0)
    
    # query_vec = [2, 0], doc_vec = [2, 0] -> d = 0 -> sim = 1.0
    sim, raw_sim = distance_to_similarity(0.0, "l2", query_vec=np.array([2, 0]), doc_vec=np.array([2, 0]))
    assert np.isclose(raw_sim, 1.0)
    
def test_cosine_space():
    # Cosine d = 0.3 -> sim = 0.7
    sim, raw_sim = distance_to_similarity(0.3, "cosine")
    assert np.isclose(raw_sim, 0.7)
    assert np.isclose(sim, 0.7)

def test_unknown_space():
    with pytest.raises(ValueError):
        distance_to_similarity(0.5, "noma'lum")
        
def test_clamping():
    # d = 4.0 -> sim = 1 - 2.0 = -1.0
    sim, raw_sim = distance_to_similarity(4.0, "l2")
    assert np.isclose(raw_sim, -1.0)
    assert np.isclose(sim, 0.0)
    
    # d = -1.0 -> sim = 1 - (-0.5) = 1.5 -> clamped 1.0
    sim, raw_sim = distance_to_similarity(-1.0, "l2")
    assert np.isclose(raw_sim, 1.5)
    assert np.isclose(sim, 1.0)
    
def test_make_scorer_unit():
    col = MockCollection("l2", embeddings=[np.array([1, 0]), np.array([0, 1])])
    scorer = make_scorer(col)
    sim, raw_sim = scorer(1.0)
    assert np.isclose(raw_sim, 0.5)

@pytest.mark.parametrize("embeddings", [[[2, 0], [0, 2]], np.array([[2, 0], [0, 2]])])
def test_make_scorer_non_unit(embeddings):
    col = MockCollection("l2", embeddings=embeddings)
    scorer = make_scorer(col)
    
    # Query / doc kiritmasak ValueError otishi kerak
    with pytest.raises(ValueError, match="Birlik emas L2 uchun query_vec va doc_vec kerak"):
        scorer(4.0)
        
    sim, raw_sim = scorer(4.0, query_vec=np.array([2, 0]), doc_vec=np.array([0, 2]))
    assert np.isclose(raw_sim, 0.0)

def test_make_scorer_missing_space():
    class NoSpaceCollection:
        metadata = {}
    with pytest.raises(ValueError, match="hnsw:space topilmadi"):
        make_scorer(NoSpaceCollection())
        
def test_make_scorer_nested_space():
    class NestedSpaceCollection:
        metadata = {"hnsw": {"space": "cosine"}}
    
    scorer = make_scorer(NestedSpaceCollection())
    sim, raw = scorer(0.3)
    assert np.isclose(raw, 0.7)

def test_unit_length_cache():
    import similarity
    similarity._unit_length_cache.clear()
    
    class CounterCollection(MockCollection):
        def __init__(self, name):
            super().__init__(space="l2", embeddings=[np.array([1, 0])])
            self.name = name
            self.get_calls = 0
            
        def get(self, limit=5, include=None):
            self.get_calls += 1
            return super().get(limit=limit, include=include)
            
    # Ishonchli o'zgarish signali yo'q: kesh ishlatilmaydi, har safar qayta tekshiriladi
    col = CounterCollection("test_col_1")
    make_scorer(col)
    make_scorer(col)
    assert col.get_calls == 2

    # Aniq write_generation signali bor: bir avlod ichida kesh, avlod o'zgarsa qayta tekshiriladi
    gen_col = CounterCollection("test_col_2")
    gen_col.write_generation = 1
    make_scorer(gen_col)
    make_scorer(gen_col)
    assert gen_col.get_calls == 1
    gen_col.write_generation = 2
    make_scorer(gen_col)
    assert gen_col.get_calls == 2

def test_make_scorer_space_from_configuration_json():
    # Chroma modify() metadata'ni to'liq almashtiradi va hnsw:space yo'qoladi; space configuration_json da qoladi
    class ConfiguredCollection:
        metadata = {"model": "bge-m3", "distance": "l2"}
        configuration_json = {"hnsw": {"space": "cosine"}}

    scorer = make_scorer(ConfiguredCollection())
    sim, raw = scorer(0.3)
    assert np.isclose(raw, 0.7)


def test_check_unit_length_exception_fails_closed_and_does_not_cache(capsys):
    """T5 regression: _check_unit_length returns False on exception, logs to stderr, and does not cache."""
    import similarity
    from similarity import _check_unit_length
    similarity._unit_length_cache.clear()

    class FailingCollection:
        id = "failing_col_test"

        def count(self):
            return 10

        def get(self, limit=5, include=None):
            raise RuntimeError("Database read failed deliberately")

    col = FailingCollection()
    # 1. On exception, must return False (fail-closed)
    res1 = _check_unit_length(col)
    assert res1 is False

    # 2. Cache must NOT store the result (cache remains empty)
    assert len(similarity._unit_length_cache) == 0

    # 3. Error must be logged to sys.stderr
    captured = capsys.readouterr()
    assert "Database read failed deliberately" in captured.err

    # 4. Next call must re-attempt and not return a stale cached value
    res2 = _check_unit_length(col)
    assert res2 is False
    assert len(similarity._unit_length_cache) == 0
