from context_management.batcher import Batcher
import pytest

def test_batches_exact_division():
    data = list(range(8))
    out = [b for b in Batcher(data, 2)]
    assert len(out) == 4
    assert all(len(b) == 2 for b in out)
    assert [x for batch in out for x in batch] == data

def test_remainder_batch():
    data = list(range(10))
    sizes = [len(b) for b in Batcher(data, 4)]
    assert sizes == [4, 4, 2]

def test_empty_iterable():
    assert list(Batcher([], 3)) == []

def test_batch_size_ge_len():
    data = list(range(5))
    out = list(Batcher(data, 10))
    assert len(out) == 1 and len(out[0]) == 5


def test_bad_batch_size_raises():
    with pytest.raises(ValueError):
        Batcher([1,2,3], 0)
