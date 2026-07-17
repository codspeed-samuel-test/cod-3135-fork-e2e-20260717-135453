from cosine import cosine_similarity


def test_cosine(benchmark):
    a = list(range(1000))
    b = list(range(1000, 2000))
    result = benchmark(lambda: cosine_similarity(a, b))
    assert result > 0
