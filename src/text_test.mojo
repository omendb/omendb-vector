from std.testing import assert_equal, assert_true
from text import tokenize, BM25Index


def test_tokenizer() raises:
    var words = tokenize("Hello, World! 123")
    assert_equal(len(words), 3)
    assert_equal(words[0], "hello")
    assert_equal(words[1], "world")
    assert_equal(words[2], "123")

    var unicode_words = tokenize("PageIndex-style source spans — BM25 ✅")
    assert_equal(len(unicode_words), 5)
    assert_equal(unicode_words[0], "pageindex")
    assert_equal(unicode_words[1], "style")
    assert_equal(unicode_words[2], "source")
    assert_equal(unicode_words[3], "spans")
    assert_equal(unicode_words[4], "bm25")


def test_bm25_basic() raises:
    var idx = BM25Index()
    _ = idx.add_document("The quick brown fox")
    _ = idx.add_document("The lazy dog")
    _ = idx.add_document("Apples and oranges")

    var results = idx.search("fox", k=1)
    assert_equal(len(results), 1)
    assert_equal(results[0][0], 0)  # fox is in doc 0

    results = idx.search("dog", k=1)
    assert_equal(results[0][0], 1)  # dog is in doc 1

    results = idx.search("the", k=2)
    assert_equal(len(results), 2)


def test_bm25_edge_cases() raises:
    var empty = BM25Index()
    var empty_results = empty.search("anything", k=10)
    assert_equal(len(empty_results), 0)

    var idx = BM25Index()
    _ = idx.add_document("Needle")
    _ = idx.add_document("needle filler filler filler filler filler filler")
    _ = idx.add_document("unrelated text only")

    var blank_results = idx.search("", k=10)
    assert_equal(len(blank_results), 0)

    var missing_results = idx.search("missing", k=10)
    assert_equal(len(missing_results), 0)

    var oversized_k = idx.search("needle", k=10)
    assert_equal(len(oversized_k), 2)
    assert_equal(oversized_k[0][0], 0)

    var repeated_query = idx.search("needle needle", k=2)
    assert_equal(len(repeated_query), 2)
    assert_equal(repeated_query[0][0], 0)

    var punct_case = idx.search("NEEDLE!!!", k=1)
    assert_equal(len(punct_case), 1)
    assert_equal(punct_case[0][0], 0)

    var duplicate = BM25Index()
    _ = duplicate.add_document("same phrase")
    _ = duplicate.add_document("same phrase")
    var duplicate_results = duplicate.search("same", k=10)
    assert_equal(len(duplicate_results), 2)
    assert_equal(duplicate_results[0][0] + duplicate_results[1][0], UInt32(1))


def main() raises:
    test_tokenizer()
    print("PASS test_tokenizer")
    test_bm25_basic()
    print("PASS test_bm25_basic")
    test_bm25_edge_cases()
    print("PASS test_bm25_edge_cases")
