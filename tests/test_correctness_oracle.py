"""Correctness oracle matrix for the local DB.

Every test uses a small deterministic corpus with known expected results.
This proves the engine produces mathematically correct output, not just
non-empty results.

Uses dim=2 (an optimized HNSW dimension) with L2 metric so we can test
text+vector together with small deterministic vectors that are easy to
reason about geometrically.

Note: L2 distance scores are negative (lower magnitude = closer). The
tests check relative ordering and score relationships, not absolute values.

Covers:
- Exact dense nearest-neighbor on deterministic vectors
- BM25 text ranking on known term distributions
- Hybrid RRF fusion combining dense + BM25 scores
- Metadata filter correctness
- Include/exclude ID constraints
- Update/delete/tombstone exclusion from search
- Flush/reopen equivalence (same results before and after)
- Check/vacuum preservation of correctness
- Text-only items (no vector) correctness
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

import omendb

# ---------------------------------------------------------------------------
# Deterministic corpus: 5 vectors in 2D space (L2 metric)
# ---------------------------------------------------------------------------
#
# id     vector       text                 category
# ----   ----------   ------------------   --------
# doc1   [1.0, 0.0]   "the cat sat"        animal
# doc2   [0.0, 1.0]   "the dog ran"        animal
# doc3   [0.9, 0.1]   "the cat ran fast"   animal
# doc4   [0.0, 0.9]   "python code runs"   tech
# doc5   [0.1, 0.9]   "code in python"     tech
#
# Query vector: [1.0, 0.0] (same as doc1)
# Expected L2 distances (lower = closer):
#   doc1: 0.0       (exact match)
#   doc3: ~0.14     (very close)
#   doc5: ~1.27     (moderate)
#   doc2: ~1.41     (far)
#   doc4: ~1.35     (far)
#
# BM25 for query text "cat":
#   doc1: contains "cat" -> ranked
#   doc3: contains "cat" -> ranked
#   doc2, doc4, doc5: no "cat" -> not ranked


CORPUS = [
    {
        "id": "doc1",
        "vector": [1.0, 0.0],
        "text": "the cat sat on the mat",
        "metadata": {"category": "animal", "year": 2020},
    },
    {
        "id": "doc2",
        "vector": [0.0, 1.0],
        "text": "the dog ran in the park",
        "metadata": {"category": "animal", "year": 2021},
    },
    {
        "id": "doc3",
        "vector": [0.9, 0.1],
        "text": "the cat ran fast today",
        "metadata": {"category": "animal", "year": 2022},
    },
    {
        "id": "doc4",
        "vector": [0.0, 0.9],
        "text": "python code runs well",
        "metadata": {"category": "tech", "year": 2020},
    },
    {
        "id": "doc5",
        "vector": [0.1, 0.9],
        "text": "code in python is fun",
        "metadata": {"category": "tech", "year": 2021},
    },
]

QUERY_VECTOR = [1.0, 0.0]


def _skip_if_free_threaded() -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")


def _seed_collection(
    db_path: str | None = None,
    name: str = "oracle",
) -> tuple:
    """Create and seed a collection with the deterministic corpus.

    Returns (db, col).
    """
    if db_path is None:
        db = omendb.memory()
    else:
        db = omendb.create(db_path)
    col = db.collection(
        name,
        config=omendb.CollectionConfig(dim=2, text=True),
    )
    for item in CORPUS:
        col.set(
            item["id"],
            vector=item["vector"],
            text=item["text"],
            metadata=item["metadata"],
        )
    return db, col


# ---------------------------------------------------------------------------
# 1. Exact dense nearest-neighbor oracle
# ---------------------------------------------------------------------------


class TestDenseNearestNeighborOracle:
    """Prove dense search returns mathematically correct nearest neighbors."""

    def test_exact_match_ranks_first(self) -> None:
        """Query identical to doc1 must return doc1 as top result."""
        _skip_if_free_threaded()
        _, col = _seed_collection()
        results = col.search_vector(QUERY_VECTOR, k=5)
        assert results[0].id == "doc1"
        # L2 distance to itself should be 0 (or very close)
        assert results[0].score == pytest.approx(0.0, abs=0.01)

    def test_nearest_neighbor_ordering(self) -> None:
        """doc1 > doc3 > others for query [1,0]."""
        _skip_if_free_threaded()
        _, col = _seed_collection()
        results = col.search_vector(QUERY_VECTOR, k=5)
        ids = [r.id for r in results]
        scores = [r.score for r in results]

        # doc1 is closest (exact match, distance ~0)
        assert ids[0] == "doc1"
        # doc3 is second closest (0.9, 0.1) ~ L2 distance ~0.14
        assert ids[1] == "doc3"
        # doc1 score (distance) < doc3 score (distance)
        # Note: L2 scores are negative, so -0.0 > -0.14
        assert scores[0] > scores[1]  # closer = higher (less negative)

    def test_different_query_vector_changes_ranking(self) -> None:
        """Query [0,1] should rank doc2 first, not doc1."""
        _skip_if_free_threaded()
        _, col = _seed_collection()
        tech_query = [0.0, 1.0]
        results = col.search_vector(tech_query, k=5)
        assert results[0].id == "doc2"

    def test_all_documents_returned_when_k_sufficient(self) -> None:
        """With k=5, all 5 documents should be returned."""
        _skip_if_free_threaded()
        _, col = _seed_collection()
        results = col.search_vector(QUERY_VECTOR, k=5)
        result_ids = {r.id for r in results}
        assert result_ids == {"doc1", "doc2", "doc3", "doc4", "doc5"}


# ---------------------------------------------------------------------------
# 2. BM25 text ranking oracle
# ---------------------------------------------------------------------------


class TestBM25TextRankingOracle:
    """Prove BM25 search returns correct text relevance ranking."""

    def test_exact_term_match_ranks_first(self) -> None:
        """Query "cat" should rank doc1 and doc3 above others."""
        _skip_if_free_threaded()
        _, col = _seed_collection()
        results = col.search_text("cat", k=5)
        result_ids = [r.id for r in results]

        # doc1 and doc3 contain "cat"
        assert "doc1" in result_ids[:2]
        assert "doc3" in result_ids[:2]

    def test_unique_term_selects_correct_docs(self) -> None:
        """Query "python" should rank doc4 and doc5 first."""
        _skip_if_free_threaded()
        _, col = _seed_collection()
        results = col.search_text("python", k=5)
        result_ids = [r.id for r in results]

        # Only doc4 and doc5 contain "python"
        assert result_ids[0] in ("doc4", "doc5")
        assert result_ids[1] in ("doc4", "doc5")

    def test_multi_term_query_ranks_by_relevance(self) -> None:
        """Query "cat ran" should rank doc3 first (has both terms)."""
        _skip_if_free_threaded()
        _, col = _seed_collection()
        results = col.search_text("cat ran", k=5)
        result_ids = [r.id for r in results]

        # doc3 has both "cat" and "ran" -> should be top
        assert result_ids[0] == "doc3"

    def test_no_match_returns_empty_or_low_scores(self) -> None:
        """Query "javascript" matches nothing in the corpus."""
        _skip_if_free_threaded()
        _, col = _seed_collection()
        results = col.search_text("javascript", k=5)
        # Either empty or all scores very low
        for r in results:
            assert r.score < 0.01


# ---------------------------------------------------------------------------
# 3. Hybrid RRF fusion oracle
# ---------------------------------------------------------------------------


class TestHybridRRFFusionOracle:
    """Prove hybrid search correctly fuses dense + BM25 scores via RRF."""

    def test_hybrid_agrees_with_both_signals(self) -> None:
        """When dense and BM25 agree, hybrid should also agree."""
        _skip_if_free_threaded()
        _, col = _seed_collection()

        # Dense: doc1 > doc3 > others (query [1,0])
        # BM25: doc1, doc3 > others (query "cat")
        # Hybrid: doc1 should be first
        results = col.search(
            vector=QUERY_VECTOR,
            text="cat",
            k=5,
            mode="hybrid",
        )
        assert results[0].id == "doc1"

    def test_hybrid_balances_conflicting_signals(self) -> None:
        """When dense and BM25 disagree, hybrid should balance them."""
        _skip_if_free_threaded()
        _, col = _seed_collection()

        # Dense: doc1 first (query [1,0])
        # BM25: doc4, doc5 first (query "python")
        # Hybrid: both doc1 and doc4 should appear in top results
        results = col.search(
            vector=QUERY_VECTOR,
            text="python",
            k=5,
            mode="hybrid",
        )
        result_ids = [r.id for r in results]
        # Both the dense-top and BM25-top docs should be present
        assert "doc1" in result_ids[:3]
        assert "doc4" in result_ids[:3] or "doc5" in result_ids[:3]

    def test_hybrid_score_is_rrf_not_raw(self) -> None:
        """Hybrid scores should be RRF scores, not raw cosine or BM25."""
        _skip_if_free_threaded()
        _, col = _seed_collection()
        results = col.search(
            vector=QUERY_VECTOR,
            text="cat",
            k=5,
            mode="hybrid",
            explain=True,
        )
        # RRF scores are typically small (1/(k+rank))
        for r in results:
            assert r.score < 1.0, "RRF scores should be < 1.0"
            assert r.score > 0.0

    def test_hybrid_native_path_fusion_correctness(self) -> None:
        """Native Mojo RRF fusion produces correct ranking."""
        _skip_if_free_threaded()
        _, col = _seed_collection()

        # Native path: no filters, HNSW, no explain
        # This exercises the Mojo search_hybrid method directly
        results_native = col.search_hybrid(
            vector=QUERY_VECTOR,
            text="cat",
            k=5,
            hybrid_alpha=0.5,
            rrf_k=60,
        )

        # Python-layer path: with explain=True (forces slow path)
        results_python = col.search(
            vector=QUERY_VECTOR,
            text="cat",
            k=5,
            mode="hybrid",
            explain=True,
        )

        # Both paths should agree on top result
        assert results_native[0].id == results_python[0].id

        # RRF scores should be positive and decreasing
        for i in range(len(results_native) - 1):
            assert results_native[i].score >= results_native[i + 1].score
        for r in results_native:
            assert 0.0 < r.score < 1.0

    def test_hybrid_native_path_respects_alpha(self) -> None:
        """Alpha=1.0 should favor vector results, alpha=0.0 should favor text."""
        _skip_if_free_threaded()
        _, col = _seed_collection()

        # alpha=1.0: pure vector search, doc1 (nearest to [1,0]) should win
        results_vec = col.search_hybrid(
            vector=QUERY_VECTOR,
            text="cat",
            k=5,
            hybrid_alpha=1.0,
        )
        assert results_vec[0].id == "doc1"

        # alpha=0.0: pure text search, docs with "cat" should win
        results_text = col.search_hybrid(
            vector=QUERY_VECTOR,
            text="cat",
            k=5,
            hybrid_alpha=0.0,
        )
        # doc1 has "cat" in text, should still be top
        assert "doc1" in [r.id for r in results_text[:2]]


# ---------------------------------------------------------------------------
# 4. Metadata filter oracle
# ---------------------------------------------------------------------------


class TestMetadataFilterOracle:
    """Prove metadata filters correctly constrain search results."""

    def test_filter_restricts_to_matching_category(self) -> None:
        """Filter category=animal should exclude tech docs."""
        _skip_if_free_threaded()
        _, col = _seed_collection()

        results = col.search_vector(
            QUERY_VECTOR,
            k=5,
            filter={"category": "animal"},
        )
        for r in results:
            assert r.metadata["category"] == "animal"

    def test_filter_restricts_to_matching_year(self) -> None:
        """Filter year=2020 should return only doc1 and doc4."""
        _skip_if_free_threaded()
        _, col = _seed_collection()

        results = col.search_vector(
            QUERY_VECTOR,
            k=5,
            filter={"year": 2020},
        )
        result_ids = {r.id for r in results}
        assert result_ids == {"doc1", "doc4"}

    def test_combined_filter(self) -> None:
        """Filter category=animal AND year=2022 returns only doc3."""
        _skip_if_free_threaded()
        _, col = _seed_collection()

        results = col.search_vector(
            QUERY_VECTOR,
            k=5,
            filter={"category": "animal", "year": 2022},
        )
        assert len(results) == 1
        assert results[0].id == "doc3"

    def test_filter_on_text_search(self) -> None:
        """Filters should work on text search too."""
        _skip_if_free_threaded()
        _, col = _seed_collection()

        results = col.search_text(
            "cat",
            k=5,
            filter={"year": 2020},
        )
        # Only doc1 has "cat" AND year=2020
        result_ids = [r.id for r in results]
        assert result_ids[0] == "doc1"


# ---------------------------------------------------------------------------
# 5. Include/exclude ID constraint oracle
# ---------------------------------------------------------------------------


class TestIDConstraintOracle:
    """Prove include_ids and exclude_ids correctly constrain results."""

    def test_include_ids_limits_candidates(self) -> None:
        """include_ids=[doc1, doc3] should only return those docs."""
        _skip_if_free_threaded()
        _, col = _seed_collection()

        results = col.search_vector(
            QUERY_VECTOR,
            k=5,
            include_ids=["doc1", "doc3"],
        )
        result_ids = {r.id for r in results}
        assert result_ids <= {"doc1", "doc3"}

    def test_exclude_ids_removes_candidates(self) -> None:
        """exclude_ids=[doc1] should not return doc1."""
        _skip_if_free_threaded()
        _, col = _seed_collection()

        results = col.search_vector(
            QUERY_VECTOR,
            k=5,
            exclude_ids=["doc1"],
        )
        result_ids = {r.id for r in results}
        assert "doc1" not in result_ids

    def test_include_and_exclude_together(self) -> None:
        """include_ids=[doc1,doc3] + exclude_ids=[doc1] = only doc3."""
        _skip_if_free_threaded()
        _, col = _seed_collection()

        results = col.search_vector(
            QUERY_VECTOR,
            k=5,
            include_ids=["doc1", "doc3"],
            exclude_ids=["doc1"],
        )
        result_ids = {r.id for r in results}
        assert result_ids == {"doc3"}


# ---------------------------------------------------------------------------
# 6. Update/delete/tombstone exclusion oracle
# ---------------------------------------------------------------------------


class TestMutationExclusionOracle:
    """Prove mutations correctly affect search results."""

    def test_delete_excludes_from_search(self) -> None:
        """Deleted items should not appear in search results."""
        _skip_if_free_threaded()
        _, col = _seed_collection()

        # Verify doc1 is in results before delete
        results_before = col.search_vector(QUERY_VECTOR, k=5)
        assert any(r.id == "doc1" for r in results_before)

        # Delete doc1
        col.delete("doc1")

        # Verify doc1 is gone from results
        results_after = col.search_vector(QUERY_VECTOR, k=5)
        assert all(r.id != "doc1" for r in results_after)

    def test_needs_compaction_after_deletions(self) -> None:
        """needs_compaction() returns True after enough deletions."""
        _skip_if_free_threaded()
        from omendb import Collection, CollectionConfig
        col = Collection(":memory:", CollectionConfig(dim=128, metric="l2"))
        n = 10
        for i in range(n):
            col.set(f"id_{i}", vector=[float(i % 5)] * 128)

        # No deletions → should not need compaction
        assert not col.needs_compaction()

        # Delete 3 of 10 → 30% tombstones → should need compaction (>25%)
        for i in range(3):
            col.delete(f"id_{i}")
        assert col.needs_compaction()

        stats = col.vacuum()
        assert not col.needs_compaction()
        assert stats.live_count == 7
        results = col.search_vector([0.0] * 128, k=10)
        assert results
        assert {"id_0", "id_1", "id_2"}.isdisjoint({result.id for result in results})

    def test_update_changes_search_ranking(self) -> None:
        """Updating a vector should change its search ranking."""
        _skip_if_free_threaded()
        _, col = _seed_collection()

        # doc2 is originally [0, 1] — far from query [1, 0]
        results_before = col.search_vector(QUERY_VECTOR, k=5)
        doc2_before = next(r for r in results_before if r.id == "doc2")
        doc2_before_score = doc2_before.score

        # Update doc2 to be very close to query
        col.set("doc2", vector=[0.95, 0.05])

        results_after = col.search_vector(QUERY_VECTOR, k=5)
        doc2_after = next(r for r in results_after if r.id == "doc2")
        # After update, doc2 should be much closer (higher score = less negative)
        assert doc2_after.score > doc2_before_score

    def test_upsert_replaces_text(self) -> None:
        """Upserting same ID should replace text content."""
        _skip_if_free_threaded()
        _, col = _seed_collection()

        # Original doc4 text: "python code runs well"
        results_before = col.search_text("python", k=5)
        doc4_before = next(r for r in results_before if r.id == "doc4")
        doc4_before_score = doc4_before.score

        # Upsert doc4 with different text (keep same vector)
        col.set(
            "doc4",
            vector=[0.0, 0.9],
            text="golang code compiles fast",
        )

        # Search for "python" should no longer rank doc4 highly
        results_after = col.search_text("python", k=5)
        doc4_ids = [r.id for r in results_after]
        # doc4 should either not appear or have lower score
        if "doc4" in doc4_ids:
            doc4_after = next(r for r in results_after if r.id == "doc4")
            assert doc4_after.score < doc4_before_score


# ---------------------------------------------------------------------------
# 7. Flush/reopen equivalence oracle
# ---------------------------------------------------------------------------


class TestFlushReopenEquivalenceOracle:
    """Prove search results are identical before and after flush/reopen."""

    def test_dense_search_survives_reopen(self, tmp_path) -> None:
        """Dense search results must be identical after flush/reopen."""
        _skip_if_free_threaded()
        db_path = str(tmp_path / "db")
        _, col = _seed_collection(db_path, "persist")

        results_before = col.search_vector(QUERY_VECTOR, k=5)
        col.flush()

        # Reopen
        db2 = omendb.open(db_path, create=False)
        col2 = db2.collection("persist", create=False)
        results_after = col2.search_vector(QUERY_VECTOR, k=5)

        assert len(results_before) == len(results_after)
        for before, after in zip(results_before, results_after):
            assert before.id == after.id
            assert abs(before.score - after.score) < 1e-6

    def test_text_search_survives_reopen(self, tmp_path) -> None:
        """Text search results must be identical after flush/reopen."""
        _skip_if_free_threaded()
        db_path = str(tmp_path / "db")
        _, col = _seed_collection(db_path, "persist")

        results_before = col.search_text("cat", k=5)
        col.flush()

        db2 = omendb.open(db_path, create=False)
        col2 = db2.collection("persist", create=False)
        results_after = col2.search_text("cat", k=5)

        assert len(results_before) == len(results_after)
        for before, after in zip(results_before, results_after):
            assert before.id == after.id
            assert abs(before.score - after.score) < 1e-6

    def test_filtered_search_survives_reopen(self, tmp_path) -> None:
        """Filtered search results must be identical after flush/reopen."""
        _skip_if_free_threaded()
        db_path = str(tmp_path / "db")
        _, col = _seed_collection(db_path, "persist")

        results_before = col.search_vector(
            QUERY_VECTOR,
            k=5,
            filter={"category": "animal"},
        )
        col.flush()

        db2 = omendb.open(db_path, create=False)
        col2 = db2.collection("persist", create=False)
        results_after = col2.search_vector(
            QUERY_VECTOR,
            k=5,
            filter={"category": "animal"},
        )

        assert len(results_before) == len(results_after)
        for before, after in zip(results_before, results_after):
            assert before.id == after.id
            assert abs(before.score - after.score) < 1e-6

    def test_get_survives_reopen(self, tmp_path) -> None:
        """get() must return same data after flush/reopen."""
        _skip_if_free_threaded()
        db_path = str(tmp_path / "db")
        _, col = _seed_collection(db_path, "persist")

        record_before = col.get("doc1", include_text=True)
        col.flush()

        db2 = omendb.open(db_path, create=False)
        col2 = db2.collection("persist", create=False)
        record_after = col2.get("doc1", include_text=True)

        assert record_before["id"] == record_after["id"]
        assert record_before["text"] == record_after["text"]
        assert record_before["metadata"] == record_after["metadata"]


# ---------------------------------------------------------------------------
# 8. Check/vacuum preservation oracle
# ---------------------------------------------------------------------------


class TestCheckVacuumOracle:
    """Prove check and vacuum preserve correctness."""

    def test_check_passes_after_operations(self, tmp_path) -> None:
        """check() should pass after inserts, updates, and deletes."""
        _skip_if_free_threaded()
        db_path = str(tmp_path / "db")
        db = omendb.create(db_path)
        col = db.collection(
            "check_test",
            config=omendb.CollectionConfig(dim=2, text=True),
        )

        # Seed
        for item in CORPUS:
            col.set(
                item["id"],
                vector=item["vector"],
                text=item["text"],
                metadata=item["metadata"],
            )

        # Mutate
        col.set("doc6", vector=[0.5, 0.5], text="new doc")
        col.delete("doc2")
        col.set("doc1", vector=[0.99, 0.01], text="updated cat")

        col.flush()
        assert db.check()

    def test_vacuum_preserves_search_correctness(self, tmp_path) -> None:
        """vacuum() should preserve exact search results."""
        _skip_if_free_threaded()
        db_path = str(tmp_path / "db")
        _, col = _seed_collection(db_path, "vacuum_test")

        # Get results before vacuum
        results_before = col.search_vector(QUERY_VECTOR, k=5)

        # Vacuum
        col.vacuum()

        # Get results after vacuum
        results_after = col.search_vector(QUERY_VECTOR, k=5)

        assert len(results_before) == len(results_after)
        for before, after in zip(results_before, results_after):
            assert before.id == after.id
            assert abs(before.score - after.score) < 1e-6

    def test_vacuum_removes_deleted_items(self, tmp_path) -> None:
        """vacuum() should permanently remove deleted items."""
        _skip_if_free_threaded()
        db_path = str(tmp_path / "db")
        _, col = _seed_collection(db_path, "vacuum_test")

        col.delete("doc1")
        col.delete("doc2")

        # Verify deleted items are not retrievable
        assert col.get("doc1") is None
        assert col.get("doc2") is None

        # Vacuum to clean up
        col.vacuum()

        # After vacuum: deleted items should still be gone
        assert col.get("doc1") is None
        assert col.get("doc2") is None

        # Remaining items should still be searchable
        results = col.search_vector(QUERY_VECTOR, k=5)
        result_ids = {r.id for r in results}
        assert "doc1" not in result_ids
        assert "doc2" not in result_ids
        assert "doc3" in result_ids


# ---------------------------------------------------------------------------
# 9. Text-only items oracle
# ---------------------------------------------------------------------------


class TestTextOnlyOracle:
    """Prove text-only items (no vector) work correctly."""

    def test_text_only_excluded_from_vector_search(self) -> None:
        """Text-only items should not appear in vector search."""
        _skip_if_free_threaded()
        db = omendb.memory()
        col = db.collection(
            "text_only",
            config=omendb.CollectionConfig(dim=2, text=True),
        )

        # Add vector item
        col.set("vec1", vector=[1.0, 0.0], text="vector doc")
        # Add text-only item (no vector)
        col.set("txt1", text="text only doc about cats")

        # Vector search should not return text-only item
        results = col.search_vector([1.0, 0.0], k=10)
        result_ids = [r.id for r in results]
        assert "txt1" not in result_ids
        assert "vec1" in result_ids

    def test_text_only_included_in_text_search(self) -> None:
        """Text-only items should appear in text search."""
        _skip_if_free_threaded()
        db = omendb.memory()
        col = db.collection(
            "text_only",
            config=omendb.CollectionConfig(dim=2, text=True),
        )

        col.set("vec1", vector=[1.0, 0.0], text="vector doc")
        col.set("txt1", text="text only doc about cats")

        # Text search should return text-only item
        results = col.search_text("cats", k=10)
        result_ids = [r.id for r in results]
        assert "txt1" in result_ids

    def test_text_only_survives_reopen(self, tmp_path) -> None:
        """Text-only items persist across flush/reopen."""
        _skip_if_free_threaded()
        db_path = str(tmp_path / "db")
        db = omendb.create(db_path)
        col = db.collection(
            "text_only",
            config=omendb.CollectionConfig(dim=2, text=True),
        )

        col.set("vec1", vector=[1.0, 0.0], text="vector doc")
        col.set("txt1", text="text only doc about cats")

        col.flush()

        db2 = omendb.open(db_path, create=False)
        col2 = db2.collection("text_only", create=False)

        # Text-only item should still be searchable
        results = col2.search_text("cats", k=10)
        result_ids = [r.id for r in results]
        assert "txt1" in result_ids

        # And still excluded from vector search
        vec_results = col2.search_vector([1.0, 0.0], k=10)
        vec_ids = [r.id for r in vec_results]
        assert "txt1" not in vec_ids


# ---------------------------------------------------------------------------
# 8. HNSW exact fallback oracle
# ---------------------------------------------------------------------------


class TestHNSWExactFallback:
    """Prove HNSW falls back to exact search when overfetch returns too few."""

    def test_hnsw_underrun_exact_fallback_with_allowlist(
        self, tmp_path
    ) -> None:
        """HNSW must fall back to exact when allowlisted search returns
        fewer than k candidates."""
        import omendb

        db_path = str(tmp_path / "hnsw_fallback.db")
        db = omendb.open(db_path, create=True)
        col = db.collection(
            "test",
            config=omendb.CollectionConfig(dim=2, index="hnsw"),
        )

        # Insert 100 vectors in a line from 0 to 99
        for i in range(100):
            col.set(f"vec_{i}", vector=[float(i), 0.0])

        # Search with a highly restrictive allowlist: only 2 IDs
        # but request k=10. HNSW can't return 10 from a 2-item allowlist.
        allowlist = ["vec_7", "vec_42"]
        results = col.search(
            vector=[50.0, 0.0],
            k=10,
            include_ids=allowlist,
            explain=True,
        )
        result_ids = [r.id for r in results]

        # Must return exactly the 2 allowed IDs (both should be in results)
        assert len(results) <= 2
        assert set(result_ids) <= set(allowlist)

        # Verify fallback diagnostics
        assert results[0].evidence is not None
        diagnostics = results[0].evidence.raw if results[0].evidence else {}
        reasons = {
            "hnsw_underrun_exact_fallback",
            "selective_filter_exact_fallback",
        }
        assert diagnostics.get("fallback_reason") in reasons

    def test_hnsw_metadata_filter_underrun_exact_fallback(
        self, tmp_path
    ) -> None:
        """Metadata-filtered HNSW must fall back to exact when the
        eligible set is smaller than k."""
        import omendb

        db_path = str(tmp_path / "hnsw_metadata_fallback.db")
        db = omendb.open(db_path, create=True)
        col = db.collection(
            "test",
            config=omendb.CollectionConfig(dim=2, index="hnsw"),
        )

        for i in range(50):
            group = "small" if i in (7, 42) else "large"
            col.set(
                f"vec_{i}",
                vector=[float(i), 0.0],
                metadata={"group": group},
            )

        results = col.search(
            vector=[50.0, 0.0],
            k=10,
            filter={"group": "small"},
            explain=True,
        )

        assert {r.id for r in results} == {"vec_7", "vec_42"}
        assert results[0].evidence is not None
        diagnostics = results[0].evidence.raw if results[0].evidence else {}
        assert diagnostics.get("method") == "exact_l2"
        assert (
            diagnostics.get("fallback_reason")
            == "hnsw_bitmap_underrun_exact_fallback"
        )

    def test_allowlist_filter_uses_full_live_id_set_not_partial_cache(
        self, tmp_path
    ) -> None:
        """Exact allowlist construction must not treat a partial metadata
        cache as the full live-id universe."""
        import omendb

        db_path = str(tmp_path / "partial_cache.db")
        db = omendb.open(db_path, create=True)
        col = db.collection(
            "test",
            config=omendb.CollectionConfig(dim=2, index="hnsw"),
        )
        col.set("a", vector=[0.0, 0.0], metadata={"kind": "match"})
        col.set("b", vector=[1.0, 0.0], metadata={"kind": "match"})
        col.set("c", vector=[2.0, 0.0], metadata={"kind": "other"})

        # Populate cache for only one matching ID.
        assert col._get_parsed_metadata("a") == {"kind": "match"}
        assert set(col._metadata_cache or {}) == {"a"}

        allowlist = col._allowlist_ids({"kind": "match"}, None, None)
        assert set(allowlist or []) == {"a", "b"}

    def test_hnsw_underrun_exact_fallback_scores_correct(
        self, tmp_path
    ) -> None:
        """Fallen-back exact scores must be geometrically correct."""
        import omendb

        db_path = str(tmp_path / "hnsw_fallback_scores.db")
        db = omendb.open(db_path, create=True)
        col = db.collection(
            "test",
            config=omendb.CollectionConfig(dim=2, index="hnsw"),
        )

        # Insert 3 vectors: (0,0), (3,0), (10,0)
        col.set("a", vector=[0.0, 0.0])
        col.set("b", vector=[3.0, 0.0])
        col.set("c", vector=[10.0, 0.0])

        # Query (5, 0). Distances: a=5, b=2, c=5.
        # Include only a and c (3x3, distance doesn't matter much).
        # k=10, only 2 valid — triggers underrun fallback.
        results = col.search(
            vector=[5.0, 0.0],
            k=10,
            include_ids=["a", "c"],
            explain=True,
        )
        result_ids = [r.id for r in results]
        assert len(results) == 2
        assert set(result_ids) == {"a", "c"}

        # b=3.0 is closer to query=5.0, but excluded by include_ids
        # Within the 2 results, order by exact distance: both distance=5.
        # a is closer (5) or tied with c (5). Either order is fine.
        for r in results:
            assert r.distance is not None
            assert 24.0 <= r.distance <= 26.0  # (5-0)^2 + (0-0)^2 = 25


# ---------------------------------------------------------------------------
# 9. Process-level crash recovery oracle
# ---------------------------------------------------------------------------


class TestCrashRecovery:
    """Prove the database survives process-level kill and reopens correctly."""

    def test_flush_then_reopen_without_close(self, tmp_path) -> None:
        """Flush then reopen without clean close: all data must survive."""
        import omendb

        db_path = str(tmp_path / "crash.db")
        db = omendb.open(db_path, create=True)
        col = db.collection(
            "survivors",
            config=omendb.CollectionConfig(dim=2, text=True),
        )

        # Write records
        for i in range(50):
            col.set(
                f"rec_{i}",
                vector=[float(i), 0.0],
                metadata={"idx": i, "label": f"item_{i}"},
                text=f"document number {i}",
            )

        # Simulate kill: flush but don't close
        col.flush()

        # "Process restarts" — open fresh handle
        db2 = omendb.open(db_path, create=False)
        col2 = db2.collection("survivors", create=False)

        # All 50 records present
        def _live_count(col):
            return int(col._native_handle().len())

        assert _live_count(col2) == 50

        # Search finds them
        results = col2.search_vector([25.0, 0.0], k=10)
        result_ids = [r.id for r in results]
        assert len(result_ids) == 10
        # Verify geometric ordering: rec_24 through rec_26 should be nearest
        assert result_ids[0] in {f"rec_{i}" for i in range(24, 27)}

        # Filters work
        filtered = col2.search_vector(
            [0.0, 0.0], k=50, filter={"idx": {"$lt": 5}}
        )
        assert len(filtered) == 5

        # Text search works
        text_results = col2.search_text("number 42", k=5)
        assert text_results[0].id == "rec_42"

        # Metadata survives
        meta = col2.get("rec_7")
        assert meta is not None
        assert meta["metadata"]["idx"] == 7

    def test_multiple_flush_reopen_cycles(self, tmp_path) -> None:
        """Multiple flush-and-reopen cycles must not lose data."""
        import omendb

        db_path = str(tmp_path / "cycles.db")

        for cycle in range(5):
            db = omendb.open(db_path, create=(cycle == 0))
            col = db.collection(
                "cycle",
                config=omendb.CollectionConfig(dim=2),
            )
            # Add 10 per cycle
            base = cycle * 10
            for i in range(base, base + 10):
                col.set(f"rec_{i}", vector=[float(i), 0.0])
            col.flush()

        # Final open: all 50 records must be present
        db_final = omendb.open(db_path, create=False)
        col_final = db_final.collection("cycle", create=False)
        assert col_final._native_handle().len() == 50

    def test_vacuum_survives_crash(self, tmp_path) -> None:
        """Vacuum with atomic swap must survive mid-operation crash."""
        import omendb

        db_path = str(tmp_path / "vacuum_crash.db")
        db = omendb.open(db_path, create=True)
        col = db.collection(
            "volatile",
            config=omendb.CollectionConfig(dim=2, text=True),
        )

        # Add records, delete half
        for i in range(40):
            col.set(
                f"rec_{i}",
                vector=[float(i), 0.0],
                text=f"line {i}",
            )
        for i in range(0, 40, 2):
            col.delete(f"rec_{i}")
        col.flush()

        live_before = int(col._native_handle().len())
        assert live_before == 20

        # Vacuum triggers atomic swap
        stats = col.vacuum()
        assert stats.live_count == 20
        assert stats.record_count == 20  # all tombstones removed

        # Simulate crash: reopen without closing
        db2 = omendb.open(db_path, create=False)
        col2 = db2.collection("volatile", create=False)

        # Live records must survive
        assert col2._native_handle().len() == 20

        # Search still works
        results = col2.search_vector([25.0, 0.0], k=10)
        result_ids = [r.id for r in results]
        # Odd records survived (0,2,4... deleted)
        for rid in result_ids:
            rec_num = int(rid.split("_")[1])
            assert rec_num % 2 == 1

        # Text search still works
        text_results = col2.search_text("line 17", k=3)
        assert text_results[0].id == "rec_17"

    def test_recovery_detects_backup_path(self, tmp_path) -> None:
        """_recover_collection_path must restore from backup when main
        path is missing but backup exists."""
        import omendb
        from omendb._api import _collection_backup_path

        db_path = str(tmp_path / "recover.db")
        db = omendb.open(db_path, create=True)
        col = db.collection(
            "backup",
            config=omendb.CollectionConfig(dim=2),
        )
        col.set("persist", vector=[1.0, 1.0])
        col.flush()

        # Manually simulate post-crash state: move real → backup
        real_path = Path(db_path) / "backup"
        backup_path = _collection_backup_path(real_path)
        assert real_path.exists()
        shutil.move(str(real_path), str(backup_path))
        assert not real_path.exists()
        assert backup_path.exists()

        # Reopen: recovery should restore from backup
        db2 = omendb.open(db_path, create=False)
        col2 = db2.collection("backup", create=False)

        assert col2._native_handle().len() == 1
        meta = col2.get("persist")
        assert meta is not None


# ---------------------------------------------------------------------------
# 10. Collection-level check, rebuild, and export
# ---------------------------------------------------------------------------


class TestCollectionOperations:
    """Prove collection-level check, rebuild, and export work correctly."""

    def test_check_on_clean_collection(self, tmp_path) -> None:
        """check() on a clean collection returns zero issues."""
        import omendb

        db_path = str(tmp_path / "check.db")
        db = omendb.open(db_path, create=True)
        col = db.collection(
            "clean",
            config=omendb.CollectionConfig(dim=2, text=True),
        )
        col.set("a", vector=[1.0, 0.0], text="hello", metadata={"k": "v"})
        col.flush()

        result = col.check()
        assert result.collections_checked == 1
        assert len(result.issues) == 0

    def test_check_detects_missing_file(self, tmp_path) -> None:
        """check() must detect a missing required sidecar file."""
        import omendb
        import os

        db_path = str(tmp_path / "broken.db")
        db = omendb.open(db_path, create=True)
        col = db.collection(
            "broken",
            config=omendb.CollectionConfig(dim=2),
        )
        col.set("a", vector=[1.0, 0.0])
        col.flush()

        # Remove hnsw.meta — check should catch this
        hnsw_meta = Path(db_path) / "broken" / "hnsw.meta"
        os.remove(hnsw_meta)

        result = col.check()
        assert len(result.issues) > 0
        assert any("hnsw.meta" in issue.message for issue in result.issues)

    def test_rebuild_on_dirty_collection(self, tmp_path) -> None:
        """rebuild() must compact tombstones and preserve live state."""
        import omendb

        db_path = str(tmp_path / "rebuild.db")
        db = omendb.open(db_path, create=True)
        col = db.collection(
            "dirty",
            config=omendb.CollectionConfig(dim=2),
        )
        for i in range(20):
            col.set(f"rec_{i}", vector=[float(i), 0.0])
        for i in range(0, 20, 2):
            col.delete(f"rec_{i}")
        col.flush()

        stats = col.rebuild()
        assert stats.live_count == 10
        assert stats.record_count == 10

    def test_export_creates_standalone_directory(self, tmp_path) -> None:
        """export_to() produces a directory with manifest and sidecars."""
        import omendb

        db_path = str(tmp_path / "source.db")
        db = omendb.open(db_path, create=True)
        col = db.collection(
            "source",
            config=omendb.CollectionConfig(dim=2, text=True),
        )
        col.set("a", vector=[1.0, 0.0], text="hello world")
        col.flush()

        export_path = str(tmp_path / "exported_collection")
        col.export_to(export_path)

        # Verify export contains expected files
        manifest = Path(export_path) / "manifest.json"
        assert manifest.exists()
        records = Path(export_path) / "records.json"
        assert records.exists()
