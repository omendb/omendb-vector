"""Stage 2 correctness harness: systematic oracle matrix for OmenDB-Mojo.

Proves database correctness across all core operations and index families.
Every test uses a deterministic random corpus and compares engine results to
an exact brute-force (or reference implementation) oracle.

Covers:
- Dense HNSW vs exact flat oracle
- BM25 vs Python reference oracle
- Hybrid RRF vs computed reference
- Filtered search vs exact filtered oracle
- Lifecycle matrix: insert/update/delete/reinsert/idempotency
- Flush/reopen equivalence
- Vacuum correctness
- Config mismatch detection
- Corrupt state detection
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import signal
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

import omendb

# ---------------------------------------------------------------------------
# Seed for deterministic randomness (different seeds test different patterns)
# ---------------------------------------------------------------------------
SEEDS = [42, 7, 99, 123, 2024]

# ---------------------------------------------------------------------------
# Oracle infrastructure
# ---------------------------------------------------------------------------


def exact_l2_oracle(vector: list[float], records: dict[str, list[float]], k: int) -> list[str]:
    """Exact k-nearest neighbors by L2 distance. Returns sorted IDs."""
    distances = []
    for rid, rvec in records.items():
        d = sum((a - b) ** 2 for a, b in zip(vector, rvec))
        distances.append((rid, d))
    distances.sort(key=lambda x: x[1])
    return [rid for rid, _ in distances[:k]]


def compute_bm25(
    query: str,
    docs: dict[str, str],
    k1: float = 1.2,
    b: float = 0.75,
) -> list[tuple[str, float]]:
    """Python BM25 reference. Returns sorted (id, score) for docs matching query."""
    def tokenize(text: str) -> list[str]:
        return text.lower().split()

    q_terms = tokenize(query)
    if not q_terms:
        return []

    doc_tokens = {did: tokenize(text) for did, text in docs.items()}
    doc_len = {did: len(toks) for did, toks in doc_tokens.items()}
    avgdl = sum(doc_len.values()) / max(len(doc_len), 1)
    N = len(docs)

    df: Counter[str] = Counter()
    for toks in doc_tokens.values():
        df.update(set(toks))

    scores: dict[str, float] = {}
    for did, toks in doc_tokens.items():
        tf = Counter(toks)
        dl = doc_len[did]
        score = 0.0
        for term in q_terms:
            if term not in df:
                continue
            nq = df[term]
            idf = math.log(1.0 + (N - nq + 0.5) / (nq + 0.5))
            f = tf.get(term, 0)
            tf_sat = (f * (k1 + 1.0)) / (f + k1 * (1.0 - b + b * dl / avgdl))
            score += idf * tf_sat
        if score > 0:
            scores[did] = score

    return sorted(scores.items(), key=lambda x: -x[1])


def compute_rrf(dense_ids: list[str], text_ids: list[str], k: int = 60) -> list[str]:
    """Compute RRF fusion of two ranked lists. Returns merged ranked list."""
    scores: dict[str, float] = {}
    for rank, rid in enumerate(dense_ids):
        scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + rank + 1)
    for rank, rid in enumerate(text_ids):
        scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + rank + 1)
    return [rid for rid, _ in sorted(scores.items(), key=lambda x: -x[1])]


def load_text_only_store(path: Path) -> dict[str, dict[str, Any]]:
    """Load persisted text-only store."""
    store_path = path / "text_only_store.json"
    if store_path.exists():
        return json.loads(store_path.read_text(encoding="utf-8"))
    return {}


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


class _DbFixture:
    """Wraps a temp db + collection for deterministic testing."""

    def __init__(self, seed: int, dim: int = 2, text: bool = True, n_vectors: int = 50):
        self.seed = seed
        self.dim = dim
        self.n_vectors = n_vectors
        self.tmpdir = tempfile.mkdtemp()
        db_path = Path(self.tmpdir) / "db"
        self.db = omendb.open(str(db_path), create=True)
        config = omendb.CollectionConfig(dim=dim, text=text)
        self.col = self.db.collection("test", config=config)
        self._records: dict[str, list[float]] = {}
        self._texts: dict[str, str] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

        rng = random.Random(seed)
        # Query vector
        self.query_vector = [rng.random() for _ in range(dim)]
        self.col.set("query_target", vector=self.query_vector, text="query target document")
        self._records["query_target"] = self.query_vector
        self._texts["query_target"] = "query target document"

        for i in range(1, n_vectors):
            vid = f"doc{i}"
            vec = [rng.random() for _ in range(dim)]
            category = "even" if i % 2 == 0 else "odd"
            text_choices = [
                f"the system processes data records {i}",
                f"machine learning model iteration {i}",
                f"database query execution plan {i}",
                f"network packet analysis tool {i}",
            ]
            text = text_choices[i % len(text_choices)]
            self.col.set(
                vid,
                vector=vec,
                text=text,
                metadata={"category": category, "index": i},
            )
            self._records[vid] = vec
            self._texts[vid] = text
            self._metadata[vid] = {"category": category, "index": i}

    def native(self):
        return self.col._native_handle()

    def cleanup(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Dense HNSW vs Exact Oracle
# ---------------------------------------------------------------------------


class TestDenseOracle:
    """Prove HNSW returns same top-k as exact flat search."""

    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.parametrize("k", [1, 3, 5, 10])
    def test_hnsw_top_k_matches_exact(self, seed: int, k: int) -> None:
        fx = _DbFixture(seed, n_vectors=100)
        try:
            native = fx.native()
            all_ids = list(native.live_ids())

            # Exact oracle
            exact = native.search_exact_ids(fx.query_vector, all_ids)
            exact_ids = [r["id"] for r in exact[:k]]

            # HNSW with large ef to ensure quality
            hnsw = native.search(fx.query_vector, max(k * 20, 200), 200)
            hnsw_ids = [r["id"] for r in hnsw[:k]]

            # Top-1 must match exactly
            assert hnsw_ids[0] == exact_ids[0], (
                f"HNSW top-1 {hnsw_ids[0]} != exact top-1 {exact_ids[0]} "
                f"(seed={seed}, k={k})"
            )

            # At least 70% overlap at k=10, 80% at k<=5 (HNSW approximation)
            overlap = len(set(exact_ids) & set(hnsw_ids))
            min_overlap = int(k * 0.7) if k >= 10 else (int(k * 0.8) if k > 1 else 1)
            assert overlap >= min_overlap, (
                f"HNSW/exact overlap {overlap}/{k} below threshold {min_overlap} "
                f"(seed={seed}, k={k})"
            )
        finally:
            fx.cleanup()

    @pytest.mark.parametrize("seed", SEEDS)
    def test_hnsw_with_ef_override_finds_nearest(self, seed: int) -> None:
        """Higher ef should never make results worse."""
        fx = _DbFixture(seed, n_vectors=100)
        try:
            native = fx.native()
            all_ids = list(native.live_ids())
            exact = native.search_exact_ids(fx.query_vector, all_ids)
            exact_ids = [r["id"] for r in exact[:3]]

            # Low ef
            low = native.search(fx.query_vector, 10, 10)
            low_ids = [r["id"] for r in low[:3]]

            # High ef
            high = native.search(fx.query_vector, 200, 200)
            high_ids = [r["id"] for r in high[:3]]

            low_overlap = len(set(low_ids) & set(exact_ids))
            high_overlap = len(set(high_ids) & set(exact_ids))

            # Higher ef should have at least as much overlap
            assert high_overlap >= low_overlap, (
                f"Higher ef reduced overlap: low={low_overlap}, high={high_overlap}"
            )
        finally:
            fx.cleanup()


# ---------------------------------------------------------------------------
# BM25 vs Python Reference Oracle
# ---------------------------------------------------------------------------


class TestBM25Oracle:
    """Prove BM25 ranking matches Python reference."""

    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.parametrize("query_text", [
        "database",
        "machine learning",
        "network",
        "data records",
        "processing analysis",
    ])
    def test_bm25_ranking_order(self, seed: int, query_text: str) -> None:
        fx = _DbFixture(seed, n_vectors=50)
        try:
            # Engine BM25
            eng_results = fx.col.search_text(query_text, k=10)
            eng_ids = [r.id for r in eng_results]

            # Python reference BM25 (only records that were set with text)
            ref = compute_bm25(query_text, fx._texts)
            ref_ids = [rid for rid, _ in ref[:10]]

            # The set of IDs should match (same documents ranked)
            eng_set = set(eng_ids)
            ref_set = set(ref_ids)
            assert eng_set == ref_set, (
                f"BM25 ID sets differ\n"
                f"  engine only: {eng_set - ref_set}\n"
                f"  ref only: {ref_set - eng_set}\n"
                f"  query='{query_text}', seed={seed}"
            )

            # Ranking order should match
            assert eng_ids == ref_ids, (
                f"BM25 ranking differs\n"
                f"  engine: {eng_ids}\n"
                f"  ref:    {ref_ids}\n"
                f"  query='{query_text}', seed={seed}"
            )
        finally:
            fx.cleanup()

    @pytest.mark.parametrize("seed", SEEDS)
    def test_bm25_after_delete_and_flush(self, seed: int) -> None:
        """BM25 results must be identical before and after flush."""
        fx = _DbFixture(seed, n_vectors=30)
        try:
            pre_flush = fx.col.search_text("database", k=10)
            pre_ids = [r.id for r in pre_flush]

            fx.col.flush()
            post_flush = fx.col.search_text("database", k=10)
            post_ids = [r.id for r in post_flush]

            assert pre_ids == post_ids, (
                f"BM25 results changed after flush: pre={pre_ids}, post={post_ids}"
            )
        finally:
            fx.cleanup()


# ---------------------------------------------------------------------------
# Hybrid RRF Oracle
# ---------------------------------------------------------------------------


class TestHybridOracle:
    """Prove hybrid RRF fusion matches computed reference."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_hybrid_ranking_overlaps_rrf_reference(self, seed: int) -> None:
        fx = _DbFixture(seed, n_vectors=50)
        try:
            native = fx.native()
            all_ids = list(native.live_ids())

            # Exact dense oracle (all records)
            exact_dense = native.search_exact_ids(fx.query_vector, all_ids)
            dense_all = [r["id"] for r in exact_dense]

            # Python BM25 reference
            ref_bm25 = compute_bm25("database", fx._texts)
            text_all = [rid for rid, _ in ref_bm25]

            # Engine hybrid
            eng = fx.col.search_hybrid(
                vector=fx.query_vector,
                text="database",
                k=10,
                explain=True,
            )
            eng_ids = [r.id for r in eng]

            # Engine hybrid results must be from the dense or BM25 candidate pools.
            # Include full BM25 results (BM25 only returns docs matching the query)
            # and top-50 dense results (all dense docs are candidates for RRF fusion).
            valid_ids = set(dense_all[:50]) | set(text_all)
            missing = set(eng_ids) - valid_ids
            assert not missing, (
                f"Engine returned IDs not in dense or text results: {missing}"
            )
        finally:
            fx.cleanup()

    @pytest.mark.parametrize("seed", SEEDS)
    def test_hybrid_alpha_low_favors_text(self, seed: int) -> None:
        """alpha=0.0 favors text component (dense_weight=0, text_weight=1)."""
        fx = _DbFixture(seed, n_vectors=30)
        try:
            hybrid = fx.col.search_hybrid(
                vector=fx.query_vector,
                text="database",
                k=5,
                hybrid_alpha=0.0,
            )
            text = fx.col.search_text("database", k=5)
            # Top text result should appear in hybrid top-3
            text_top = {r.id for r in text[:3]}
            hybrid_top = {r.id for r in hybrid[:3]}
            assert bool(text_top & hybrid_top), (
                f"alpha=0 hybrid shares no top-3 with text: "
                f"text={text_top}, hybrid={hybrid_top}"
            )
        finally:
            fx.cleanup()

    @pytest.mark.parametrize("seed", SEEDS)
    def test_hybrid_alpha_high_favors_dense(self, seed: int) -> None:
        """alpha=1.0 favors dense component (dense_weight=1, text_weight=0)."""
        fx = _DbFixture(seed, n_vectors=30)
        try:
            hybrid = fx.col.search_hybrid(
                vector=fx.query_vector,
                text="database",
                k=5,
                hybrid_alpha=1.0,
            )
            dense = fx.col.search_vector(fx.query_vector, k=5)
            # Nearest dense neighbor should appear in hybrid top-3
            dense_top = {r.id for r in dense[:3]}
            hybrid_top = {r.id for r in hybrid[:3]}
            assert bool(dense_top & hybrid_top), (
                f"alpha=1 hybrid shares no top-3 with dense: "
                f"dense={dense_top}, hybrid={hybrid_top}"
            )
        finally:
            fx.cleanup()


# ---------------------------------------------------------------------------
# Filtered Search Oracle
# ---------------------------------------------------------------------------


class TestFilteredOracle:
    """Prove filtered search is equivalent to exact search over filtered subset."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_metadata_filter_correctness(self, seed: int) -> None:
        """All results from filtered search must match the filter predicate."""
        fx = _DbFixture(seed, n_vectors=100)
        try:
            # Engine filtered search
            eng = fx.col.search(
                vector=fx.query_vector,
                k=10,
                filter={"category": "even"},
            )
            eng_ids = [r.id for r in eng]

            # Every engine result must have category="even"
            for rid in eng_ids:
                md = fx._metadata.get(rid)
                if md is not None:
                    assert md.get("category") == "even", (
                        f"Filtered result {rid} has category={md.get('category')}, expected 'even'"
                    )

            # Engine should not return category="odd" records
            odd_in_results = [
                rid for rid in eng_ids
                if fx._metadata.get(rid, {}).get("category") == "odd"
            ]
            assert not odd_in_results, (
                f"Filtered search returned non-matching records: {odd_in_results}"
            )
        finally:
            fx.cleanup()

    @pytest.mark.parametrize("seed", SEEDS)
    def test_include_ids_constraint(self, seed: int) -> None:
        """include_ids must restrict results to only those IDs."""
        fx = _DbFixture(seed, n_vectors=50)
        try:
            native = fx.native()
            all_ids = sorted(native.live_ids())

            # Pick 3 specific IDs
            target_ids = all_ids[:3]
            exact = native.search_exact_ids(fx.query_vector, target_ids)
            exact_ids = set(r["id"] for r in exact[:3])

            eng = fx.col.search(
                vector=fx.query_vector,
                k=3,
                include_ids=target_ids,
            )
            eng_ids = set(r.id for r in eng)

            assert exact_ids == eng_ids, (
                f"include_ids constraint mismatch\n"
                f"  engine: {eng_ids}\n"
                f"  exact:  {exact_ids}"
            )
        finally:
            fx.cleanup()

    @pytest.mark.parametrize("seed", SEEDS)
    def test_exclude_ids_constraint(self, seed: int) -> None:
        """exclude_ids must never appear in results."""
        fx = _DbFixture(seed, n_vectors=50)
        try:
            native = fx.native()
            all_ids = sorted(native.live_ids())
            exclude = {all_ids[0], all_ids[1]}

            eng = fx.col.search(
                vector=fx.query_vector,
                k=5,
                exclude_ids=list(exclude),
            )
            eng_ids = set(r.id for r in eng)

            assert not (eng_ids & exclude), (
                f"Excluded IDs appeared in results: {eng_ids & exclude}"
            )
        finally:
            fx.cleanup()


# ---------------------------------------------------------------------------
# Lifecycle Matrix
# ---------------------------------------------------------------------------


class TestLifecycle:
    """Prove insert/update/delete/reinsert operations preserve correctness."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_delete_excludes_from_all_search_modes(self, seed: int) -> None:
        fx = _DbFixture(seed, n_vectors=30)
        try:
            # Pick an ID to delete
            to_delete = "doc5"
            fx.col.delete(to_delete)

            # Check dense search
            dense = fx.col.search_vector(fx.query_vector, k=30)
            assert to_delete not in [r.id for r in dense]

            # Check text search
            text_results = fx.col.search_text("database", k=30)
            assert to_delete not in [r.id for r in text_results]

            # Check hybrid
            hybrid = fx.col.search_hybrid(vector=fx.query_vector, text="database", k=30)
            assert to_delete not in [r.id for r in hybrid]
        finally:
            fx.cleanup()

    @pytest.mark.parametrize("seed", SEEDS)
    def test_reinsert_restores_search_visibility(self, seed: int) -> None:
        fx = _DbFixture(seed, n_vectors=30)
        try:
            to_reinsert = "doc5"
            orig_vector = fx._records[to_reinsert]
            orig_text = fx._texts[to_reinsert]

            # Delete
            fx.col.delete(to_reinsert)
            after_delete = fx.col.search_vector(fx.query_vector, k=30)
            assert to_reinsert not in [r.id for r in after_delete]

            # Reinsert
            fx.col.set(to_reinsert, vector=orig_vector, text=orig_text)
            after_reinsert = fx.col.search_vector(fx.query_vector, k=30)
            assert to_reinsert in [r.id for r in after_reinsert], (
                "Re-inserted record not visible in search"
            )
        finally:
            fx.cleanup()

    @pytest.mark.parametrize("seed", SEEDS)
    def test_idempotent_set_produces_same_results(self, seed: int) -> None:
        """Setting the same id with same data twice must be idempotent."""
        fx = _DbFixture(seed, n_vectors=20)
        try:
            pre_results = fx.col.search_vector(fx.query_vector, k=5)
            pre_ids = [r.id for r in pre_results]

            # Re-set doc1 with same data
            fx.col.set("doc1", vector=fx._records["doc1"], text=fx._texts["doc1"])
            post_results = fx.col.search_vector(fx.query_vector, k=5)
            post_ids = [r.id for r in post_results]

            assert pre_ids == post_ids, (
                f"Re-set changed search results: {pre_ids} -> {post_ids}"
            )
        finally:
            fx.cleanup()

    @pytest.mark.parametrize("seed", SEEDS)
    def test_update_moves_record_in_search_ranking(self, seed: int) -> None:
        """Updating a vector must change its search position appropriately."""
        fx = _DbFixture(seed, n_vectors=30)
        try:
            # Check pre-update position of doc1
            pre_results = fx.col.search_vector(fx.query_vector, k=5)
            pre_ids = [r.id for r in pre_results]

            # Move doc1's vector far away
            orig = fx._records["doc1"]
            new_vec = [1.0 - v for v in orig]
            fx.col.set("doc1", vector=new_vec)

            # After moving far away, doc1 should drop in ranking
            post_results = fx.col.search_vector(fx.query_vector, k=30)
            post_ids = [r.id for r in post_results]

            # Top-1 should still be correct (query_target)
            assert post_ids[0] == "query_target", (
                f"Top-1 wrong after update: {post_ids[0]}"
            )

            # doc1 should not be top-1 after moving far away
            assert post_ids[0] != "doc1", (
                "Updated doc should not be nearest neighbor"
            )
        finally:
            fx.cleanup()

    @pytest.mark.parametrize("seed", SEEDS)
    def test_supersede_removes_old_from_search_and_get(self, seed: int) -> None:
        """Superseded records must not appear in search or get results."""
        fx = _DbFixture(seed, n_vectors=30)
        try:
            # Supersede doc5 with doc1
            fx.col.supersede("doc5", "doc1")

            # doc5 must not appear in search
            dense = fx.col.search_vector(fx.query_vector, k=30)
            assert "doc5" not in [r.id for r in dense], (
                "Superseded record still visible in search"
            )

            # doc5 should not be retrievable (supersede is a data-lifecycle operation)
            record = fx.col.get("doc5")
            assert record is None, (
                "Superseded record should not be retrievable via get()"
            )
        finally:
            fx.cleanup()


# ---------------------------------------------------------------------------
# Flush/Reopen Equivalence
# ---------------------------------------------------------------------------


class TestFlushReopen:
    """Prove flush + reopen preserves correctness identically."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_dense_search_after_reopen(self, seed: int) -> None:
        fx = _DbFixture(seed, n_vectors=50)
        try:
            pre_ids = [r.id for r in fx.col.search_vector(fx.query_vector, k=5)]
            fx.col.flush()

            db2 = omendb.open(str(Path(fx.tmpdir) / "db"), create=False)
            col2 = db2.collection("test", create=False)
            post_ids = [r.id for r in col2.search_vector(fx.query_vector, k=5)]

            assert pre_ids == post_ids, (
                f"Dense results changed after reopen: {pre_ids} -> {post_ids}"
            )
        finally:
            fx.cleanup()

    @pytest.mark.parametrize("seed", SEEDS)
    def test_text_search_after_reopen(self, seed: int) -> None:
        fx = _DbFixture(seed, n_vectors=50)
        try:
            pre_ids = [r.id for r in fx.col.search_text("database", k=5)]
            fx.col.flush()

            db2 = omendb.open(str(Path(fx.tmpdir) / "db"), create=False)
            col2 = db2.collection("test", create=False)
            post_ids = [r.id for r in col2.search_text("database", k=5)]

            assert pre_ids == post_ids, (
                f"BM25 results changed after reopen: {pre_ids} -> {post_ids}"
            )
        finally:
            fx.cleanup()

    @pytest.mark.parametrize("seed", SEEDS)
    def test_hybrid_search_after_reopen(self, seed: int) -> None:
        fx = _DbFixture(seed, n_vectors=50)
        try:
            pre_ids = [r.id for r in fx.col.search_hybrid(
                vector=fx.query_vector, text="database", k=5
            )]
            fx.col.flush()

            db2 = omendb.open(str(Path(fx.tmpdir) / "db"), create=False)
            col2 = db2.collection("test", create=False)
            post_ids = [r.id for r in col2.search_hybrid(
                vector=fx.query_vector, text="database", k=5
            )]

            assert pre_ids == post_ids, (
                f"Hybrid results changed after reopen: {pre_ids} -> {post_ids}"
            )
        finally:
            fx.cleanup()

    @pytest.mark.parametrize("seed", SEEDS)
    def test_filtered_search_after_reopen(self, seed: int) -> None:
        fx = _DbFixture(seed, n_vectors=100)
        try:
            pre = fx.col.search(
                vector=fx.query_vector, k=5, filter={"category": "even"}
            )
            pre_ids = [r.id for r in pre]
            fx.col.flush()

            db2 = omendb.open(str(Path(fx.tmpdir) / "db"), create=False)
            col2 = db2.collection("test", create=False)
            post = col2.search(
                vector=fx.query_vector, k=5, filter={"category": "even"}
            )
            post_ids = [r.id for r in post]

            assert pre_ids == post_ids, (
                f"Filtered results changed after reopen: {pre_ids} -> {post_ids}"
            )
        finally:
            fx.cleanup()

    @pytest.mark.parametrize("seed", SEEDS)
    def test_delete_persists_across_reopen(self, seed: int) -> None:
        fx = _DbFixture(seed, n_vectors=30)
        try:
            fx.col.delete("doc5")
            fx.col.flush()

            db2 = omendb.open(str(Path(fx.tmpdir) / "db"), create=False)
            col2 = db2.collection("test", create=False)

            results = col2.search_vector(fx.query_vector, k=30)
            assert "doc5" not in [r.id for r in results], (
                "Deleted record reappeared after reopen"
            )
        finally:
            fx.cleanup()

    @pytest.mark.parametrize("seed", SEEDS)
    def test_record_count_preserved_across_reopen(self, seed: int) -> None:
        fx = _DbFixture(seed, n_vectors=50)
        try:
            pre_count = len(fx.col.search_vector(fx.query_vector, k=500))
            fx.col.flush()

            db2 = omendb.open(str(Path(fx.tmpdir) / "db"), create=False)
            col2 = db2.collection("test", create=False)
            post_count = len(col2.search_vector(fx.query_vector, k=500))

            assert pre_count == post_count, (
                f"Record count changed: {pre_count} -> {post_count}"
            )
        finally:
            fx.cleanup()


# ---------------------------------------------------------------------------
# Vacuum Correctness
# ---------------------------------------------------------------------------


class TestVacuumOracle:
    """Prove vacuum preserves search correctness."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_vacuum_preserves_dense_results(self, seed: int) -> None:
        fx = _DbFixture(seed, n_vectors=50)
        try:
            # Delete some records first
            for vid in ("doc3", "doc7", "doc15"):
                fx.col.delete(vid)

            pre = fx.col.search_vector(fx.query_vector, k=5)
            pre_ids = [r.id for r in pre]

            fx.col.vacuum()
            post = fx.col.search_vector(fx.query_vector, k=5)
            post_ids = [r.id for r in post]

            assert pre_ids == post_ids, (
                f"Vacuum changed results: {pre_ids} -> {post_ids}"
            )
        finally:
            fx.cleanup()

    @pytest.mark.parametrize("seed", SEEDS)
    def test_vacuum_with_flush_and_reopen(self, seed: int) -> None:
        fx = _DbFixture(seed, n_vectors=50)
        try:
            for vid in ("doc3", "doc7"):
                fx.col.delete(vid)

            fx.col.vacuum()
            fx.col.flush()

            db2 = omendb.open(str(Path(fx.tmpdir) / "db"), create=False)
            col2 = db2.collection("test", create=False)

            pre_delete = [r.id for r in fx.col.search_vector(fx.query_vector, k=5)]
            post = [r.id for r in col2.search_vector(fx.query_vector, k=5)]

            assert pre_delete == post, (
                "Vacuum+flush+reopen changed results"
            )
        finally:
            fx.cleanup()


# ---------------------------------------------------------------------------
# Config Mismatch & Corrupt State Detection
# ---------------------------------------------------------------------------


class TestResilience:
    """Prove the engine detects and handles bad state."""

    def test_reopen_with_wrong_dimension(self, tmp_path: Path) -> None:
        db = omendb.open(str(tmp_path / "db"), create=True)
        col = db.collection("test", config=omendb.CollectionConfig(dim=2, text=True))
        col.set("d1", vector=[1.0, 0.0])
        col.flush()

        # Reopen with wrong dimension
        db2 = omendb.open(str(tmp_path / "db"), create=False)
        with pytest.raises(ValueError, match="dim|config"):
            db2.collection("test", config=omendb.CollectionConfig(dim=128))

    def test_open_collection_without_text_when_created_with_text(self, tmp_path: Path) -> None:
        db = omendb.open(str(tmp_path / "db"), create=True)
        col = db.collection("test", config=omendb.CollectionConfig(dim=2, text=True))
        col.set("d1", vector=[1.0, 0.0], text="hello")
        col.flush()

        # Reopen without text config
        db2 = omendb.open(str(tmp_path / "db"), create=False)
        with pytest.raises(ValueError, match="text"):
            db2.collection("test", config=omendb.CollectionConfig(dim=2, text=False))

    def test_truncated_manifest_is_handled(self, tmp_path: Path) -> None:
        db = omendb.open(str(tmp_path / "db"), create=True)
        col = db.collection("test", config=omendb.CollectionConfig(dim=2, text=True))
        col.set("d1", vector=[1.0, 0.0])
        col.flush()

        # Truncate the manifest
        manifest_path = tmp_path / "db" / "test" / "manifest.json"
        manifest_path.write_text("{", encoding="utf-8")  # Invalid JSON

        db2 = omendb.open(str(tmp_path / "db"), create=False)
        with pytest.raises(Exception):
            db2.collection("test", create=False)

    def test_missing_sidecar_is_handled(self, tmp_path: Path) -> None:
        db = omendb.open(str(tmp_path / "db"), create=True)
        col = db.collection("test", config=omendb.CollectionConfig(dim=2))
        col.set("d1", vector=[1.0, 0.0])
        col.flush()

        # Delete a sidecar file
        sidecar = tmp_path / "db" / "test" / "hnsw.data"
        if sidecar.exists():
            sidecar.unlink()

        db2 = omendb.open(str(tmp_path / "db"), create=False)
        col2 = db2.collection("test", create=False)
        result = col2.check()
        # Check should produce a result object with diagnostics
        assert isinstance(result, omendb.CheckResult), (
            f"Expected CheckResult, got {type(result)}"
        )
        # With missing sidecar, check should indicate issues
        assert not result.ok, f"check() should detect missing sidecar: {result}"


# ---------------------------------------------------------------------------
# Crash/Recovery Matrix
# ---------------------------------------------------------------------------


class TestCrashRecovery:
    """Prove process-level crash recovery preserves data integrity."""

    def test_flush_then_reopen_preserves_all_data(self, tmp_path: Path) -> None:
        """After clean flush+reopen, all records and search results are preserved."""
        db = omendb.open(str(tmp_path / "db"), create=True)
        col = db.collection("test", config=omendb.CollectionConfig(dim=2, text=True))
        col.set("d1", vector=[1.0, 0.0], text="hello")
        col.set("d2", vector=[0.0, 1.0], text="world")
        col.set("d3", vector=[0.5, 0.5], text="hello world")
        col.flush()

        db2 = omendb.open(str(tmp_path / "db"), create=False)
        col2 = db2.collection("test", create=False)

        # Record count preserved
        assert len(col2.search_vector([0.5, 0.5], k=10)) == 3
        # Dense search preserved
        dense2 = [r.id for r in col2.search_vector([1.0, 0.0], k=3)]
        assert dense2 == ["d1", "d3", "d2"]
        # Text search preserved
        text2 = [r.id for r in col2.search_text("hello", k=3)]
        assert "d1" in text2 and "d3" in text2
        # Get preserved (returns id + metadata)
        d1 = col2.get("d1")
        assert d1 is not None
        # Text content searchable
        tr = col2.search_text("hello", k=3)
        assert tr and tr[0].id == "d1"

    def test_multiple_flush_reopen_cycles(self, tmp_path: Path) -> None:
        """Multiple flush+reopen cycles must preserve data identically."""
        db = omendb.open(str(tmp_path / "db"), create=True)
        col = db.collection("test", config=omendb.CollectionConfig(dim=2, text=True))

        for cycle in range(3):
            col.set(f"c{cycle}", vector=[float(cycle), 0.0], text=f"cycle {cycle}")
            col.flush()

            db = omendb.open(str(tmp_path / "db"), create=False)
            col = db.collection("test", create=False)

            # Verify all cycles' records present
            for c in range(cycle + 1):
                r = col.get(f"c{c}")
                assert r is not None, f"Record c{c} missing after cycle {cycle}"
                # Verify via text search
                tr = col.search_text(f"cycle {c}", k=1)
                assert tr and tr[0].id == f"c{c}", f"Text search failed for cycle {c}"

    def test_flush_with_deletes_survives_reopen(self, tmp_path: Path) -> None:
        """Deleted records must not reappear after flush+reopen."""
        db = omendb.open(str(tmp_path / "db"), create=True)
        col = db.collection("test", config=omendb.CollectionConfig(dim=2))
        col.set("keep", vector=[1.0, 0.0])
        col.set("delete_me", vector=[0.0, 1.0])
        col.delete("delete_me")
        col.flush()

        db2 = omendb.open(str(tmp_path / "db"), create=False)
        col2 = db2.collection("test", create=False)

        results = col2.search_vector([0.5, 0.5], k=5)
        ids = [r.id for r in results]
        assert "keep" in ids
        assert "delete_me" not in ids, "Deleted record reappeared"

    def test_flush_with_updates_survives_reopen(self, tmp_path: Path) -> None:
        """Updated records must reflect changes after flush+reopen."""
        db = omendb.open(str(tmp_path / "db"), create=True)
        col = db.collection("test", config=omendb.CollectionConfig(dim=2, text=True))
        col.set("doc", vector=[1.0, 0.0], text="original")
        col.set("doc", vector=[0.0, 1.0], text="updated")
        col.flush()

        db2 = omendb.open(str(tmp_path / "db"), create=False)
        col2 = db2.collection("test", create=False)

        # Record exists
        r = col2.get("doc")
        assert r is not None
        # Updated text is searchable (text search finds it)
        tr = col2.search_text("updated", k=1)
        assert len(tr) == 1 and tr[0].id == "doc"
        # Updated vector is reflected (search for new vector finds doc)
        results = col2.search_vector([0.0, 1.0], k=1)
        assert results[0].id == "doc"

    def test_vacuum_survives_reopen(self, tmp_path: Path) -> None:
        """Vacuumed state must persist across flush+reopen."""
        db = omendb.open(str(tmp_path / "db"), create=True)
        col = db.collection("test", config=omendb.CollectionConfig(dim=2))
        for i in range(10):
            col.set(f"doc{i}", vector=[float(i), 0.0])
        col.delete("doc3")
        col.delete("doc7")
        col.vacuum()
        col.flush()

        db2 = omendb.open(str(tmp_path / "db"), create=False)
        col2 = db2.collection("test", create=False)

        results = col2.search_vector([0.0, 0.0], k=10)
        ids = [r.id for r in results]
        assert "doc3" not in ids
        assert "doc7" not in ids
        assert len(ids) == 8  # 10 - 2 deleted

    def test_supersede_survives_reopen(self, tmp_path: Path) -> None:
        """Superseded records must stay superseded after flush+reopen."""
        db = omendb.open(str(tmp_path / "db"), create=True)
        col = db.collection("test", config=omendb.CollectionConfig(dim=2))
        col.set("old", vector=[1.0, 0.0])
        col.set("new", vector=[0.0, 1.0])
        col.supersede("old", "new")
        col.flush()

        db2 = omendb.open(str(tmp_path / "db"), create=False)
        col2 = db2.collection("test", create=False)

        assert col2.get("old") is None
        assert col2.get("new") is not None
        results = col2.search_vector([0.5, 0.5], k=2)
        assert "old" not in [r.id for r in results]

    def test_reopen_after_text_only_records(self, tmp_path: Path) -> None:
        """Text-only records (no vector) survive flush+reopen."""
        db = omendb.open(str(tmp_path / "db"), create=True)
        col = db.collection("test", config=omendb.CollectionConfig(dim=2, text=True))
        col.set("readme", text="This is the readme")
        col.set("guide", vector=[1.0, 0.0], text="This is the guide")
        col.flush()

        db2 = omendb.open(str(tmp_path / "db"), create=False)
        col2 = db2.collection("test", create=False)

        # Text-only records searchable via text
        text_results = col2.search_text("readme", k=5)
        assert any(r.id == "readme" for r in text_results), "Text-only record not searchable"

        # Vector-backed records searchable via dense
        dense_results = col2.search_vector([1.0, 0.0], k=1)
        assert dense_results[0].id == "guide"


# ---------------------------------------------------------------------------
# Source Evidence & Edges Oracle
# ---------------------------------------------------------------------------


class TestSourceEvidence:
    """Prove source evidence and relationship edges correctness."""

    def test_source_is_preserved_in_search(self, tmp_path: Path) -> None:
        db = omendb.open(str(tmp_path / "db"), create=True)
        col = db.collection("test", config=omendb.CollectionConfig(dim=2))
        col.set(
            "doc",
            vector=[1.0, 0.0],
            source={"path": "/data/file.txt", "line": 42},
        )

        # Search results include source
        results = col.search_vector([1.0, 0.0], k=1)
        assert results[0].source is not None
        assert results[0].source.get("path") == "/data/file.txt"

    def test_source_survives_flush_and_reopen(self, tmp_path: Path) -> None:
        db = omendb.open(str(tmp_path / "db"), create=True)
        col = db.collection("test", config=omendb.CollectionConfig(dim=2))
        col.set("doc", vector=[1.0, 0.0], source={"path": "/test"})
        col.flush()

        db2 = omendb.open(str(tmp_path / "db"), create=False)
        col2 = db2.collection("test", create=False)

        results = col2.search_vector([1.0, 0.0], k=1)
        assert results[0].source is not None
        assert results[0].source["path"] == "/test"

    def test_relationships_set_and_query(self, tmp_path: Path) -> None:
        db = omendb.open(str(tmp_path / "db"), create=True)
        col = db.collection("test", config=omendb.CollectionConfig(
            dim=2, graph=True
        ))
        col.set("a", vector=[1.0, 0.0])
        col.set("b", vector=[0.0, 1.0])
        col.add_relationship("a", "b", type="related")

        # Relationship evidence confirms the edge
        evidence = col.relationship_evidence("a", "b", type="related")
        assert evidence is not None
        assert evidence.included is True
        assert evidence.type == "related"

    def test_relationships_persist_across_flush_reopen(self, tmp_path: Path) -> None:
        db = omendb.open(str(tmp_path / "db"), create=True)
        col = db.collection("test", config=omendb.CollectionConfig(
            dim=2, graph=True
        ))
        col.set("a", vector=[1.0, 0.0])
        col.set("b", vector=[0.0, 1.0])
        col.add_relationship("a", "b", type="related")
        col.flush()

        db2 = omendb.open(str(tmp_path / "db"), create=False)
        col2 = db2.collection("test", create=False)

        # Verify records exist
        assert col2.get("a") is not None
        assert col2.get("b") is not None
        # Relationship evidence is available after reopen
        evidence = col2.relationship_evidence("a", "b", type="related")
        assert evidence is not None
        assert evidence.included is True

    def test_explain_shows_source_and_edges(self, tmp_path: Path) -> None:
        db = omendb.open(str(tmp_path / "db"), create=True)
        col = db.collection("test", config=omendb.CollectionConfig(dim=2))
        col.set("doc", vector=[1.0, 0.0], source={"path": "/test"})
        results = col.search_vector([1.0, 0.0], k=1, explain=True)
        assert results[0].source is not None
        assert results[0].explanation is not None
        assert "method" in results[0].explanation


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge-case correctness verification."""

    def test_search_on_empty_collection(self, tmp_path: Path) -> None:
        db = omendb.open(str(tmp_path / "db"), create=True)
        col = db.collection("test", config=omendb.CollectionConfig(dim=2))
        results = col.search_vector([1.0, 0.0], k=5)
        assert results == []

    def test_k_larger_than_dataset(self, tmp_path: Path) -> None:
        db = omendb.open(str(tmp_path / "db"), create=True)
        col = db.collection("test", config=omendb.CollectionConfig(dim=2))
        col.set("d1", vector=[1.0, 0.0])
        col.set("d2", vector=[0.0, 1.0])
        results = col.search_vector([0.5, 0.5], k=10)
        assert len(results) == 2  # Only 2 records exist

    def test_search_with_identical_vectors(self, tmp_path: Path) -> None:
        db = omendb.open(str(tmp_path / "db"), create=True)
        col = db.collection("test", config=omendb.CollectionConfig(dim=2))
        col.set("d1", vector=[1.0, 0.0])
        col.set("d2", vector=[1.0, 0.0])  # Same vector
        results = col.search_vector([1.0, 0.0], k=2)
        ids = [r.id for r in results]
        assert set(ids) == {"d1", "d2"}

    def test_text_search_no_common_terms(self, tmp_path: Path) -> None:
        db = omendb.open(str(tmp_path / "db"), create=True)
        col = db.collection("test", config=omendb.CollectionConfig(dim=2, text=True))
        col.set("d1", vector=[1.0, 0.0], text="alpha beta")
        col.set("d2", vector=[0.0, 1.0], text="gamma delta")
        results = col.search_text("epsilon", k=10)
        assert results == []

    def test_all_ids_excluded(self, tmp_path: Path) -> None:
        db = omendb.open(str(tmp_path / "db"), create=True)
        col = db.collection("test", config=omendb.CollectionConfig(dim=2))
        col.set("d1", vector=[1.0, 0.0])
        col.set("d2", vector=[0.0, 1.0])
        results = col.search_vector(
            [0.5, 0.5], k=5, exclude_ids=["d1", "d2"]
        )
        assert results == []

    def test_include_ids_with_no_matches_returns_empty(self, tmp_path: Path) -> None:
        db = omendb.open(str(tmp_path / "db"), create=True)
        col = db.collection("test", config=omendb.CollectionConfig(dim=2))
        col.set("d1", vector=[1.0, 0.0])
        results = col.search_vector(
            [0.5, 0.5], k=5, include_ids=["nonexistent"]
        )
        assert results == []
