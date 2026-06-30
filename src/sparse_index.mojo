"""
Sparse vector support: SparseVector + SparseInvertedIndex.

A sibling index family (not a VectorIndex backend — sparse has no dense vector).
Three-way RRF fusion composes this alongside VectorIndex (dense) and BM25Index
(text) in the planner.

Per sparse-vectors.md: plain dot-product accumulation is correct for v0.
WAND/MaxScore skip optimization is deferred.
"""

from std.memory import Span


# ── SparseVector ──────────────────────────────────────────────────


struct SparseVector(Copyable, Movable):
    """Sorted (dimension, weight) pairs. Invariant: dims is strictly ascending.
    """

    var dims: List[UInt32]
    var weights: List[Float32]

    def __init__(out self):
        self.dims = List[UInt32]()
        self.weights = List[Float32]()

    def __init__(out self, *, deinit take: Self):
        self.dims = take.dims^
        self.weights = take.weights^

    def __init__(out self, *, copy: Self):
        # Rebuild dims and weights by copying elements (List is not ImplicitlyCopyable)
        var new_dims = List[UInt32]()
        var new_weights = List[Float32]()
        for i in range(len(copy.dims)):
            new_dims.append(copy.dims[i])
            new_weights.append(copy.weights[i])
        self.dims = new_dims^
        self.weights = new_weights^

    def nnz(ref self) -> Int:
        """Number of non-zero dimensions."""
        return len(self.dims)


# ── SparsePosting ─────────────────────────────────────────────────


struct SparsePosting(ImplicitlyCopyable):
    """A single (doc_id, weight) entry in a posting list."""

    var doc_id: UInt32
    var weight: Float32

    def __init__(out self, doc_id: UInt32, weight: Float32):
        self.doc_id = doc_id
        self.weight = weight


struct SparseSearchResult(ImplicitlyCopyable):
    var doc_id: UInt32
    var score: Float32

    def __init__(out self, doc_id: UInt32, score: Float32):
        self.doc_id = doc_id
        self.score = score


# ── SparseInvertedIndex ───────────────────────────────────────────


struct SparseInvertedIndex(Movable):
    """Inverted index over sparse vector dimensions.

    Mirror of BM25Index: posting_lists holds the per-dimension posting lists,
    and dim_to_idx maps dimension keys to indices into posting_lists.
    This avoids putting List directly in a Dict (Mojo 1.0 constraint).
    """

    var posting_lists: List[List[SparsePosting]]
    var dim_to_idx: Dict[UInt32, Int]
    var num_docs: Int
    var max_dim: UInt32

    def __init__(out self):
        self.posting_lists = List[List[SparsePosting]]()
        self.dim_to_idx = Dict[UInt32, Int]()
        self.num_docs = 0
        self.max_dim = 0

    def __init__(out self, *, deinit take: Self):
        self.posting_lists = take.posting_lists^
        self.dim_to_idx = take.dim_to_idx^
        self.num_docs = take.num_docs
        self.max_dim = take.max_dim

    def insert(mut self, ref vector: SparseVector, doc_id: UInt32) raises:
        """Insert a sparse vector for the given document ID.

        doc_id comes from the store's vector_id (not a sequential counter),
        so sparse results map directly to store vector_ids.
        Each nonzero dimension's posting list gets a (doc_id, weight) entry.
        """
        for i in range(len(vector.dims)):
            var dim = vector.dims[i]
            var weight = vector.weights[i]
            if weight == 0.0:
                continue
            if dim not in self.dim_to_idx:
                self.dim_to_idx[dim] = len(self.posting_lists)
                self.posting_lists.append(List[SparsePosting]())
                if dim > self.max_dim:
                    self.max_dim = dim
            var idx = self.dim_to_idx[dim]
            self.posting_lists[idx].append(SparsePosting(doc_id, weight))
        if doc_id >= UInt32(self.num_docs):
            self.num_docs = Int(doc_id) + 1

    def search(
        ref self, query: SparseVector, k: Int
    ) raises -> List[SparseSearchResult]:
        """Dot-product retrieval: accumulate scores and return top-k.

        Returns List of SparseSearchResult sorted by descending score.
        Empty index or query returns empty list.
        """
        if self.num_docs == 0 or k <= 0 or query.nnz() == 0:
            var empty = List[SparseSearchResult]()
            return empty^

        var scores = Dict[UInt32, Float32]()

        for qi in range(len(query.dims)):
            var dim = query.dims[qi]
            var q_weight = query.weights[qi]
            if q_weight == 0.0:
                continue
            if dim not in self.dim_to_idx:
                continue
            var idx = self.dim_to_idx[dim]
            ref postings = self.posting_lists[idx]
            for pi in range(len(postings)):
                var posting = postings[pi]
                var doc = posting.doc_id
                if doc in scores:
                    scores[doc] += q_weight * posting.weight
                else:
                    scores[doc] = q_weight * posting.weight

        # Extract top-k by score (simple selection, k is small)
        var result = List[SparseSearchResult]()
        var score_keys = List[UInt32]()
        for key in scores:
            score_keys.append(key)
        if len(score_keys) == 0:
            return result^

        var used = List[Bool]()
        for _ in range(len(score_keys)):
            used.append(False)

        var n = min(k, len(score_keys))
        for _ in range(n):
            var best_idx = -1
            var best_score: Float32 = -1.0
            for i in range(len(score_keys)):
                if not used[i]:
                    var s = scores[score_keys[i]]
                    if s > best_score:
                        best_score = s
                        best_idx = i
            if best_idx >= 0:
                var doc = score_keys[best_idx]
                result.append(SparseSearchResult(doc, best_score))
                used[best_idx] = True
        return result^

    # ── Helpers ───────────────────────────────────────────────────────


def sparse_dot_product(a: SparseVector, b: SparseVector) -> Float32:
    """Dot product of two sparse vectors. O(nnz(a) + nnz(b)) merge-intersection.
    """
    var result: Float32 = 0.0
    var ai = 0
    var bi = 0
    while ai < len(a.dims) and bi < len(b.dims):
        if a.dims[ai] < b.dims[bi]:
            ai += 1
        elif a.dims[ai] > b.dims[bi]:
            bi += 1
        else:
            result += a.weights[ai] * b.weights[bi]
            ai += 1
            bi += 1
    return result
