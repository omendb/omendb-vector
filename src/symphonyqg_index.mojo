"""
SymphonyQG backend — co-located RaBitQ graph index.

Phase 1: standalone backend. Not wired into VectorIndex yet.
Flat-layout storage: parallel lists for vectors/codes/neighbors.
R=32 degree bound (FastScan SIMD constraint).

Reference: SymphonyQG, SIGMOD 2025 — arXiv:2411.12229
"""

from rabitq import (
    RaBitQEncoder,
    RaBitQCode,
    _popcount_u64,
)
from store_types import Metric
from hnsw import Candidate
from std.memory import Span
from std.math import sqrt


struct SymphonyQGIndex[dim: Int](Movable):
    """RaBitQ-quantized graph ANN index, single-layer Vamana graph.

    Flat-layout storage (parallel lists to avoid Mojo List Copyable constraint):
      - raw_f32: flat Float32 array, n * dim elements
      - codes: flat UInt64 array, n * R * words_per_vec
      - neighbor_ids: flat UInt32 array, n * R
      - neighbor_counts: Int per node (how many of R slots are used)
    """

    var raw_f32: List[Float32]
    var codes: List[UInt64]
    var neighbor_ids: List[UInt32]
    var neighbor_counts: List[Int]
    var encoder: RaBitQEncoder[Self.dim]
    var entry_point: Int
    var _num_elements: Int
    var _num_deleted: Int
    var _deleted: List[Bool]
    var metric: Metric
    var _fitted: Bool

    comptime R: Int = 32
    comptime words_per_vec: Int = (Self.dim + 63) // 64

    def __init__(out self, metric: Metric = Metric.L2):
        self.raw_f32 = List[Float32]()
        self.codes = List[UInt64]()
        self.neighbor_ids = List[UInt32]()
        self.neighbor_counts = List[Int]()
        self.encoder = RaBitQEncoder[Self.dim]()
        self.entry_point = 0
        self._num_elements = 0
        self._num_deleted = 0
        self._deleted = List[Bool]()
        self.metric = metric
        self._fitted = False

    def __init__(out self, *, deinit take: Self):
        self.raw_f32 = take.raw_f32^
        self.codes = take.codes^
        self.neighbor_ids = take.neighbor_ids^
        self.neighbor_counts = take.neighbor_counts^
        self.encoder = take.encoder.copy()
        self.entry_point = take.entry_point
        self._num_elements = take._num_elements
        self._num_deleted = take._num_deleted
        self._deleted = take._deleted^
        self._fitted = take._fitted
        self.metric = take.metric

    # ── VectorIndex contract ──────────────────────────────────────

    def num_elements(ref self) -> Int:
        return self._num_elements

    def insert(mut self, vector: Span[Float32, _], code: RaBitQCode) raises:
        """Insert a vector with its precomputed RaBitQ code.

        Vamana-style: greedy search, RobustPrune to R neighbors,
        co-located codes + ids stored in flat arrays.
        """
        var node_id = self._num_elements

        # Store raw vector
        self.raw_f32.reserve((node_id + 1) * Self.dim)
        for d in range(Self.dim):
            self.raw_f32.append(vector[d])

        # Store RaBitQ codes (tracked by flat array index, not List.len call)
        self.codes.reserve((node_id + 1) * Self.R * Self.words_per_vec)
        for i in range(len(code.bits)):
            self.codes.append(code.bits[i])
        # Pad remaining code slots with zeros
        while len(self.codes) < (node_id + 1) * Self.R * Self.words_per_vec:
            self.codes.append(0)

        # Store neighbor IDs
        self.neighbor_ids.reserve((node_id + 1) * Self.R)
        for _ in range(Self.R):
            self.neighbor_ids.append(0)
        self.neighbor_counts.append(0)
        self._deleted.append(False)

        if self._num_elements == 0:
            self.entry_point = 0
        else:
            # Greedy search for candidate neighbors
            var candidates = self._greedy_search(vector, 100)

            # Sort candidates by distance (closest first)
            var cspan = Span[Candidate, MutAnyOrigin](
                ptr=candidates.unsafe_ptr(), length=len(candidates)
            )
            _sort_candidates(cspan)

            # Add forward edges from new node to R closest candidates
            var n_nbrs = min(Self.R, len(candidates))
            var nbr_start = node_id * Self.R
            for i in range(n_nbrs):
                var nbr = Int(candidates[i].id)
                self.neighbor_ids[nbr_start + i] = UInt32(nbr)
            self.neighbor_counts[node_id] = n_nbrs

            # Add reverse edges (from candidate to new node) with pruning
            for i in range(n_nbrs):
                var nbr = Int(candidates[i].id)
                var r_start = nbr * Self.R
                var r_count = self.neighbor_counts[nbr]
                # Add new node as a neighbor of the candidate
                if r_count < Self.R:
                    self.neighbor_ids[r_start + r_count] = UInt32(node_id)
                    self.neighbor_counts[nbr] = r_count + 1
                else:
                    # Prune: replace farthest neighbor if new node is closer
                    # Recompute distances from nbr to all its neighbors
                    var nbr_vec = Span[Float32, _](
                        ptr=self.raw_f32.unsafe_ptr() + nbr * Self.dim,
                        length=Self.dim,
                    )
                    var worst_idx = -1
                    var worst_dist: Float32 = -1.0
                    for j in range(Self.R):
                        var other = Int(self.neighbor_ids[r_start + j])
                        var other_ptr = self.raw_f32.unsafe_ptr() + other * Self.dim
                        var d: Float32 = 0.0
                        for k in range(Self.dim):
                            var diff = nbr_vec[k] - other_ptr[k]
                            d += diff * diff
                        if d > worst_dist:
                            worst_dist = d
                            worst_idx = j
                    # Distance from nbr to new node
                    var new_dist: Float32 = 0.0
                    for k in range(Self.dim):
                        var diff = nbr_vec[k] - vector[k]
                        new_dist += diff * diff
                    if new_dist < worst_dist and worst_idx >= 0:
                        self.neighbor_ids[r_start + worst_idx] = UInt32(node_id)

        self._num_elements += 1

    def search(
        ref self, query: Span[Float32, _], k: Int, ef: Int
    ) raises -> List[Candidate]:
        """Vamana beam search with RaBitQ distance estimates.

        Each hop: FastScan all R neighbor codes in one sequential pass.
        No exact rerank — RaBitQ error bounds make beam termination safe.
        """
        if self._num_elements == 0:
            return List[Candidate]()

        var qcode = self.encoder.encode(query)

        var visited = List[Bool]()
        for _ in range(self._num_elements):
            visited.append(False)

        var beam = List[Candidate]()
        var ep = self.entry_point

        # Entry point: exact F32 distance
        var ep_vec = Span[Float32, _](
            ptr=self.raw_f32.unsafe_ptr() + ep * Self.dim, length=Self.dim
        )
        var ep_dist = _metric_distance[Self.dim](self.metric, query, ep_vec)
        beam.append(Candidate(UInt32(ep), ep_dist))
        visited[ep] = True

        var results = List[Candidate]()

        while len(results) < ef:
            # Find best unexpanded node in beam
            var best_idx = -1
            var best_dist: Float32 = 1e30
            for i in range(len(beam)):
                if beam[i].distance < best_dist:
                    best_dist = beam[i].distance
                    best_idx = i
            if best_idx < 0:
                break

            var node = Int(beam[best_idx].id)
            beam[best_idx] = beam[len(beam) - 1]
            _ = beam.pop()
            results.append(Candidate(UInt32(node), best_dist))

            # F32 exact distances for all R neighbors
            var nbr_start = node * Self.R
            var nbr_count = self.neighbor_counts[node]

            for ni in range(nbr_count):
                var nbr = Int(self.neighbor_ids[nbr_start + ni])
                if nbr >= self._num_elements or self._deleted[nbr] or visited[nbr]:
                    continue
                visited[nbr] = True

                var nbr_vec = Span[Float32, _](
                    ptr=self.raw_f32.unsafe_ptr() + nbr * Self.dim, length=Self.dim
                )
                var exact_dist = _metric_distance[Self.dim](self.metric, query, nbr_vec)
                beam.append(Candidate(UInt32(nbr), exact_dist))

        # Sort results by F32 distance
        var span = Span[Candidate, MutAnyOrigin](
            ptr=results.unsafe_ptr(), length=len(results)
        )
        _sort_candidates(span)

        var top_k = List[Candidate]()
        top_k.reserve(min(k, len(results)))
        for i in range(min(k, len(results))):
            top_k.append(results[i])
        return top_k^

    def get_vector(ref self, id: Int) -> Span[Float32, origin_of(self.raw_f32)]:
        return Span(
            ptr=self.raw_f32.unsafe_ptr() + id * Self.dim, length=Self.dim
        )

    def delete(mut self, id: Int):
        if id < len(self._deleted) and not self._deleted[id]:
            self._deleted[id] = True
            self._num_deleted += 1

    # ── internal helpers ──────────────────────────────────────────

    def _greedy_search(
        ref self, query: Span[Float32, _], ef: Int
    ) raises -> List[Candidate]:
        """Exact F32 brute-force for construction."""
        var all = List[Candidate]()
        var raw_ptr = self.raw_f32.unsafe_ptr()
        for i in range(self._num_elements):
            if self._deleted[i]:
                continue
            var a_ptr = raw_ptr + i * Self.dim
            var d: Float32 = 0.0
            if self.metric == Metric.L2:
                for k in range(Self.dim):
                    var diff = a_ptr[k] - query[k]
                    d += diff * diff
            elif self.metric == Metric.DOT:
                for k in range(Self.dim):
                    d -= a_ptr[k] * query[k]
            else:
                var dot: Float32 = 0.0
                var na: Float32 = 0.0
                var nb: Float32 = 0.0
                for k in range(Self.dim):
                    dot += a_ptr[k] * query[k]
                    na += a_ptr[k] * a_ptr[k]
                    nb += query[k] * query[k]
                d = 1.0 - dot / (sqrt(na) * sqrt(nb)) if na > 0 else 1.0
            all.append(Candidate(UInt32(i), d))

        var s = Span[Candidate, MutAnyOrigin](
            ptr=all.unsafe_ptr(), length=len(all)
        )
        _sort_candidates(s)

        var top = List[Candidate]()
        for i in range(min(ef, len(all))):
            top.append(all[i])
        return top^

    def _prune_neighbors(mut self, node: Int) raises:
        """RobustPrune: keep closest R neighbors by F32 distance."""
        var start = node * Self.R
        var nbrs = List[Candidate]()
        var raw_ptr = self.raw_f32.unsafe_ptr()
        for ni in range(self.neighbor_counts[node]):
            var nid = Int(self.neighbor_ids[start + ni])
            # Compute distance directly using pointers (avoid Span aliasing)
            var a_ptr = raw_ptr + node * Self.dim
            var b_ptr = raw_ptr + nid * Self.dim
            var d: Float32 = 0.0
            if self.metric == Metric.L2:
                for k in range(Self.dim):
                    var diff = a_ptr[k] - b_ptr[k]
                    d += diff * diff
            elif self.metric == Metric.DOT:
                for k in range(Self.dim):
                    d -= a_ptr[k] * b_ptr[k]
            else:
                var dot: Float32 = 0.0
                var na: Float32 = 0.0
                var nb: Float32 = 0.0
                for k in range(Self.dim):
                    dot += a_ptr[k] * b_ptr[k]
                    na += a_ptr[k] * a_ptr[k]
                    nb += b_ptr[k] * b_ptr[k]
                d = 1.0 - dot / (sqrt(na) * sqrt(nb)) if na > 0 and nb > 0 else 1.0
            nbrs.append(Candidate(self.neighbor_ids[start + ni], d))

        var s = Span[Candidate, MutAnyOrigin](
            ptr=nbrs.unsafe_ptr(), length=len(nbrs)
        )
        _sort_candidates(s)

        var keep = min(Self.R, len(nbrs))
        for i in range(keep):
            self.neighbor_ids[start + i] = nbrs[i].id
        self.neighbor_counts[node] = keep


# ── helpers ───────────────────────────────────────────────────────

@parameter
def _sort_candidates(span: Span[Candidate, MutAnyOrigin]):
    """Sort candidates by ascending distance (simple selection sort)."""
    for i in range(len(span)):
        var best = i
        for j in range(i + 1, len(span)):
            if span[j].distance < span[best].distance:
                best = j
        if best != i:
            var tmp = span[i]
            span[i] = span[best]
            span[best] = tmp


def _metric_distance[dim: Int](
    metric: Metric, a: Span[Float32, _], b: Span[Float32, _]
) -> Float32:
    """Compute distance according to metric."""
    if metric == Metric.L2:
        var s: Float32 = 0.0
        for d in range(dim):
            var diff = a[d] - b[d]
            s += diff * diff
        return s
    elif metric == Metric.DOT:
        var s: Float32 = 0.0
        for d in range(dim):
            s += a[d] * b[d]
        return -s
    else:  # COSINE
        var dot: Float32 = 0.0
        var na: Float32 = 0.0
        var nb: Float32 = 0.0
        for d in range(dim):
            dot += a[d] * b[d]
            na += a[d] * a[d]
            nb += b[d] * b[d]
        if na == 0.0 or nb == 0.0:
            return 1.0
        return 1.0 - dot / (sqrt(na) * sqrt(nb))
