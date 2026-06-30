"""
VectorIndex — unified search contract for vector indexes.

Phase 0: wraps HNSWIndex directly. The contract surface is the stable API;
the wrapped backend is an implementation detail. When Phase 1 adds SymphonyQG,
this becomes a discriminated union or vtable dispatch. Until then, the store
never touches HNSW internals through this type.

Leaks closed (per vector-index-contract.md):
  1. Scratch objects — internal to backend (search_filtered takes no scratch arg)
  2. Raw vector buffer — store goes through get_vector(id), never index.data
  3. HNSW config — store calls config_snapshot(), never .M / .ef_construction directly
"""

from hnsw import Candidate, HNSWIndex
from persistence import save_hnsw, load_hnsw as _load_hnsw
from store_types import Metric
from symphonyqg_index import SymphonyQGIndex
from rabitq import RaBitQCode
from std.memory import Span
from std.python import Python, PythonObject


struct VectorIndex[dim: Int](Movable):
    """Type-erased vector index. In Phase 0 the only backend is HNSW.

    All methods match the VectorIndex contract (vector-index-contract.md).
    The store should never access _backend directly.
    """

    var _backend: HNSWIndex[Self.dim]
    var _qg_backend: SymphonyQGIndex[Self.dim]
    var _using_qg: Bool
    var _qg_first: Bool

    # ── constructors ──────────────────────────────────────────────

    def __init__(
        out self,
        M: Int = 16,
        ef_construction: Int = 100,
        alpha: Float32 = 1.0,
        metric: Metric = Metric.L2,
        backend: Int = 0,  # 0=HNSW, 1=SymphonyQG
    ):
        self._backend = HNSWIndex[Self.dim](
            M=M,
            ef_construction=ef_construction,
            alpha=alpha,
            metric=metric,
        )
        self._qg_backend = SymphonyQGIndex[Self.dim](metric)
        self._using_qg = backend == 1
        self._qg_first = True

    def __init__(out self, *, deinit take: Self):
        self._backend = take._backend^
        self._qg_backend = take._qg_backend^
        self._using_qg = take._using_qg
        self._qg_first = take._qg_first

    # ── contract: identity / size ─────────────────────────────────

    def num_elements(ref self) -> Int:
        if self._using_qg:
            return self._qg_backend.num_elements()
        return self._backend.num_elements

    # ── contract: mutation ────────────────────────────────────────

    def insert(mut self, vector: Span[Float32, _]) raises:
        if self._using_qg:
            var code: RaBitQCode
            if self._qg_first:
                code = self._qg_backend.encoder.encode_first(vector)
                self._qg_first = False
            else:
                code = self._qg_backend.encoder.encode(vector)
            self._qg_backend.insert(vector, code)
        else:
            self._backend.insert(vector)

    def insert_batch(mut self, vectors: Span[Float32, _], count: Int):
        self._backend.insert_batch_serial(vectors, count)

    def delete(mut self, id: Int):
        """Phase 0: no-op. Logical deletion + repair are managed at the store
        level (per deletion-policy-engine.md). The index retains the node."""
        pass

    def enable_build_profile(mut self):
        self._backend.enable_build_profile()

    def disable_build_profile(mut self):
        self._backend.disable_build_profile()

    def build_profile_report(ref self) -> String:
        """Returns a formatted build-profile report string (empty if profiling
        was not enabled)."""
        if not self._backend.build_profile_enabled:
            return String("build profiling not enabled")
        var result = String("inserts: ") + String(self._backend.build_profile_inserts) + "\n"
        result += String("total_ns: ") + String(self._backend.build_profile_total_ns) + "\n"
        result += String("avg_per_insert_ns: ")
        if self._backend.build_profile_inserts > 0:
            result += String(
                self._backend.build_profile_total_ns
                // self._backend.build_profile_inserts
            )
        else:
            result += "0"
        result += "\n"
        result += String("storage_ns: ") + String(self._backend.build_profile_storage_ns) + "\n"
        result += String("quantization_ns: ") + String(self._backend.build_profile_quantization_ns) + "\n"
        result += String("query_prep_ns: ") + String(self._backend.build_profile_query_prep_ns) + "\n"
        result += String("connect_ns: ") + String(self._backend.build_profile_connect_ns) + "\n"
        result += String("upper_descent_ns: ") + String(self._backend.build_profile_upper_descent_ns) + "\n"
        result += String("layer_search_ns: ") + String(self._backend.build_profile_layer_search_ns) + "\n"
        result += String("neighbor_select_ns: ") + String(self._backend.build_profile_neighbor_select_ns) + "\n"
        result += String("link_prune_ns: ") + String(self._backend.build_profile_link_prune_ns) + "\n"
        return result

    def reset_build_profile(mut self):
        self._backend.reset_build_profile()

    # ── contract: search ──────────────────────────────────────────

    def search(
        ref self, query: Span[Float32, _], k: Int, ef: Int
    ) raises -> List[Candidate]:
        if self._using_qg:
            return self._qg_backend.search(query, k, ef)
        return self._backend.search(query, k, ef_search=ef)

    def search_filtered(
        ref self,
        query: Span[Float32, _],
        allowed: List[UInt8],
        k: Int,
        ef: Int,
    ) raises -> List[Candidate]:
        """Filtered search with bitmap allowlist. Backend chooses exact vs
        approximate routing by selectivity; the store does not reach into
        backend vectors."""
        return self._backend.search_filtered(query, allowed, k, ef)

    def search_exact_filtered(
        ref self,
        query: Span[Float32, _],
        allowed: List[UInt8],
        k: Int,
    ) raises -> List[Candidate]:
        """Exact brute-force search over eligible items. Backend-guaranteed
        correct top-k (used in the recall oracle)."""
        return self._backend.search_exact_filtered(query, allowed, k)

    # ── contract: vector access ───────────────────────────────────

    def get_vector(
        ref self, id: Int
    ) raises -> Span[Float32, origin_of(self._backend.data)]:
        """Return a Span over the vector at id. For HNSW only — QG backend
        handles exact distances internally and should not need this."""
        return self._backend.get_vector(id)

    # ── contract: persistence ─────────────────────────────────────

    def save(ref self, dir: String) raises:
        """Delegate to backend-specific serialization. Phase 0: save_hnsw."""
        save_hnsw[Self.dim](dir, self._backend)

    @staticmethod
    def load(dir: String) raises -> VectorIndex[Self.dim]:
        """Load from backend-specific serialization. Phase 0: load_hnsw."""
        var vi = VectorIndex[Self.dim]()
        var loaded = _load_hnsw[Self.dim](dir)
        vi._backend = loaded^
        return vi^

    def load_from_data(
        mut self,
        data: List[Float32],
        codes: List[UInt8],
        layer_neighbors: List[List[UInt32]],
        layer_counts: List[List[UInt32]],
        node_levels: List[Int],
        ep: UInt32,
        max_level: Int,
        num_elements: Int,
        use_f32_construction: Bool = True,
    ) raises:
        """Direct data load (used for merge-from-other-backend operations)."""
        self._backend.load_from_data(
            data^,
            codes^,
            layer_neighbors^,
            layer_counts^,
            node_levels^,
            ep,
            max_level,
            num_elements,
            use_f32_construction,
        )

    # ── contract: config snapshot (replaces manifest hardcoded fields) ──

    def config_snapshot(ref self) raises -> Dict[String, PythonObject]:
        """Return backend-specific config for the manifest. Keys are stable;
        consumers map them into their own config structs. HNSW returns:
          M, ef_construction, alpha, metric."""
        var snapshot = Dict[String, PythonObject]()
        snapshot["M"] = PythonObject(self._backend.M)
        snapshot["ef_construction"] = PythonObject(self._backend.ef_construction)
        snapshot["alpha"] = PythonObject(self._backend.alpha)
        if self._backend.metric == Metric.L2:
            snapshot["metric"] = PythonObject("l2")
        elif self._backend.metric == Metric.COSINE:
            snapshot["metric"] = PythonObject("cosine")
        elif self._backend.metric == Metric.DOT:
            snapshot["metric"] = PythonObject("dot")
        return snapshot

    # ── accessors (Phase 0: exposed so store's _validate_loaded_config can verify) ──

    def _M(ref self) -> Int:
        return self._backend.M

    def _ef_construction(ref self) -> Int:
        return self._backend.ef_construction

    def _alpha(ref self) -> Float32:
        return self._backend.alpha

    def _metric(ref self) -> Metric:
        return self._backend.metric

    def _reserve(mut self, capacity: Int):
        self._backend.reserve(capacity)
