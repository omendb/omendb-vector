from std.collections import Dict, List
from math import ceil
from distance import l2_distance
from graph import GraphDirection, PropertyGraph
from vector_index import Candidate, VectorIndex
from persistence import save_hnsw, save_metadata_index, load_metadata_index
from std.python import Python, PythonObject
from text import BM25Index
from sparse_index import (
    SparseVector,
    SparseSearchResult,
    SparseInvertedIndex,
    SparsePosting,
)
from memory_budget import MemoryBudget
from metadata_index import MetadataIndex
from filter import evaluate_filter_to_bitmap
from bitmap import Bitmap
from store_types import (
    Metric,
    HNSWParams,
    SearchOptions,
    VectorStoreOptions,
    SearchResult,
    _bool_flag,
    _config_fingerprint,
    _sha256_file,
    _snapshot_checksums,
    _validate_snapshot_checksums,
)


struct VectorStore[dim: Int](Movable):
    """Embedded vector database with text search, graph, and lifecycle management.
    """

    # --- Core state ---
    var options: VectorStoreOptions
    var index: VectorIndex[Self.dim]
    var external_to_vector: Dict[String, UInt64]
    var vector_external_ids: List[String]
    var metadata: List[Optional[String]]
    var source_spans: List[Optional[String]]
    var deleted: List[Bool]
    var deleted_count: Int
    var text_only: List[Bool]
    var text_only_count: Int
    var persistent_path: Optional[String]
    var text_index: BM25Index
    var text_doc_vector_ids: List[UInt64]
    var text_doc_texts: List[String]
    var graph: PropertyGraph
    var memory_budget: MemoryBudget
    var metadata_index: MetadataIndex
    # --- Sparse vector support ---
    var sparse_index: SparseInvertedIndex
    var sparse_vectors: List[SparseVector]
    # --- Deletion policy ---
    var _needs_compaction: Bool
    var _tombstones_since_repair: Int
    # --- Timestamps ---
    var created_at: List[Float64]  # Unix timestamp
    var updated_at: List[Float64]  # Unix timestamp
    var superseded_at: List[Float64]  # 0.0 = not superseded

    # === Construction & Lifecycle ===

    def __init__(
        out self, options: VectorStoreOptions = VectorStoreOptions()
    ) raises:
        self.options = options
        self.index = VectorIndex[Self.dim](
            M=options.hnsw.M,
            ef_construction=options.hnsw.ef_construction,
            alpha=options.hnsw.alpha,
            metric=options.metric,
            backend=options.backend,
        )
        self.external_to_vector = Dict[String, UInt64]()
        self.vector_external_ids = List[String]()
        self.metadata = List[Optional[String]]()
        self.source_spans = List[Optional[String]]()
        self.deleted = List[Bool]()
        self.deleted_count = 0
        self.text_only = List[Bool]()
        self.text_only_count = 0
        self.persistent_path = None
        self.text_index = BM25Index()
        self.text_doc_vector_ids = List[UInt64]()
        self.text_doc_texts = List[String]()
        self.graph = PropertyGraph()
        self.metadata_index = MetadataIndex()
        self.sparse_index = SparseInvertedIndex()
        self.sparse_vectors = List[SparseVector]()
        self._needs_compaction = False
        self._tombstones_since_repair = 0
        self.memory_budget = MemoryBudget()
        self.created_at = List[Float64]()
        self.updated_at = List[Float64]()
        self.superseded_at = List[Float64]()

    def __init__(out self, *, deinit take: Self):
        self.options = take.options
        self.index = take.index^
        self.external_to_vector = take.external_to_vector^
        self.vector_external_ids = take.vector_external_ids^
        self.metadata = take.metadata^
        self.source_spans = take.source_spans^
        self.deleted = take.deleted^
        self.deleted_count = take.deleted_count
        self.text_only = take.text_only^
        self.text_only_count = take.text_only_count
        self.persistent_path = take.persistent_path^
        self.text_index = take.text_index^
        self.text_doc_vector_ids = take.text_doc_vector_ids^
        self.text_doc_texts = take.text_doc_texts^
        self.graph = take.graph^
        self.memory_budget = take.memory_budget^
        self.metadata_index = take.metadata_index^
        self.sparse_index = take.sparse_index^
        self.sparse_vectors = take.sparse_vectors^
        self._needs_compaction = take._needs_compaction
        self._tombstones_since_repair = take._tombstones_since_repair
        self.created_at = take.created_at^
        self.updated_at = take.updated_at^
        self.superseded_at = take.superseded_at^

    @staticmethod
    def create_in_memory(
        options: VectorStoreOptions = VectorStoreOptions(),
    ) raises -> Self:
        return Self(options)

    @staticmethod
    def create(
        path: String, options: VectorStoreOptions = VectorStoreOptions()
    ) raises -> Self:
        var os = Python.import_module("os")
        if os.path.exists(path) and os.path.exists(path + "/manifest.json"):
            raise Error(
                "store already exists at this path: use open() to load, or"
                " choose a different path"
            )
        _ = os.makedirs(path, exist_ok=True)

        var store = Self(options)
        store.persistent_path = Optional[String](path)
        store.flush()
        return store^

    @staticmethod
    def open(path: String) raises -> Self:
        var json = Python.import_module("json")
        var builtins = Python.import_module("builtins")
        var os = Python.import_module("os")
        if not os.path.exists(path + "/manifest.json"):
            raise Error("store not found: no manifest.json at this path")

        var manifest_file = builtins.open(path + "/manifest.json", "r")
        var manifest = json.loads(manifest_file.read())
        _ = manifest_file.close()
        Self._validate_manifest_layout(path, manifest)
        _validate_snapshot_checksums(path, manifest)

        var manifest_dim = Int(py=manifest["dim"])
        if manifest_dim != Self.dim:
            raise Error(
                "store dimension mismatch: stored dim does not match this"
                " VectorStore type"
            )

        var loaded_index = VectorIndex[Self.dim].load(path + "/hnsw")
        var hnsw_params = HNSWParams(
            M=loaded_index._M(),
            ef_construction=loaded_index._ef_construction(),
            ef_search=100,
            alpha=loaded_index._alpha(),
        )
        Self._validate_loaded_config(manifest, loaded_index)
        var store = Self(
            VectorStoreOptions(hnsw=hnsw_params, metric=loaded_index._metric())
        )
        store.index = loaded_index^
        store.persistent_path = Optional[String](path)
        store._load_records(path)
        store._load_tombstones(path)
        store._load_snapshot_features(path, manifest)

        # Load metadata index (backward compat: rebuild if empty)
        store.metadata_index = load_metadata_index(path + "/hnsw.midx")
        if (
            store.metadata_index.num_docs == 0
            and len(store.vector_external_ids) > 0
        ):
            # Rebuild from stored metadata JSON
            store.metadata_index = MetadataIndex()
            for i in range(len(store.metadata)):
                if store.metadata[i] and not store.deleted[i]:
                    var json_mod = Python.import_module("json")
                    var meta_obj = json_mod.loads(store.metadata[i].value())
                    store.metadata_index.index_json(i, meta_obj)

        if len(store.vector_external_ids) != store.index.num_elements():
            raise Error(
                "store integrity error: record count mismatch between manifest"
                " and data"
            )
        # Record memory budget for loaded data
        store.memory_budget.record_vector_insert(
            Self.dim * store.index.num_elements()
        )
        store.memory_budget.record_code_insert(
            Self.dim * store.index.num_elements()
        )
        return store^

    def is_persistent(ref self) -> Bool:
        return self.persistent_path != None

    def flush(ref self) raises:
        if not self.is_persistent():
            return
        var path = self.persistent_path.value()
        var os = Python.import_module("os")
        var shutil = Python.import_module("shutil")
        var tempfile = Python.import_module("tempfile")
        var parent = String(os.path.dirname(path))
        if parent == "":
            parent = "."
        _ = os.makedirs(parent, exist_ok=True)
        var basename = String(os.path.basename(path))
        var tmp_path = String(
            tempfile.mkdtemp(prefix=".omendb-" + basename + "-", dir=parent)
        )
        var backup_path = path + ".bak"
        try:
            self._write_snapshot(tmp_path)
            if Bool(os.path.exists(backup_path)):
                shutil.rmtree(backup_path)
            if Bool(os.path.exists(path)):
                os.rename(path, backup_path)
            os.rename(tmp_path, path)
            if Bool(os.path.exists(backup_path)):
                shutil.rmtree(backup_path)
        except e:
            if Bool(os.path.exists(tmp_path)):
                shutil.rmtree(tmp_path)
            if not Bool(os.path.exists(path)) and Bool(
                os.path.exists(backup_path)
            ):
                os.rename(backup_path, path)
            raise e^

    def _write_snapshot(ref self, path: String) raises:
        var os = Python.import_module("os")
        _ = os.makedirs(path, exist_ok=True)
        save_hnsw(path + "/hnsw", self.index._backend)
        save_metadata_index(path + "/hnsw.midx", self.metadata_index)
        self._write_records(path)
        self._write_tombstones(path)
        self._write_text_docs(path)
        self._write_graph_edges(path)
        self._write_sparse_index(path)
        self._write_manifest(path, _snapshot_checksums(path))

    # === Query & Inspection ===

    def len(ref self) -> Int:
        return len(self.vector_external_ids) - self.deleted_count

    def count(ref self) -> Int:
        """Alias for len() — returns the number of live items."""
        return self.len()

    def memory_usage(ref self) -> String:
        """Human-readable memory usage summary."""
        return self.memory_budget.summary()

    def exists(ref self, id: String) raises -> Bool:
        """Check if an item exists and is not deleted or text-only."""
        if id not in self.external_to_vector:
            return False
        var vector_id = Int(self.external_to_vector[id])
        if self.deleted[vector_id]:
            return False
        if vector_id < len(self.text_only) and self.text_only[vector_id]:
            return False
        return True

    def close(ref self):
        """Release resources. After close(), the store should not be used."""
        # Mojo manages memory via RAII; this is a semantic marker for
        # embedded usage where the caller wants explicit lifecycle control.
        # Future: flush pending writes, release file handles, etc.
        pass

    def set(
        mut self,
        id: String,
        vector: Span[Float32, _],
        metadata: Optional[String] = None,
        source_span: Optional[String] = None,
    ) raises -> UInt64:
        if id.byte_length() == 0:
            raise Error("id must not be empty")
        if len(vector) != Self.dim:
            raise Error(
                "vector dimension mismatch: expected "
                + String(Self.dim)
                + ", got "
                + String(len(vector))
            )
        if id in self.external_to_vector:
            _ = self.delete(id)

        var vector_id = UInt64(len(self.vector_external_ids))
        self.index.insert(vector)
        self.memory_budget.record_vector_insert(Self.dim)
        self.memory_budget.record_code_insert(Self.dim)
        self.external_to_vector[id] = vector_id
        self.vector_external_ids.append(id)
        self.metadata.append(metadata)
        self.source_spans.append(source_span)
        self.deleted.append(False)
        # Timestamps
        var time_mod = Python.import_module("time")
        var ts = Float64(Float64(py=time_mod.time()))
        self.created_at.append(ts)
        self.updated_at.append(ts)
        self.superseded_at.append(0.0)
        if self.options.graph_enabled:
            _ = self.graph.add_node_with_id(id, "Vector", vector_id=vector_id)
        return vector_id

    def set_with_metadata_obj(
        mut self,
        id: String,
        vector: Span[Float32, _],
        metadata: Optional[String] = None,
        source_span: Optional[String] = None,
        metadata_obj: PythonObject = Python.none(),
    ) raises -> UInt64:
        """Like set() but also indexes metadata_obj for filtered search."""
        var vector_id = self.set(id, vector, metadata, source_span)
        if metadata_obj is not Python.none():
            self.metadata_index.index_json(Int(vector_id), metadata_obj)
        return vector_id

    def set_batch(
        mut self,
        ids: List[String],
        vectors: Span[Float32, _],
        metadata_list: Optional[List[Optional[String]]] = None,
    ) raises -> List[UInt64]:
        """Batch insert: ids list, vectors flat (len(ids)*dim), optional metadata list.
        """
        var count = len(ids)
        if count == 0:
            return List[UInt64]()
        if len(vectors) != count * Self.dim:
            raise Error(
                "vector data length mismatch: expected "
                + String(count * Self.dim)
                + ", got "
                + String(len(vectors))
            )

        var vector_ids = List[UInt64]()
        vector_ids.reserve(count)
        var time_mod = Python.import_module("time")
        var ts = Float64(Float64(py=time_mod.time()))

        for i in range(count):
            var id = ids[i]
            if id.byte_length() == 0:
                raise Error("id must not be empty")
            if id in self.external_to_vector:
                _ = self.delete(id)

            var offset = i * Self.dim
            var vec = List[Float32]()
            vec.reserve(Self.dim)
            for d in range(Self.dim):
                vec.append(vectors[offset + d])
            self.index.insert(Span(vec))
            self.memory_budget.record_vector_insert(Self.dim)
            self.memory_budget.record_code_insert(Self.dim)

            var vid = UInt64(len(self.vector_external_ids))
            self.external_to_vector[id] = vid
            self.vector_external_ids.append(id)

            var meta: Optional[String] = None
            if metadata_list is not None:
                if i < len(metadata_list.value()):
                    meta = metadata_list.value()[i]
            self.metadata.append(meta)
            self.source_spans.append(Optional[String](None))
            self.deleted.append(False)
            self.created_at.append(ts)
            self.updated_at.append(ts)
            self.superseded_at.append(0.0)
            if self.options.graph_enabled:
                _ = self.graph.add_node_with_id(id, "Vector", vector_id=vid)
            vector_ids.append(vid)

        return vector_ids^

    def delete(mut self, id: String) raises -> Bool:
        if id not in self.external_to_vector:
            return False

        var vector_id = self.external_to_vector.pop(id)
        var idx = Int(vector_id)
        if idx >= len(self.deleted):
            raise Error("delete vector ID out of range")
        if self.deleted[idx]:
            return False

        self.deleted[idx] = True
        self.deleted_count += 1
        self._tombstones_since_repair += 1
        self._check_deletion_policy()
        self.metadata_index.remove(idx)
        if self.options.graph_enabled:
            _ = self.graph.remove_node_by_external_id(id)
        return True

    def supersede(mut self, old_id: String, new_id: String) raises -> Bool:
        """Mark old_id as superseded by new_id. Returns True if successful."""
        if old_id not in self.external_to_vector:
            return False
        if new_id not in self.external_to_vector:
            raise Error("replacement item '" + new_id + "' not found")

        var old_vector_id = self.external_to_vector[old_id]
        var idx = Int(old_vector_id)
        if idx >= len(self.deleted):
            raise Error("supersede vector ID out of range")
        if self.deleted[idx]:
            return False

        # Mark as superseded (soft delete + timestamp)
        self.deleted[idx] = True
        self.deleted_count += 1
        self._tombstones_since_repair += 1
        self._check_deletion_policy()
        self.metadata_index.remove(idx)
        var time_mod = Python.import_module("time")
        if self.options.graph_enabled:
            _ = self.graph.remove_node_by_external_id(old_id)
        return True

    def needs_compaction(ref self) -> Bool:
        """True when tombstone ratio exceeds 25%%. Caller should vacuum."""
        return self._needs_compaction

    def _check_deletion_policy(mut self):
        """Flag for compaction when tombstone ratio >= 25%%."""
        var n = len(self.vector_external_ids)
        if n == 0:
            return
        var ratio = Float32(self.deleted_count) / Float32(n)
        if ratio >= 0.25:
            self._needs_compaction = True

    # === Text Search ===

    def enable_text_search(mut self):
        self.options.text_enabled = True

    def set_sparse(
        mut self,
        id: String,
        ref sparse: SparseVector,
    ) raises:
        """Set sparse vector for an existing item (separate call like index_metadata).

        Raises if sparse is not enabled or id is not found.
        """
        if not self.options.sparse_enabled:
            self.options.sparse_enabled = True
        if id not in self.external_to_vector:
            raise Error("id not found, call set() or set_text() first")
        var vector_id = Int(self.external_to_vector[id])
        # Ensure sparse_vectors is sized correctly
        while len(self.sparse_vectors) <= vector_id:
            self.sparse_vectors.append(SparseVector())
        # Build a copy for storage (SparseVector is Copyable)
        var stored = SparseVector(copy=sparse)
        self.sparse_vectors[vector_id] = stored^
        self.sparse_index.insert(sparse, UInt32(vector_id))

    # === Text Search ===

    def set_text(
        mut self,
        id: String,
        vector: Span[Float32, _],
        text: String,
        metadata: Optional[String] = None,
        source_span: Optional[String] = None,
    ) raises -> UInt64:
        if not self.options.text_enabled:
            self.enable_text_search()

        var vector_id = self.set(
            id, vector, metadata=metadata, source_span=source_span
        )
        var doc_id = self.text_index.add_document(text)
        if Int(doc_id) != len(self.text_doc_vector_ids):
            raise Error("text document ID mismatch")
        self.text_doc_vector_ids.append(vector_id)
        self.text_doc_texts.append(text)
        return vector_id

    def set_text_only(
        mut self,
        id: String,
        text: String,
        metadata: Optional[String] = None,
        source_span: Optional[String] = None,
    ) raises -> UInt64:
        if not self.options.text_enabled:
            self.enable_text_search()

        # Store a placeholder vector (all zeros) for text-only items
        var placeholder = List[Float32]()
        for _ in range(Self.dim):
            placeholder.append(0.0)
        var vector_id = self.set(
            id,
            Span(ptr=placeholder.unsafe_ptr(), length=Self.dim),
            metadata=metadata,
            source_span=source_span,
        )
        # Mark as text-only BEFORE adding text doc so that if add_document
        # raises, the item is already flagged and won't appear in vector search
        while len(self.text_only) <= Int(vector_id):
            self.text_only.append(False)
        self.text_only[Int(vector_id)] = True
        self.text_only_count += 1
        var doc_id = self.text_index.add_document(text)
        if Int(doc_id) != len(self.text_doc_vector_ids):
            raise Error("text document ID mismatch")
        self.text_doc_vector_ids.append(vector_id)
        self.text_doc_texts.append(text)
        return vector_id

    def get(ref self, id: String) raises -> List[Float32]:
        if id not in self.external_to_vector:
            return List[Float32]()

        var vector_id = self.external_to_vector[id]
        var span = self.index.get_vector(Int(vector_id))
        var result = List[Float32]()
        for i in range(len(span)):
            result.append(span[i])
        return result^

    def search_text(
        ref self, query: String, k: Int = 10
    ) raises -> List[SearchResult]:
        if not self.options.text_enabled:
            raise Error(
                "text search not enabled: call enable_text_search() first"
            )
        if k <= 0:
            return List[SearchResult]()

        var candidate_k = k + self.deleted_count
        var text_results = self.text_index.search(query, candidate_k)
        var results = List[SearchResult]()
        for i in range(len(text_results)):
            var doc_id = Int(text_results[i][0])
            var score = text_results[i][1]
            if doc_id >= len(self.text_doc_vector_ids):
                raise Error("text document vector mapping missing")
            var vector_id = self.text_doc_vector_ids[doc_id]
            if Int(vector_id) >= len(self.vector_external_ids):
                raise Error("text document vector_id out of bounds")
            if self.deleted[Int(vector_id)]:
                continue
            var external_id = self.vector_external_ids[Int(vector_id)]
            var result_metadata = self.metadata[Int(vector_id)]
            var result_source_span = self.source_spans[Int(vector_id)]
            results.append(
                SearchResult(
                    external_id,
                    vector_id,
                    -score,
                    metadata=result_metadata,
                    source_span=result_source_span,
                )
            )
            if len(results) == k:
                break
        return results^

    def search_hybrid(
        ref self,
        query: Span[Float32, _],
        text_query: String,
        k: Int = 10,
        alpha: Float32 = 0.5,
        rrf_k: Int = 60,
        ef_search: Int = 200,
        sparse_query: Optional[SparseVector] = None,
    ) raises -> List[SearchResult]:
        """Hybrid search: dense + sparse + BM25 with RRF fusion.

        When sparse_query is not None, does 3-way RRF:
          alpha * RRF(dense_rank) + (1-alpha)/2 * RRF(sparse_rank) + (1-alpha)/2 * RRF(text_rank)
        When sparse_query is None, falls back to 2-way (dense + BM25) unchanged.
        """
        if len(query) != Self.dim:
            raise Error(
                "query dimension mismatch: expected "
                + String(Self.dim)
                + ", got "
                + String(len(query))
            )
        if k <= 0:
            return List[SearchResult]()
        if alpha < 0.0 or alpha > 1.0:
            raise Error("alpha must be between 0.0 and 1.0")

        var candidate_k = k + self.deleted_count

        # --- Vector search (HNSW) ---
        var vector_candidates = self.index.search(
            query,
            k=candidate_k,
            ef=max(ef_search, candidate_k),
        )
        # Filter deleted/text_only and build rank map
        var vector_ranks = Dict[UInt64, Int]()
        var vector_distances = Dict[UInt64, Float32]()
        var vector_rank_counter = 0
        for i in range(len(vector_candidates)):
            var vid = UInt64(vector_candidates[i].id)
            if self.deleted[Int(vid)]:
                continue
            if Int(vid) < len(self.text_only) and self.text_only[Int(vid)]:
                continue
            vector_rank_counter += 1
            vector_ranks[vid] = vector_rank_counter
            vector_distances[vid] = vector_candidates[i].distance

        # --- Text search (BM25) ---
        var text_rank_map = Dict[UInt64, Int]()
        if self.options.text_enabled:
            var text_results = self.text_index.search(text_query, candidate_k)
            var text_rank_counter = 0
            for i in range(len(text_results)):
                var doc_id = Int(text_results[i][0])
                if doc_id >= len(self.text_doc_vector_ids):
                    continue
                var vid = self.text_doc_vector_ids[doc_id]
                if Int(vid) >= len(self.vector_external_ids):
                    continue
                if self.deleted[Int(vid)]:
                    continue
                text_rank_counter += 1
                text_rank_map[vid] = text_rank_counter

        # --- Sparse search ---
        var sparse_rank_map = Dict[UInt64, Int]()
        if sparse_query is not None:
            var sparse_results = self.sparse_index.search(
                sparse_query.value(), candidate_k
            )
            var sparse_rank_counter = 0
            for i in range(len(sparse_results)):
                var doc_id = Int(sparse_results[i].doc_id)
                var vid = UInt64(doc_id)
                if Int(vid) >= len(self.vector_external_ids):
                    continue
                if self.deleted[Int(vid)]:
                    continue
                if Int(vid) < len(self.text_only) and self.text_only[Int(vid)]:
                    continue
                sparse_rank_counter += 1
                sparse_rank_map[vid] = sparse_rank_counter

        # --- RRF fusion ---
        # Collect all candidate vector_ids from all searches
        var candidate_ids = List[UInt64]()
        var seen = Dict[UInt64, Bool]()
        for vid in vector_ranks:
            if vid not in seen:
                candidate_ids.append(vid)
                seen[vid] = True
        for vid in text_rank_map:
            if vid not in seen:
                candidate_ids.append(vid)
                seen[vid] = True
        for vid in sparse_rank_map:
            if vid not in seen:
                candidate_ids.append(vid)
                seen[vid] = True

        # Compute RRF scores
        var rrf_scores = Dict[UInt64, Float32]()
        var use_sparse = sparse_query is not None
        for i in range(len(candidate_ids)):
            var vid = candidate_ids[i]
            var score: Float32 = 0.0
            if vid in vector_ranks:
                score += alpha / (Float32(rrf_k) + Float32(vector_ranks[vid]))
            if vid in text_rank_map:
                var text_weight: Float32 = 1.0 - alpha
                if use_sparse:
                    text_weight = text_weight / 2.0
                score += text_weight / (
                    Float32(rrf_k) + Float32(text_rank_map[vid])
                )
            if vid in sparse_rank_map:
                score += (
                    (1.0 - alpha)
                    / 2.0
                    / (Float32(rrf_k) + Float32(sparse_rank_map[vid]))
                )
            rrf_scores[vid] = score

        # Partial sort for top-k
        var result_ids = List[UInt64]()
        for vid in candidate_ids:
            result_ids.append(vid)

        # Simple selection sort for top-k
        for i in range(min(k, len(result_ids))):
            var best_idx = i
            var best_score = rrf_scores[result_ids[i]]
            for j in range(i + 1, len(result_ids)):
                var s = rrf_scores[result_ids[j]]
                if s > best_score:
                    best_score = s
                    best_idx = j
            if best_idx != i:
                var tmp = result_ids[i]
                result_ids[i] = result_ids[best_idx]
                result_ids[best_idx] = tmp

        # Build results
        var results = List[SearchResult]()
        for i in range(min(k, len(result_ids))):
            var vid = result_ids[i]
            var rrf_score = rrf_scores[vid]
            # Use actual vector distance if available, else text-only distance placeholder
            var dist: Float32 = 0.0
            if vid in vector_distances:
                dist = vector_distances[vid]
            var external_id = self.vector_external_ids[Int(vid)]
            results.append(
                SearchResult(
                    external_id,
                    vid,
                    dist,
                    metadata=self.metadata[Int(vid)],
                    source_span=self.source_spans[Int(vid)],
                    rrf_score=rrf_score,
                    has_rrf_score=True,
                )
            )

        return results^

    # === Graph Operations ===

    def enable_graph(mut self) raises:
        if self.options.graph_enabled:
            return

        self.options.graph_enabled = True
        for i in range(len(self.vector_external_ids)):
            _ = self.graph.add_node_with_id(
                self.vector_external_ids[i],
                "Vector",
                vector_id=UInt64(i),
            )

    def add_edge(
        mut self,
        from_id: String,
        to_id: String,
        edge_type: String,
        weight: Optional[Float32] = None,
    ) raises -> UInt64:
        if not self.options.graph_enabled:
            self.enable_graph()
        if (
            from_id not in self.external_to_vector
            or to_id not in self.external_to_vector
        ):
            raise Error(
                "edge source or target not found: both from_id and to_id must"
                " be added first"
            )

        var src = self.graph.node_id_for(from_id)
        var dst = self.graph.node_id_for(to_id)
        if not src or not dst:
            raise Error(
                "edge source or target not found: both from_id and to_id must"
                " be added first"
            )
        return self.graph.add_edge(src.value(), dst.value(), edge_type, weight)

    def remove_edge(
        mut self, from_id: String, to_id: String, edge_type: String
    ) raises -> Bool:
        if not self.options.graph_enabled:
            raise Error("graph not enabled: call enable_graph() first")

        var src = self.graph.node_id_for(from_id)
        var dst = self.graph.node_id_for(to_id)
        if not src or not dst:
            return False
        return self.graph.remove_edge(src.value(), dst.value(), edge_type)

    def neighbors(
        ref self,
        id: String,
        direction: GraphDirection = GraphDirection.OUTGOING,
        edge_type: Optional[String] = None,
    ) raises -> List[String]:
        if not self.options.graph_enabled:
            raise Error("graph not enabled: call enable_graph() first")

        var node_id = self.graph.node_id_for(id)
        var result = List[String]()
        if not node_id or id not in self.external_to_vector:
            return result^

        var node_neighbors = self.graph.neighbors(
            node_id.value(), direction, edge_type
        )
        for i in range(len(node_neighbors)):
            var external_id = self.graph.external_id_for(node_neighbors[i])
            if external_id and external_id.value() in self.external_to_vector:
                result.append(external_id.value())
        return result^

    def traverse(
        ref self,
        start_id: String,
        direction: GraphDirection = GraphDirection.OUTGOING,
        max_depth: Int = 10,
        edge_type: Optional[String] = None,
    ) raises -> List[String]:
        if not self.options.graph_enabled:
            raise Error("graph not enabled: call enable_graph() first")

        var node_id = self.graph.node_id_for(start_id)
        var result = List[String]()
        if not node_id or start_id not in self.external_to_vector:
            return result^

        var node_results = self.graph.traverse(
            node_id.value(), direction, max_depth, edge_type
        )
        for i in range(len(node_results)):
            var external_id = self.graph.external_id_for(node_results[i])
            if external_id and external_id.value() in self.external_to_vector:
                result.append(external_id.value())
        return result^

    def has_path(
        ref self,
        from_id: String,
        to_id: String,
        direction: GraphDirection = GraphDirection.OUTGOING,
        max_depth: Int = 10,
        edge_type: Optional[String] = None,
    ) raises -> Bool:
        if not self.options.graph_enabled:
            raise Error("graph not enabled: call enable_graph() first")
        return (
            len(
                self.shortest_path(
                    from_id, to_id, direction, max_depth, edge_type
                )
            )
            > 0
        )

    def shortest_path(
        ref self,
        from_id: String,
        to_id: String,
        direction: GraphDirection = GraphDirection.OUTGOING,
        max_depth: Int = 10,
        edge_type: Optional[String] = None,
    ) raises -> List[String]:
        if not self.options.graph_enabled:
            raise Error("graph not enabled: call enable_graph() first")
        if (
            from_id not in self.external_to_vector
            or to_id not in self.external_to_vector
        ):
            return List[String]()
        var path = self.graph.shortest_path(
            from_id, to_id, direction, max_depth, edge_type
        )
        for i in range(len(path)):
            if path[i] not in self.external_to_vector:
                return List[String]()
        return path^

    # === Vector Search ===

    def search(
        ref self,
        query: Span[Float32, _],
        options: SearchOptions = SearchOptions(),
        allowlist: Optional[List[UInt8]] = None,
    ) raises -> List[SearchResult]:
        if len(query) != Self.dim:
            raise Error(
                "query dimension mismatch: expected "
                + String(Self.dim)
                + ", got "
                + String(len(query))
            )
        if options.k <= 0:
            return List[SearchResult]()

        if allowlist:
            # Count live allowed items (skip deleted and text_only)
            var allowed_count = 0
            var live_count = 0
            for i in range(len(allowlist.value())):
                if self.deleted[i]:
                    continue
                if i < len(self.text_only) and self.text_only[i]:
                    continue
                live_count += 1
                if allowlist.value()[i] == 1:
                    allowed_count += 1
            var selectivity = Float32(allowed_count) / Float32(
                max(live_count, 1)
            )

            if selectivity < 0.1 and allowed_count <= options.k * 10:
                # Highly selective: exact scan over allowlist
                return self._exact_scan(query, options, allowlist.value())^

            # Build effective allowlist: merge caller allowlist with deleted/text_only masks.
            # This lets search_filtered skip deleted/text_only items during traversal
            # instead of wasting exploration on items that will be post-filtered.
            var effective_allowed = List[UInt8](
                capacity=len(self.vector_external_ids)
            )
            for i in range(len(self.vector_external_ids)):
                if i < len(allowlist.value()) and allowlist.value()[i] == 0:
                    effective_allowed.append(0)
                elif self.deleted[i]:
                    effective_allowed.append(0)
                elif i < len(self.text_only) and self.text_only[i]:
                    effective_allowed.append(0)
                else:
                    effective_allowed.append(1)

            # Moderate and high selectivity: HNSW in-filtering (SOTA neighborhood-adaptive)
            var ef = max(options.ef_search, options.k * 2)
            var candidates = self.index.search_filtered(
                query,
                effective_allowed,
                k=options.k,
                ef=ef,
            )
            var results = List[SearchResult]()
            for i in range(len(candidates)):
                var candidate = candidates[i]
                if (
                    options.max_distance
                    and candidate.distance > options.max_distance.value()
                ):
                    continue
                var vector_id = UInt64(candidate.id)
                var external_id = self.vector_external_ids[Int(vector_id)]
                results.append(
                    SearchResult(
                        external_id,
                        vector_id,
                        candidate.distance,
                        metadata=self.metadata[Int(vector_id)],
                        source_span=self.source_spans[Int(vector_id)],
                    )
                )
            return results^

        # No allowlist: standard HNSW search with deleted/text_only post-filter
        var effective_k = options.k
        if self.deleted_count > 0:
            effective_k = effective_k + self.deleted_count
        if self.text_only_count > 0:
            effective_k = effective_k + self.text_only_count
        var candidates = self.index.search(
            query,
            k=effective_k,
            ef=max(options.ef_search, effective_k),
        )
        var results = List[SearchResult]()
        for i in range(len(candidates)):
            if len(results) >= options.k:
                break
            var candidate = candidates[i]
            if (
                options.max_distance
                and candidate.distance > options.max_distance.value()
            ):
                continue
            var vector_id = UInt64(candidate.id)
            if self.deleted[Int(vector_id)]:
                continue
            if (
                Int(vector_id) < len(self.text_only)
                and self.text_only[Int(vector_id)]
            ):
                continue
            var external_id = self.vector_external_ids[Int(vector_id)]
            results.append(
                SearchResult(
                    external_id,
                    vector_id,
                    candidate.distance,
                    metadata=self.metadata[Int(vector_id)],
                    source_span=self.source_spans[Int(vector_id)],
                )
            )
        return results^

    # === Internal Search Helpers ===

    def _exact_scan(
        ref self,
        query: Span[Float32, _],
        options: SearchOptions,
        allowlist: List[UInt8],
    ) raises -> List[SearchResult]:
        """Exact distance scan over allowlisted items. Used for highly selective filters.
        """
        # Collect (distance, vector_id) for all allowed items
        var distances = List[Float32]()
        var ids = List[UInt32]()
        var n = len(self.vector_external_ids)

        for i in range(n):
            if i >= len(allowlist) or allowlist[i] == 0:
                continue
            # Skip deleted and text_only items
            if self.deleted[i]:
                continue
            if i < len(self.text_only) and self.text_only[i]:
                continue
            # Compute distance via backend contract
            var vec_span = self.index.get_vector(i)
            var dist = l2_distance[Self.dim](query, vec_span)
            if options.max_distance and dist > options.max_distance.value():
                continue
            distances.append(dist)
            ids.append(UInt32(i))

        # Partial sort to get top-k
        var k = min(options.k, len(distances))
        var results = List[SearchResult]()
        # Simple selection sort for top-k (k is small)
        for _ in range(k):
            var min_idx = -1
            var min_dist: Float32 = 1e30
            for j in range(len(distances)):
                if distances[j] < min_dist:
                    min_dist = distances[j]
                    min_idx = j
            if min_idx >= 0:
                var vid = ids[min_idx]
                results.append(
                    SearchResult(
                        self.vector_external_ids[Int(vid)],
                        UInt64(vid),
                        distances[min_idx],
                        metadata=self.metadata[Int(vid)],
                        source_span=self.source_spans[Int(vid)],
                    )
                )
                # Remove selected
                _ = distances.pop(min_idx)
                _ = ids.pop(min_idx)
        return results^

    def _live_allowlist(ref self) -> List[UInt8]:
        var allowed = List[UInt8]()
        allowed.reserve(len(self.vector_external_ids))
        for i in range(len(self.vector_external_ids)):
            if self.deleted[i]:
                allowed.append(0)
            elif i < len(self.text_only) and self.text_only[i]:
                allowed.append(0)
            else:
                allowed.append(1)
        return allowed^

    # === Persistence ===

    def _write_manifest(ref self, path: String, checksums: PythonObject) raises:
        var json = Python.import_module("json")
        var builtins = Python.import_module("builtins")
        var manifest = Python.dict()
        manifest["format_version"] = 1
        manifest["store_layout"] = "single_segment_v1"
        manifest["segment_id"] = "segment0"
        manifest["index_mode"] = "hnsw"
        manifest["dim"] = Self.dim
        if self.options.metric == Metric.COSINE:
            manifest["metric"] = "cosine"
        elif self.options.metric == Metric.DOT:
            manifest["metric"] = "dot"
        else:
            manifest["metric"] = "l2"
        manifest["num_vectors"] = len(self.vector_external_ids)
        manifest["record_count"] = len(self.vector_external_ids)
        manifest["tombstone_count"] = self.deleted_count
        manifest["M"] = self.index._M()
        manifest["ef_construction"] = self.index._ef_construction()
        manifest["alpha"] = self.index._alpha()
        manifest["text_enabled"] = self.options.text_enabled
        manifest["graph_enabled"] = self.options.graph_enabled
        manifest["sparse_enabled"] = self.options.sparse_enabled
        manifest["config_fingerprint"] = _config_fingerprint(
            Self.dim,
            String(manifest["metric"]),
            self.index._M(),
            self.index._ef_construction(),
            self.index._alpha(),
            self.options.text_enabled,
            self.options.graph_enabled,
        )

        var files = Python.dict()
        files["hnsw_prefix"] = "hnsw"
        files["records"] = "records.json"
        files["tombstones"] = "tombstones.json"
        if self.options.text_enabled:
            files["text_docs"] = "text_docs.json"
        if self.options.graph_enabled:
            files["graph_edges"] = "graph_edges.json"
        manifest["files"] = files
        manifest["checksums"] = checksums

        var manifest_file = builtins.open(path + "/manifest.json", "w")
        _ = manifest_file.write(json.dumps(manifest, sort_keys=True))
        _ = manifest_file.close()

    def _write_records(ref self, path: String) raises:
        var json = Python.import_module("json")
        var builtins = Python.import_module("builtins")
        var records = Python.list()
        for i in range(len(self.vector_external_ids)):
            var record = Python.dict()
            record["id"] = self.vector_external_ids[i]
            record["vector_id"] = i
            if self.metadata[i]:
                record["has_metadata"] = True
                record["metadata"] = self.metadata[i].value()
            else:
                record["has_metadata"] = False
                record["metadata"] = ""
            if self.source_spans[i]:
                record["has_source"] = True
                record["source"] = self.source_spans[i].value()
            else:
                record["has_source"] = False
                record["source"] = ""
            if i < len(self.text_only) and self.text_only[i]:
                record["text_only"] = True
            else:
                record["text_only"] = False
            # Timestamps
            if i < len(self.created_at):
                record["created_at"] = self.created_at[i]
                record["updated_at"] = self.updated_at[i]
                if self.superseded_at[i] > 0.0:
                    record["superseded_at"] = self.superseded_at[i]
            records.append(record)

        var records_file = builtins.open(path + "/records.json", "w")
        _ = records_file.write(json.dumps(records))
        _ = records_file.close()

    def _write_tombstones(ref self, path: String) raises:
        var json = Python.import_module("json")
        var builtins = Python.import_module("builtins")
        var tombstones = Python.list()
        for i in range(len(self.vector_external_ids)):
            var tombstone = Python.dict()
            tombstone["vector_id"] = i
            tombstone["deleted"] = self.deleted[i]
            tombstones.append(tombstone)

        var tombstones_file = builtins.open(path + "/tombstones.json", "w")
        _ = tombstones_file.write(json.dumps(tombstones))
        _ = tombstones_file.close()

    def _load_records(mut self, path: String) raises:
        var json = Python.import_module("json")
        var builtins = Python.import_module("builtins")
        var os = Python.import_module("os")
        if not os.path.exists(path + "/records.json"):
            raise Error("persistent store records not found")

        var records_file = builtins.open(path + "/records.json", "r")
        var records = json.loads(records_file.read())
        _ = records_file.close()

        self.external_to_vector = Dict[String, UInt64]()
        self.vector_external_ids = List[String]()
        self.metadata = List[Optional[String]]()
        self.source_spans = List[Optional[String]]()
        self.deleted = List[Bool]()
        self.deleted_count = 0
        self.text_only = List[Bool]()
        self.text_only_count = 0

        for i in range(Int(py=builtins.len(records))):
            var record = records[i]
            var id = String(record["id"])
            var vector_id = UInt64(Int(py=record["vector_id"]))
            if Int(vector_id) != i:
                raise Error("persistent store vector ID mismatch")
            self.external_to_vector[id] = vector_id
            self.vector_external_ids.append(id)
            self.deleted.append(False)
            if Bool(record.get("text_only", False)):
                self.text_only.append(True)
                self.text_only_count += 1
            else:
                self.text_only.append(False)
            if Bool(record["has_metadata"]):
                self.metadata.append(
                    Optional[String](String(record["metadata"]))
                )
            else:
                self.metadata.append(Optional[String](None))
            if Bool(record.get("has_source", False)):
                self.source_spans.append(
                    Optional[String](String(record["source"]))
                )
            else:
                self.source_spans.append(Optional[String](None))
            # Timestamps
            self.created_at.append(Float64(py=record.get("created_at", 0.0)))
            self.updated_at.append(Float64(py=record.get("updated_at", 0.0)))
            self.superseded_at.append(
                Float64(py=record.get("superseded_at", 0.0))
            )

    def _load_tombstones(mut self, path: String) raises:
        var json = Python.import_module("json")
        var builtins = Python.import_module("builtins")
        var os = Python.import_module("os")
        if not os.path.exists(path + "/tombstones.json"):
            raise Error("persistent store tombstones not found")

        var tombstones_file = builtins.open(path + "/tombstones.json", "r")
        var tombstones = json.loads(tombstones_file.read())
        _ = tombstones_file.close()

        if Int(py=builtins.len(tombstones)) != len(self.vector_external_ids):
            raise Error("persistent store tombstone count mismatch")
        self.external_to_vector = Dict[String, UInt64]()
        self.deleted_count = 0
        for i in range(Int(py=builtins.len(tombstones))):
            var tombstone = tombstones[i]
            var vector_id = Int(py=tombstone["vector_id"])
            if vector_id != i:
                raise Error("persistent store tombstone vector ID mismatch")
            if Bool(tombstone["deleted"]):
                self.deleted[i] = True
                self.deleted_count += 1
            else:
                self.external_to_vector[self.vector_external_ids[i]] = UInt64(i)

    def _write_text_docs(ref self, path: String) raises:
        if not self.options.text_enabled:
            return

        var json = Python.import_module("json")
        var builtins = Python.import_module("builtins")
        var docs = Python.list()
        for i in range(len(self.text_doc_vector_ids)):
            var doc = Python.dict()
            doc["vector_id"] = Int(self.text_doc_vector_ids[i])
            doc["text"] = self.text_doc_texts[i]
            docs.append(doc)

        var docs_file = builtins.open(path + "/text_docs.json", "w")
        _ = docs_file.write(json.dumps(docs))
        _ = docs_file.close()

    def _write_graph_edges(ref self, path: String) raises:
        if not self.options.graph_enabled:
            return

        var json = Python.import_module("json")
        var builtins = Python.import_module("builtins")
        var edges = Python.list()
        for i in range(len(self.graph.edges)):
            ref edge = self.graph.edges[i]
            var from_id = self.graph.external_id_for(edge.src)
            var to_id = self.graph.external_id_for(edge.dst)
            if not from_id or not to_id:
                raise Error("graph edge endpoint missing external ID")

            var edge_record = Python.dict()
            edge_record["from"] = from_id.value()
            edge_record["to"] = to_id.value()
            edge_record["edge_type"] = edge.edge_type
            if edge.weight:
                edge_record["has_weight"] = True
                edge_record["weight"] = edge.weight.value()
            else:
                edge_record["has_weight"] = False
                edge_record["weight"] = 0.0
            edges.append(edge_record)

        var edges_file = builtins.open(path + "/graph_edges.json", "w")
        _ = edges_file.write(json.dumps(edges))
        _ = edges_file.close()

    def _load_snapshot_features(
        mut self, path: String, manifest: PythonObject
    ) raises:
        if Bool(manifest["text_enabled"]):
            self._load_text_docs(path)
        if Bool(manifest["graph_enabled"]):
            self._load_graph_edges(path)
        if "sparse_enabled" in manifest and Bool(manifest["sparse_enabled"]):
            self._load_sparse_index(path)

    def _load_text_docs(mut self, path: String) raises:
        var json = Python.import_module("json")
        var builtins = Python.import_module("builtins")
        var os = Python.import_module("os")
        if not os.path.exists(path + "/text_docs.json"):
            raise Error("persistent store text docs not found")

        var docs_file = builtins.open(path + "/text_docs.json", "r")
        var docs = json.loads(docs_file.read())
        _ = docs_file.close()

        self.options.text_enabled = True
        self.text_index = BM25Index()
        self.text_doc_vector_ids = List[UInt64]()
        self.text_doc_texts = List[String]()

        for i in range(Int(py=builtins.len(docs))):
            var doc = docs[i]
            var vector_id = UInt64(Int(py=doc["vector_id"]))
            if Int(vector_id) >= len(self.vector_external_ids):
                raise Error("persistent text doc vector ID out of range")
            var text = String(doc["text"])
            var doc_id = self.text_index.add_document(text)
            if Int(doc_id) != i:
                raise Error("persistent text doc ID mismatch")
            self.text_doc_vector_ids.append(vector_id)
            self.text_doc_texts.append(text)

    def _load_graph_edges(mut self, path: String) raises:
        var json = Python.import_module("json")
        var builtins = Python.import_module("builtins")
        var os = Python.import_module("os")
        if not os.path.exists(path + "/graph_edges.json"):
            raise Error("persistent store graph edges not found")

        var edges_file = builtins.open(path + "/graph_edges.json", "r")
        var edges = json.loads(edges_file.read())
        _ = edges_file.close()

        self.enable_graph()
        for i in range(Int(py=builtins.len(edges))):
            var edge = edges[i]
            var weight = Optional[Float32](None)
            if Bool(edge["has_weight"]):
                weight = Optional[Float32](Float32(py=edge["weight"]))
            _ = self.add_edge(
                String(edge["from"]),
                String(edge["to"]),
                String(edge["edge_type"]),
                weight=weight,
            )

    def _write_sparse_index(ref self, path: String) raises:
        """Save sparse inverted index as JSON."""
        if not self.options.sparse_enabled or self.sparse_index.num_docs == 0:
            return
        var json = Python.import_module("json")
        var builtins = Python.import_module("builtins")
        var data = Python.dict()
        data["num_docs"] = self.sparse_index.num_docs
        data["max_dim"] = Int(self.sparse_index.max_dim)
        # Serialize dimensions and their posting lists
        var dim_list = Python.list()
        var lengths = Python.list()
        var all_postings = Python.list()
        var dim_keys = List[UInt32]()
        for d in self.sparse_index.dim_to_idx:
            dim_keys.append(d)
        for di in range(len(dim_keys)):
            var d = dim_keys[di]
            _ = dim_list.append(Int(d))
            var idx = self.sparse_index.dim_to_idx[d]
            ref pl = self.sparse_index.posting_lists[idx]
            _ = lengths.append(len(pl))
            for pi in range(len(pl)):
                var p = pl[pi]
                var entry = Python.dict()
                entry["doc_id"] = Int(p.doc_id)
                entry["weight"] = Float64(p.weight)
                _ = all_postings.append(entry)
        data["dimensions"] = dim_list
        data["posting_list_lengths"] = lengths
        data["postings"] = all_postings
        var f = builtins.open(path + "/sparse_index.json", "w")
        _ = f.write(json.dumps(data))
        _ = f.close()

    def _load_sparse_index(mut self, path: String) raises:
        """Load sparse inverted index from JSON."""
        var json = Python.import_module("json")
        var builtins = Python.import_module("builtins")
        var os = Python.import_module("os")
        if not os.path.exists(path + "/sparse_index.json"):
            return
        var f = builtins.open(path + "/sparse_index.json", "r")
        var data = json.loads(f.read())
        _ = f.close()
        self.options.sparse_enabled = True
        self.sparse_index = SparseInvertedIndex()
        self.sparse_index.num_docs = Int(py=data["num_docs"])
        self.sparse_index.max_dim = UInt32(Int(py=data["max_dim"]))
        var dims = data["dimensions"]
        var lengths = data["posting_list_lengths"]
        var postings = data["postings"]
        var posting_idx = 0
        for di in range(len(dims)):
            var d = UInt32(Int(py=dims[di]))
            var pl_len = Int(py=lengths[di])
            var pl = List[SparsePosting]()
            for pi in range(pl_len):
                var p = postings[posting_idx]
                var doc_id = UInt32(Int(py=p["doc_id"]))
                var weight = Float32(py=p["weight"])
                pl.append(SparsePosting(doc_id, weight))
                posting_idx += 1
            self.sparse_index.dim_to_idx[d] = len(
                self.sparse_index.posting_lists
            )
            self.sparse_index.posting_lists.append(pl^)

    @staticmethod
    def _validate_manifest_layout(path: String, manifest: PythonObject) raises:
        var os = Python.import_module("os")
        if Int(py=manifest["format_version"]) != 1:
            raise Error("persistent store format version mismatch")
        if String(manifest["store_layout"]) != "single_segment_v1":
            raise Error("persistent store layout mismatch")
        if String(manifest["index_mode"]) != "hnsw":
            raise Error("persistent store index mode mismatch")
        var metric_str = String(manifest["metric"])
        if metric_str not in ("l2", "cosine", "dot"):
            raise Error("persistent store metric mismatch")

        if not os.path.exists(path + "/records.json"):
            raise Error("persistent store records not found")
        if not os.path.exists(path + "/tombstones.json"):
            raise Error("persistent store tombstones not found")
        if not os.path.exists(path + "/hnsw.meta"):
            raise Error("persistent store HNSW metadata not found")
        _ = manifest["checksums"]

    @staticmethod
    def _validate_loaded_config(
        manifest: PythonObject, ref loaded_index: VectorIndex[Self.dim]
    ) raises:
        var manifest_count = Int(py=manifest["num_vectors"])
        if manifest_count != loaded_index.num_elements():
            raise Error("persistent store vector count mismatch")
        if Int(py=manifest["record_count"]) != manifest_count:
            raise Error(
                "store integrity error: record count mismatch between manifest"
                " and data"
            )
        if Int(py=manifest["tombstone_count"]) < 0:
            raise Error("persistent store tombstone count mismatch")
        if Int(py=manifest["M"]) != loaded_index._M():
            raise Error("persistent store HNSW M mismatch")
        if (
            Int(py=manifest["ef_construction"])
            != loaded_index._ef_construction()
        ):
            raise Error("persistent store HNSW ef_construction mismatch")
        if Float32(py=manifest["alpha"]) != loaded_index._alpha():
            raise Error("persistent store HNSW alpha mismatch")

        var expected = _config_fingerprint(
            Self.dim,
            String(manifest["metric"]),
            Int(py=manifest["M"]),
            Int(py=manifest["ef_construction"]),
            Float32(py=manifest["alpha"]),
            Bool(manifest["text_enabled"]),
            Bool(manifest["graph_enabled"]),
        )
        if String(manifest["config_fingerprint"]) != expected:
            raise Error("persistent store config fingerprint mismatch")
