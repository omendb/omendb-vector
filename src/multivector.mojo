from std.collections import Dict, List
from std.ffi import c_int, external_call, get_errno
from std.math import sqrt
from std.os import listdir, makedirs, remove, rmdir
from std.os.path import basename, dirname, exists, isdir, isfile
from std.tempfile import mkdtemp
from hnsw import HNSWIndex
from persistence import (
    load_list_f32,
    load_list_int,
    load_list_u8,
    save_list_f32,
    save_list_int,
    save_list_u8,
)
from text import BM25Index


comptime MANIFEST_FORMAT_VERSION = 2
comptime ENCODING_NONE = 0
comptime ENCODING_MUVERA = 1
comptime RECORD_META_WIDTH = 8
comptime SOURCE_META_WIDTH = 2
comptime TEXT_META_WIDTH = 3


@fieldwise_init
struct MultiVectorManifest(ImplicitlyCopyable):
    var format_version: Int
    var dim: Int
    var record_count: Int
    var tombstone_count: Int
    var text_enabled: Bool
    var encoding_mode: Int


def _append_string_bytes(
    mut buffer: List[UInt8], value: String
) -> Tuple[Int, Int]:
    var offset = len(buffer)
    var ptr = value.unsafe_ptr()
    for i in range(value.byte_length()):
        buffer.append(UInt8(ptr[i]))
    return (offset, value.byte_length())


def _string_from_bytes(
    ref buffer: List[UInt8], offset: Int, length: Int
) raises -> String:
    if offset < 0 or length < 0 or offset + length > len(buffer):
        raise Error("persistent multivector string bounds mismatch")
    if length == 0:
        return String("")
    return String(
        from_utf8=Span(
            ptr=(buffer.unsafe_ptr() + offset).bitcast[Byte](), length=length
        )
    )


def _encoding_json(encoding_mode: Int) raises -> String:
    if encoding_mode == ENCODING_NONE:
        return "none"
    if encoding_mode == ENCODING_MUVERA:
        return "muvera"
    raise Error("persistent multivector encoding mismatch")


def _bool_json(value: Bool) -> String:
    if value:
        return "true"
    return "false"


def _rename(old_path: String, new_path: String) raises:
    var old_path_c = old_path
    var new_path_c = new_path
    var error = external_call["rename", c_int](
        old_path_c.as_c_string_slice().unsafe_ptr(),
        new_path_c.as_c_string_slice().unsafe_ptr(),
    )
    if error != 0:
        var err = get_errno()
        raise Error(
            "Can not rename path: ",
            old_path,
            " -> ",
            new_path,
            " Err: ",
            String(err),
        )


def _remove_tree(path: String) raises:
    if not exists(path):
        return
    for name in listdir(path):
        var child = path + "/" + name
        if isdir(child):
            _remove_tree(child)
        else:
            remove(child)
    rmdir(path)


struct MultiVectorResult(Copyable, Movable):
    var id: String
    var doc_id: UInt64
    var distance: Float32
    var metadata: Optional[String]
    var source_span: Optional[String]

    def __init__(
        out self,
        id: String,
        doc_id: UInt64,
        distance: Float32,
        metadata: Optional[String] = None,
        source_span: Optional[String] = None,
    ):
        self.id = id
        self.doc_id = doc_id
        self.distance = distance
        self.metadata = metadata
        self.source_span = source_span

    def __init__(out self, *, copy: Self):
        self.id = copy.id
        self.doc_id = copy.doc_id
        self.distance = copy.distance
        self.metadata = copy.metadata
        self.source_span = copy.source_span

    def __init__(out self, *, deinit take: Self):
        self.id = take.id^
        self.doc_id = take.doc_id
        self.distance = take.distance
        self.metadata = take.metadata^
        self.source_span = take.source_span^


struct MultiVectorFDEIndex[
    dim: Int, partition_count: Int, repetition_count: Int
](Movable):
    comptime FDE_DIM = Self.dim * Self.partition_count * Self.repetition_count

    var doc_ids: List[UInt64]
    var hnsw: HNSWIndex[Self.FDE_DIM]

    def __init__(
        out self,
        M: Int = 16,
        ef_construction: Int = 100,
    ):
        self.doc_ids = List[UInt64]()
        self.hnsw = HNSWIndex[Self.FDE_DIM](
            M=M,
            ef_construction=ef_construction,
            min_val=-1.0,
            max_val=1.0,
        )

    def __init__(out self, *, deinit take: Self):
        self.doc_ids = take.doc_ids^
        self.hnsw = take.hnsw^

    def reserve(mut self, capacity: Int):
        self.doc_ids.reserve(capacity)
        self.hnsw.reserve(capacity)

    def len(ref self) -> Int:
        return len(self.doc_ids)

    def encoded_float_count(ref self) -> Int:
        return len(self.doc_ids) * Self.FDE_DIM

    def encoded_code_count(ref self) -> Int:
        return len(self.hnsw.codes)

    def graph_u32_count(ref self) -> Int:
        var total = 0
        for layer in range(self.hnsw.graph.layer_count()):
            total += self.hnsw.graph.layer_neighbors_len(layer)
            total += self.hnsw.graph.layer_counts_len(layer)
        return total

    def insert(
        mut self,
        doc_id: UInt64,
        vectors: Span[Float32, _],
        vector_count: Int,
    ) raises:
        var encoding = Self.encode_vectors(vectors, vector_count)
        self.doc_ids.append(doc_id)
        self.hnsw.insert(Span(ptr=encoding.unsafe_ptr(), length=len(encoding)))

    def search(
        ref self,
        query_vectors: Span[Float32, _],
        query_vector_count: Int,
        candidate_k: Int,
        ef_search: Int = 100,
    ) raises -> List[UInt64]:
        if candidate_k <= 0:
            return List[UInt64]()
        var encoding = Self.encode_vectors(query_vectors, query_vector_count)
        return self.search_encoded(
            Span(ptr=encoding.unsafe_ptr(), length=len(encoding)),
            candidate_k,
            ef_search=ef_search,
        )

    def search_encoded(
        ref self,
        encoded_query: Span[Float32, _],
        candidate_k: Int,
        ef_search: Int = 100,
    ) raises -> List[UInt64]:
        if candidate_k <= 0:
            return List[UInt64]()
        if len(encoded_query) != Self.FDE_DIM:
            raise Error("MuVERA encoded query dimension mismatch")
        var effective_ef = ef_search
        if effective_ef < candidate_k:
            effective_ef = candidate_k

        var candidates = self.hnsw.search(
            encoded_query, k=candidate_k, ef_search=effective_ef
        )
        var results = List[UInt64]()
        results.reserve(len(candidates))
        for i in range(len(candidates)):
            var ordinal = Int(candidates[i].id)
            if ordinal >= len(self.doc_ids):
                raise Error("MuVERA candidate doc ID out of range")
            results.append(self.doc_ids[ordinal])
        return results^

    @staticmethod
    def encode_vectors(
        vectors: Span[Float32, _], vector_count: Int
    ) raises -> List[Float32]:
        if Self.partition_count <= 0:
            raise Error("MuVERA partition count must be positive")
        if Self.repetition_count <= 0:
            raise Error("MuVERA repetition count must be positive")
        if vector_count <= 0:
            raise Error("MuVERA vector count must be positive")
        if len(vectors) != vector_count * Self.dim:
            raise Error("MuVERA vector dimension mismatch")

        var encoding = List[Float32]()
        encoding.reserve(Self.FDE_DIM)
        for _ in range(Self.FDE_DIM):
            encoding.append(0.0)

        for repetition_idx in range(Self.repetition_count):
            for vector_idx in range(vector_count):
                var bucket = Self._bucket_for_vector(
                    vectors, vector_idx, repetition_idx
                )
                var offset = (
                    repetition_idx * Self.partition_count + bucket
                ) * Self.dim
                for dim_idx in range(Self.dim):
                    encoding[offset + dim_idx] += vectors[
                        vector_idx * Self.dim + dim_idx
                    ]

        Self._normalize(encoding)
        return encoding^

    @staticmethod
    def _bucket_for_vector(
        vectors: Span[Float32, _], vector_idx: Int, repetition_idx: Int
    ) -> Int:
        var best_bucket = 0
        var best_score: Float32 = -3.402823e38
        for bucket in range(Self.partition_count):
            var score: Float32 = 0.0
            for dim_idx in range(Self.dim):
                score += vectors[
                    vector_idx * Self.dim + dim_idx
                ] * Self._projection_weight(dim_idx, bucket, repetition_idx)
            if score > best_score:
                best_score = score
                best_bucket = bucket
        return best_bucket

    @staticmethod
    def _projection_weight(
        dim_idx: Int, bucket: Int, repetition_idx: Int
    ) -> Float32:
        var raw = (
            (dim_idx + 17) * 1_103_515_245
            + (bucket + 31) * 12_345
            + (repetition_idx + 7) * 2_654_435_761
        ) % 2_003
        return (Float32(raw) / 1001.0) - 1.0

    @staticmethod
    def _normalize(mut encoding: List[Float32]):
        var norm_sq: Float32 = 0.0
        for i in range(len(encoding)):
            norm_sq += encoding[i] * encoding[i]
        if norm_sq <= 0.0:
            return
        var inv_norm = Float32(1.0 / sqrt(Float64(norm_sq)))
        for i in range(len(encoding)):
            encoding[i] *= inv_norm


struct MultiVectorExactStore[dim: Int](Movable):
    var external_to_doc: Dict[String, UInt64]
    var external_ids: List[String]
    var metadata: List[Optional[String]]
    var source_spans: List[Optional[String]]
    var deleted: List[Bool]
    var deleted_count: Int
    var vector_counts: List[Int]
    var record_vectors: List[List[Float32]]
    var text_index: BM25Index
    var text_doc_ids: List[UInt64]
    var text_doc_texts: List[String]
    var text_enabled: Bool
    var persistent_path: Optional[String]

    def __init__(out self):
        self.external_to_doc = Dict[String, UInt64]()
        self.external_ids = List[String]()
        self.metadata = List[Optional[String]]()
        self.source_spans = List[Optional[String]]()
        self.deleted = List[Bool]()
        self.deleted_count = 0
        self.vector_counts = List[Int]()
        self.record_vectors = List[List[Float32]]()
        self.text_index = BM25Index()
        self.text_doc_ids = List[UInt64]()
        self.text_doc_texts = List[String]()
        self.text_enabled = False
        self.persistent_path = None

    def __init__(out self, *, deinit take: Self):
        self.external_to_doc = take.external_to_doc^
        self.external_ids = take.external_ids^
        self.metadata = take.metadata^
        self.source_spans = take.source_spans^
        self.deleted = take.deleted^
        self.deleted_count = take.deleted_count
        self.vector_counts = take.vector_counts^
        self.record_vectors = take.record_vectors^
        self.text_index = take.text_index^
        self.text_doc_ids = take.text_doc_ids^
        self.text_doc_texts = take.text_doc_texts^
        self.text_enabled = take.text_enabled
        self.persistent_path = take.persistent_path^

    @staticmethod
    def create(path: String) raises -> Self:
        if exists(path) and exists(path + "/manifest.json"):
            raise Error("persistent multivector store already exists")
        makedirs(path, exist_ok=True)
        var store = Self()
        store.persistent_path = Optional[String](path)
        store.flush()
        return store^

    @staticmethod
    def open(path: String) raises -> Self:
        if not exists(path + "/manifest.meta"):
            raise Error("persistent multivector manifest not found")

        var manifest = Self._validate_manifest(path)

        var store = Self()
        store.persistent_path = Optional[String](path)
        store._load_records(path)
        if manifest.text_enabled:
            store._load_text_docs(path)
        return store^

    def is_persistent(ref self) -> Bool:
        return self.persistent_path != None

    def flush(ref self) raises:
        if not self.is_persistent():
            return
        var path = self.persistent_path.value()
        var parent = dirname(path)
        if parent == "":
            parent = "."
        makedirs(parent, exist_ok=True)
        var tmp_path = mkdtemp(
            prefix=".omendb-multivector-" + basename(path) + "-",
            dir=parent,
        )
        var backup_path = path + ".bak"
        try:
            self._write_snapshot(tmp_path)
            if exists(backup_path):
                _remove_tree(backup_path)
            if exists(path):
                _rename(path, backup_path)
            _rename(tmp_path, path)
            if exists(backup_path):
                _remove_tree(backup_path)
        except e:
            if exists(tmp_path):
                _remove_tree(tmp_path)
            if not exists(path) and exists(backup_path):
                _rename(backup_path, path)
            raise e^

    def len(ref self) -> Int:
        return len(self.external_ids) - self.deleted_count

    def enable_text_search(mut self):
        self.text_enabled = True

    def set_vectors(
        mut self,
        id: String,
        vectors: Span[Float32, _],
        vector_count: Int,
        metadata: Optional[String] = None,
        source_span: Optional[String] = None,
    ) raises -> UInt64:
        if id.byte_length() == 0:
            raise Error("id must not be empty")
        if vector_count <= 0:
            raise Error("vector count must be positive")
        if len(vectors) != vector_count * Self.dim:
            raise Error("multi-vector dimension mismatch")
        if id in self.external_to_doc:
            _ = self.delete(id)

        var doc_id = UInt64(len(self.external_ids))
        var stored = List[Float32]()
        stored.reserve(len(vectors))
        for i in range(len(vectors)):
            stored.append(vectors[i])

        self.external_to_doc[id] = doc_id
        self.external_ids.append(id)
        self.metadata.append(metadata)
        self.source_spans.append(source_span)
        self.deleted.append(False)
        self.vector_counts.append(vector_count)
        self.record_vectors.append(stored^)
        return doc_id

    def set_vectors_text(
        mut self,
        id: String,
        vectors: Span[Float32, _],
        vector_count: Int,
        text: String,
        metadata: Optional[String] = None,
        source_span: Optional[String] = None,
    ) raises -> UInt64:
        if not self.text_enabled:
            self.enable_text_search()

        var doc_id = self.set_vectors(
            id, vectors, vector_count, metadata, source_span
        )
        var text_doc_id = self.text_index.add_document(text)
        if Int(text_doc_id) != len(self.text_doc_ids):
            raise Error("text document ID mismatch")
        self.text_doc_ids.append(doc_id)
        self.text_doc_texts.append(text)
        return doc_id

    def delete(mut self, id: String) raises -> Bool:
        if id not in self.external_to_doc:
            return False
        var doc_id = self.external_to_doc.pop(id)
        var idx = Int(doc_id)
        if idx >= len(self.deleted):
            raise Error("delete doc ID out of range")
        if self.deleted[idx]:
            return False
        self.deleted[idx] = True
        self.deleted_count += 1
        return True

    def supersede(mut self, old_id: String, new_id: String) raises -> Bool:
        """Mark old_id as superseded by new_id."""
        if old_id not in self.external_to_doc:
            return False
        if new_id not in self.external_to_doc:
            raise Error("replacement item '" + new_id + "' not found")
        var doc_id = self.external_to_doc.pop(old_id)
        var idx = Int(doc_id)
        if idx >= len(self.deleted):
            raise Error("supersede doc ID out of range")
        if self.deleted[idx]:
            return False
        self.deleted[idx] = True
        self.deleted_count += 1
        return True

    def get_vectors(ref self, id: String) raises -> List[Float32]:
        if id not in self.external_to_doc:
            return List[Float32]()
        var doc_id = Int(self.external_to_doc[id])
        if self.deleted[doc_id]:
            return List[Float32]()

        var result = List[Float32]()
        result.reserve(len(self.record_vectors[doc_id]))
        ref stored = self.record_vectors[doc_id]
        for i in range(len(stored)):
            result.append(stored[i])
        return result^

    def get_vector_count(ref self, id: String) raises -> Int:
        if id not in self.external_to_doc:
            return 0
        var doc_id = Int(self.external_to_doc[id])
        if self.deleted[doc_id]:
            return 0
        return self.vector_counts[doc_id]

    def get_metadata(ref self, id: String) raises -> Optional[String]:
        if id not in self.external_to_doc:
            return None
        var doc_id = Int(self.external_to_doc[id])
        if self.deleted[doc_id]:
            return None
        return self.metadata[doc_id]

    def get_source_span(ref self, id: String) raises -> Optional[String]:
        if id not in self.external_to_doc:
            return None
        var doc_id = Int(self.external_to_doc[id])
        if self.deleted[doc_id]:
            return None
        return self.source_spans[doc_id]

    def search_vectors(
        ref self,
        query_vectors: Span[Float32, _],
        query_vector_count: Int,
        k: Int = 10,
    ) raises -> List[MultiVectorResult]:
        if query_vector_count <= 0:
            raise Error("query vector count must be positive")
        if len(query_vectors) != query_vector_count * Self.dim:
            raise Error("query vector dimension mismatch")
        if k <= 0:
            return List[MultiVectorResult]()

        var results = List[MultiVectorResult]()
        for i in range(len(self.external_ids)):
            if self.deleted[i]:
                continue
            var score = self._maxsim_score_for_doc(
                query_vectors, query_vector_count, i
            )
            results.append(
                MultiVectorResult(
                    self.external_ids[i],
                    UInt64(i),
                    -score,
                    metadata=self.metadata[i],
                    source_span=self.source_spans[i],
                )
            )

        return _top_k(results^, k)

    def search_vectors_from_doc_ids(
        ref self,
        query_vectors: Span[Float32, _],
        query_vector_count: Int,
        candidate_doc_ids: Span[UInt64, _],
        k: Int = 10,
    ) raises -> List[MultiVectorResult]:
        if query_vector_count <= 0:
            raise Error("query vector count must be positive")
        if len(query_vectors) != query_vector_count * Self.dim:
            raise Error("query vector dimension mismatch")
        if k <= 0 or len(candidate_doc_ids) == 0:
            return List[MultiVectorResult]()

        var results = List[MultiVectorResult]()
        for i in range(len(candidate_doc_ids)):
            var doc_id = Int(candidate_doc_ids[i])
            if doc_id >= len(self.external_ids):
                raise Error("candidate doc ID out of range")
            if self.deleted[doc_id]:
                continue
            var score = self._maxsim_score_for_doc(
                query_vectors, query_vector_count, doc_id
            )
            results.append(
                MultiVectorResult(
                    self.external_ids[doc_id],
                    UInt64(doc_id),
                    -score,
                    metadata=self.metadata[doc_id],
                    source_span=self.source_spans[doc_id],
                )
            )

        return _top_k(results^, k)

    def search_vectors_with_options(
        ref self,
        query_vectors: Span[Float32, _],
        query_vector_count: Int,
        k: Int = 10,
    ) raises -> List[MultiVectorResult]:
        return self.search_vectors(query_vectors, query_vector_count, k)

    def search_hybrid_vectors(
        ref self,
        text_query: String,
        query_vectors: Span[Float32, _],
        query_vector_count: Int,
        k: Int = 10,
        text_candidate_k: Int = 100,
    ) raises -> List[MultiVectorResult]:
        if not self.text_enabled:
            raise Error("text search is not enabled")
        if query_vector_count <= 0:
            raise Error("query vector count must be positive")
        if len(query_vectors) != query_vector_count * Self.dim:
            raise Error("query vector dimension mismatch")
        if k <= 0:
            return List[MultiVectorResult]()

        var text_results = self.text_index.search(text_query, text_candidate_k)
        var results = List[MultiVectorResult]()
        for i in range(len(text_results)):
            var text_doc_id = Int(text_results[i][0])
            if text_doc_id >= len(self.text_doc_ids):
                raise Error("text document mapping missing")
            var doc_id = Int(self.text_doc_ids[text_doc_id])
            if self.deleted[doc_id]:
                continue
            var score = self._maxsim_score_for_doc(
                query_vectors, query_vector_count, doc_id
            )
            results.append(
                MultiVectorResult(
                    self.external_ids[doc_id],
                    UInt64(doc_id),
                    -score,
                    metadata=self.metadata[doc_id],
                    source_span=self.source_spans[doc_id],
                )
            )

        return _top_k(results^, k)

    def _maxsim_score_for_doc(
        ref self,
        query_vectors: Span[Float32, _],
        query_vector_count: Int,
        doc_id: Int,
    ) -> Float32:
        var total: Float32 = 0.0
        var doc_vector_count = self.vector_counts[doc_id]
        ref doc_vectors = self.record_vectors[doc_id]
        for q in range(query_vector_count):
            var best: Float32 = -3.402823e38
            for d in range(doc_vector_count):
                var dot: Float32 = 0.0
                for j in range(Self.dim):
                    dot += (
                        query_vectors[q * Self.dim + j]
                        * doc_vectors[d * Self.dim + j]
                    )
                if dot > best:
                    best = dot
            total += best
        return total

    def _write_snapshot(ref self, path: String) raises:
        makedirs(path, exist_ok=True)
        self._write_records(path)
        self._write_source_spans(path)
        self._write_vectors(path)
        if self.text_enabled:
            self._write_text_docs(path)
        self._write_manifest(path, ENCODING_NONE)

    def _write_manifest(ref self, path: String, encoding_mode: Int) raises:
        var meta = List[Int]()
        meta.append(MANIFEST_FORMAT_VERSION)
        meta.append(Self.dim)
        meta.append(len(self.external_ids))
        meta.append(self.deleted_count)
        meta.append(1 if self.text_enabled else 0)
        meta.append(encoding_mode)
        save_list_int(path + "/manifest.meta", meta)

        var json = String("{")
        json += '"format_version":' + String(MANIFEST_FORMAT_VERSION)
        json += ',"store_layout":"multivector_exact_v1"'
        json += ',"index_mode":"exact_maxsim"'
        json += ',"dim":' + String(Self.dim)
        json += ',"record_count":' + String(len(self.external_ids))
        json += ',"tombstone_count":' + String(self.deleted_count)
        json += ',"text_enabled":' + _bool_json(self.text_enabled)
        json += ',"source_spans":true'
        json += ',"encoding_mode":"' + _encoding_json(encoding_mode) + '"'
        if encoding_mode == ENCODING_MUVERA:
            json += ',"candidate_index_mode":"fde_hnsw_rebuild_v0"'
            json += ',"candidate_partitions":2'
            json += ',"candidate_repetitions":2'
            json += ',"candidate_k":500'
            json += ',"ef_search":100'
        json += "}"
        with open(path + "/manifest.json", "w") as manifest_file:
            manifest_file.write(json)

    def _write_records(ref self, path: String) raises:
        var meta = List[Int]()
        var strings = List[UInt8]()
        var vector_offset = 0
        for i in range(len(self.external_ids)):
            var id_span = _append_string_bytes(strings, self.external_ids[i])
            var metadata_offset = -1
            var metadata_len = -1
            if self.metadata[i]:
                var metadata_span = _append_string_bytes(
                    strings, self.metadata[i].value()
                )
                metadata_offset = metadata_span[0]
                metadata_len = metadata_span[1]

            meta.append(i)
            meta.append(self.vector_counts[i])
            meta.append(vector_offset)
            meta.append(1 if self.deleted[i] else 0)
            meta.append(id_span[0])
            meta.append(id_span[1])
            meta.append(metadata_offset)
            meta.append(metadata_len)
            vector_offset += len(self.record_vectors[i])

        save_list_int(path + "/records.meta", meta)
        save_list_u8(path + "/records.strings", strings)

    def _write_source_spans(ref self, path: String) raises:
        var meta = List[Int]()
        var strings = List[UInt8]()
        for i in range(len(self.source_spans)):
            if self.source_spans[i]:
                var source_span = _append_string_bytes(
                    strings, self.source_spans[i].value()
                )
                meta.append(source_span[0])
                meta.append(source_span[1])
            else:
                meta.append(-1)
                meta.append(-1)

        save_list_int(path + "/source_spans.meta", meta)
        save_list_u8(path + "/source_spans.strings", strings)

    def _write_vectors(ref self, path: String) raises:
        var total = 0
        for i in range(len(self.record_vectors)):
            total += len(self.record_vectors[i])

        var vectors = List[Float32]()
        vectors.reserve(total)
        for i in range(len(self.record_vectors)):
            ref doc_vectors = self.record_vectors[i]
            for j in range(len(doc_vectors)):
                vectors.append(doc_vectors[j])

        save_list_f32(path + "/vectors.f32", vectors)

    def _write_text_docs(ref self, path: String) raises:
        var meta = List[Int]()
        var strings = List[UInt8]()
        for i in range(len(self.text_doc_ids)):
            var text_span = _append_string_bytes(
                strings, self.text_doc_texts[i]
            )
            meta.append(Int(self.text_doc_ids[i]))
            meta.append(text_span[0])
            meta.append(text_span[1])

        save_list_int(path + "/text_docs.meta", meta)
        save_list_u8(path + "/text_docs.strings", strings)

    def _load_records(mut self, path: String) raises:
        if not exists(path + "/records.meta"):
            raise Error("persistent multivector records not found")

        self.external_to_doc = Dict[String, UInt64]()
        self.external_ids = List[String]()
        self.metadata = List[Optional[String]]()
        self.source_spans = List[Optional[String]]()
        self.deleted = List[Bool]()
        self.deleted_count = 0
        self.vector_counts = List[Int]()
        self.record_vectors = List[List[Float32]]()
        var record_meta = load_list_int(path + "/records.meta")
        var record_strings = load_list_u8(path + "/records.strings")
        if len(record_meta) % RECORD_META_WIDTH != 0:
            raise Error("persistent multivector record metadata mismatch")
        var all_vectors = load_list_f32(path + "/vectors.f32")
        var expected_offset = 0
        var record_count = len(record_meta) // RECORD_META_WIDTH

        for i in range(record_count):
            var base = i * RECORD_META_WIDTH
            var doc_id = record_meta[base]
            if doc_id != i:
                raise Error("persistent multivector doc ID mismatch")
            var vector_count = record_meta[base + 1]
            var vector_offset = record_meta[base + 2]
            var deleted_flag = record_meta[base + 3]
            var id = _string_from_bytes(
                record_strings, record_meta[base + 4], record_meta[base + 5]
            )
            var metadata_offset = record_meta[base + 6]
            var metadata_len = record_meta[base + 7]
            self.external_ids.append(id)
            if metadata_offset >= 0:
                self.metadata.append(
                    Optional[String](
                        _string_from_bytes(
                            record_strings, metadata_offset, metadata_len
                        )
                    )
                )
            else:
                self.metadata.append(Optional[String](None))
            self.source_spans.append(Optional[String](None))
            var vector_len = vector_count * Self.dim
            if vector_count <= 0:
                raise Error("persistent multivector count mismatch")
            if vector_offset != expected_offset:
                raise Error("persistent multivector offset mismatch")
            if vector_offset + vector_len > len(all_vectors):
                raise Error("persistent multivector dimension mismatch")
            var vectors = List[Float32]()
            vectors.reserve(vector_len)
            for j in range(vector_len):
                vectors.append(all_vectors[vector_offset + j])
            expected_offset += vector_len
            self.vector_counts.append(vector_count)
            self.record_vectors.append(vectors^)
            if deleted_flag == 1:
                self.deleted.append(True)
                self.deleted_count += 1
            else:
                self.deleted.append(False)
                self.external_to_doc[id] = UInt64(i)

        if expected_offset != len(all_vectors):
            raise Error("persistent multivector length mismatch")
        if exists(path + "/source_spans.meta"):
            self._load_source_spans(path)

    def _load_source_spans(mut self, path: String) raises:
        if not exists(path + "/source_spans.strings"):
            raise Error("persistent multivector source span strings not found")
        var source_meta = load_list_int(path + "/source_spans.meta")
        var source_strings = load_list_u8(path + "/source_spans.strings")
        if len(source_meta) % SOURCE_META_WIDTH != 0:
            raise Error("persistent multivector source span metadata mismatch")
        if len(source_meta) // SOURCE_META_WIDTH != len(self.external_ids):
            raise Error("persistent multivector source span count mismatch")

        self.source_spans = List[Optional[String]]()
        for i in range(len(source_meta) // SOURCE_META_WIDTH):
            var base = i * SOURCE_META_WIDTH
            var source_offset = source_meta[base]
            var source_len = source_meta[base + 1]
            if source_offset >= 0:
                self.source_spans.append(
                    Optional[String](
                        _string_from_bytes(
                            source_strings, source_offset, source_len
                        )
                    )
                )
            else:
                self.source_spans.append(Optional[String](None))

    def _load_text_docs(mut self, path: String) raises:
        if not exists(path + "/text_docs.meta"):
            raise Error("persistent multivector text docs not found")

        self.text_enabled = True
        self.text_index = BM25Index()
        self.text_doc_ids = List[UInt64]()
        self.text_doc_texts = List[String]()
        var text_meta = load_list_int(path + "/text_docs.meta")
        var text_strings = load_list_u8(path + "/text_docs.strings")
        if len(text_meta) % TEXT_META_WIDTH != 0:
            raise Error("persistent multivector text metadata mismatch")

        for i in range(len(text_meta) // TEXT_META_WIDTH):
            var base = i * TEXT_META_WIDTH
            var doc_id = UInt64(text_meta[base])
            if Int(doc_id) >= len(self.external_ids):
                raise Error("persistent multivector text doc ID out of range")
            var text = _string_from_bytes(
                text_strings, text_meta[base + 1], text_meta[base + 2]
            )
            var text_doc_id = self.text_index.add_document(text)
            if Int(text_doc_id) != i:
                raise Error("persistent multivector text doc mapping mismatch")
            self.text_doc_ids.append(doc_id)
            self.text_doc_texts.append(text)

    @staticmethod
    def _validate_manifest(path: String) raises -> MultiVectorManifest:
        var meta = load_list_int(path + "/manifest.meta")
        if len(meta) < 6:
            raise Error("persistent multivector manifest mismatch")
        var manifest = MultiVectorManifest(
            meta[0], meta[1], meta[2], meta[3], meta[4] == 1, meta[5]
        )
        if manifest.format_version != MANIFEST_FORMAT_VERSION:
            raise Error("persistent multivector format version mismatch")
        if manifest.dim != Self.dim:
            raise Error("persistent multivector dimension mismatch")
        if manifest.record_count < 0:
            raise Error("persistent multivector record count mismatch")
        if manifest.tombstone_count < 0:
            raise Error("persistent multivector tombstone count mismatch")
        if not exists(path + "/records.meta"):
            raise Error("persistent multivector records not found")
        if not exists(path + "/records.strings"):
            raise Error("persistent multivector record strings not found")
        if not exists(path + "/vectors.f32"):
            raise Error("persistent multivector vector data not found")
        if manifest.text_enabled:
            if not exists(path + "/text_docs.meta"):
                raise Error("persistent multivector text docs not found")
            if not exists(path + "/text_docs.strings"):
                raise Error("persistent multivector text strings not found")
        return manifest


struct MultiVectorMuveraStore[dim: Int](Movable):
    comptime PARTITION_COUNT = 2
    comptime REPETITION_COUNT = 2
    comptime CANDIDATE_K = 500
    comptime SMALL_EXACT_FALLBACK_COUNT = 3000
    comptime EF_SEARCH = 100
    comptime M = 16
    comptime EF_CONSTRUCTION = 100

    var exact: MultiVectorExactStore[Self.dim]
    var candidate_index: MultiVectorFDEIndex[
        Self.dim, Self.PARTITION_COUNT, Self.REPETITION_COUNT
    ]

    def __init__(out self):
        self.exact = MultiVectorExactStore[Self.dim]()
        self.candidate_index = MultiVectorFDEIndex[
            Self.dim, Self.PARTITION_COUNT, Self.REPETITION_COUNT
        ](M=Self.M, ef_construction=Self.EF_CONSTRUCTION)

    def __init__(out self, *, deinit take: Self):
        self.exact = take.exact^
        self.candidate_index = take.candidate_index^

    def __init__(out self, var exact: MultiVectorExactStore[Self.dim]) raises:
        self.exact = exact^
        self.candidate_index = MultiVectorFDEIndex[
            Self.dim, Self.PARTITION_COUNT, Self.REPETITION_COUNT
        ](M=Self.M, ef_construction=Self.EF_CONSTRUCTION)
        self.rebuild_candidate_index()

    @staticmethod
    def create(path: String) raises -> Self:
        var store = Self(MultiVectorExactStore[Self.dim].create(path))
        store._write_manifest_encoding_marker()
        return store^

    @staticmethod
    def open(path: String) raises -> Self:
        return Self(MultiVectorExactStore[Self.dim].open(path))

    def len(ref self) -> Int:
        return self.exact.len()

    def enable_text_search(mut self):
        self.exact.enable_text_search()

    def flush(ref self) raises:
        self.exact.flush()
        self._write_manifest_encoding_marker()

    def set_vectors(
        mut self,
        id: String,
        vectors: Span[Float32, _],
        vector_count: Int,
        metadata: Optional[String] = None,
        source_span: Optional[String] = None,
    ) raises -> UInt64:
        var doc_id = self.exact.set_vectors(
            id,
            vectors,
            vector_count,
            metadata=metadata,
            source_span=source_span,
        )
        self.candidate_index.insert(doc_id, vectors, vector_count)
        return doc_id

    def set_vectors_text(
        mut self,
        id: String,
        vectors: Span[Float32, _],
        vector_count: Int,
        text: String,
        metadata: Optional[String] = None,
        source_span: Optional[String] = None,
    ) raises -> UInt64:
        var doc_id = self.exact.set_vectors_text(
            id,
            vectors,
            vector_count,
            text,
            metadata=metadata,
            source_span=source_span,
        )
        self.candidate_index.insert(doc_id, vectors, vector_count)
        return doc_id

    def delete(mut self, id: String) raises -> Bool:
        return self.exact.delete(id)

    def supersede(mut self, old_id: String, new_id: String) raises -> Bool:
        return self.exact.supersede(old_id, new_id)

    def get_vectors(ref self, id: String) raises -> List[Float32]:
        return self.exact.get_vectors(id)

    def get_vector_count(ref self, id: String) raises -> Int:
        return self.exact.get_vector_count(id)

    def get_metadata(ref self, id: String) raises -> Optional[String]:
        return self.exact.get_metadata(id)

    def get_source_span(ref self, id: String) raises -> Optional[String]:
        return self.exact.get_source_span(id)

    def search_vectors_exact(
        ref self,
        query_vectors: Span[Float32, _],
        query_vector_count: Int,
        k: Int = 10,
    ) raises -> List[MultiVectorResult]:
        return self.exact.search_vectors(query_vectors, query_vector_count, k)

    def search_vectors(
        ref self,
        query_vectors: Span[Float32, _],
        query_vector_count: Int,
        k: Int = 10,
    ) raises -> List[MultiVectorResult]:
        if query_vector_count <= 0:
            raise Error("query vector count must be positive")
        if len(query_vectors) != query_vector_count * Self.dim:
            raise Error("query vector dimension mismatch")
        if k <= 0:
            return List[MultiVectorResult]()
        var live_count = self.exact.len()
        if (
            live_count <= Self.SMALL_EXACT_FALLBACK_COUNT
            or self.candidate_index.len() == 0
        ):
            return self.exact.search_vectors(
                query_vectors, query_vector_count, k
            )

        var candidate_k = Self.CANDIDATE_K
        if candidate_k < k:
            candidate_k = k
        var candidate_doc_ids = self.candidate_index.search(
            query_vectors,
            query_vector_count,
            candidate_k=candidate_k,
            ef_search=Self.EF_SEARCH,
        )
        if len(candidate_doc_ids) < min(k, live_count):
            return self.exact.search_vectors(
                query_vectors, query_vector_count, k
            )

        var reranked = self.exact.search_vectors_from_doc_ids(
            query_vectors,
            query_vector_count,
            Span(
                ptr=candidate_doc_ids.unsafe_ptr(),
                length=len(candidate_doc_ids),
            ),
            k=k,
        )
        if len(reranked) < min(k, live_count):
            return self.exact.search_vectors(
                query_vectors, query_vector_count, k
            )
        return reranked^

    def search_vectors_with_options(
        ref self,
        query_vectors: Span[Float32, _],
        query_vector_count: Int,
        k: Int = 10,
    ) raises -> List[MultiVectorResult]:
        return self.search_vectors(query_vectors, query_vector_count, k)

    def search_hybrid_vectors(
        ref self,
        text_query: String,
        query_vectors: Span[Float32, _],
        query_vector_count: Int,
        k: Int = 10,
        text_candidate_k: Int = 100,
    ) raises -> List[MultiVectorResult]:
        return self.exact.search_hybrid_vectors(
            text_query,
            query_vectors,
            query_vector_count,
            k=k,
            text_candidate_k=text_candidate_k,
        )

    def rebuild_candidate_index(mut self) raises:
        self.candidate_index = MultiVectorFDEIndex[
            Self.dim, Self.PARTITION_COUNT, Self.REPETITION_COUNT
        ](M=Self.M, ef_construction=Self.EF_CONSTRUCTION)
        self.candidate_index.reserve(self.exact.len())
        for doc_id in range(len(self.exact.external_ids)):
            if self.exact.deleted[doc_id]:
                continue
            ref vectors = self.exact.record_vectors[doc_id]
            self.candidate_index.insert(
                UInt64(doc_id),
                Span(ptr=vectors.unsafe_ptr(), length=len(vectors)),
                self.exact.vector_counts[doc_id],
            )

    def _write_manifest_encoding_marker(ref self) raises:
        if not self.exact.is_persistent():
            return
        var path = self.exact.persistent_path.value() + ""
        self.exact._write_manifest(path, ENCODING_MUVERA)


def _top_k(
    var results: List[MultiVectorResult], k: Int
) -> List[MultiVectorResult]:
    @parameter
    def cmp(a: MultiVectorResult, b: MultiVectorResult) -> Bool:
        return a.distance < b.distance

    var span = Span[MultiVectorResult, MutAnyOrigin](
        ptr=results.unsafe_ptr(), length=len(results)
    )
    sort[cmp](span)

    var top = List[MultiVectorResult]()
    for i in range(min(k, len(results))):
        top.append(results[i].copy())
    return top^
