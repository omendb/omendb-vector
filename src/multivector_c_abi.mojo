from multivector import MultiVectorExactStore, MultiVectorResult
from std.ffi import CStringSlice
from std.memory import OptionalUnsafePointer, alloc


struct OmenDBVectorResults(Movable):
    var ids: List[String]
    var distances: List[Float32]
    var metadata: List[String]

    def __init__(out self, var results: List[MultiVectorResult]):
        self.ids = List[String]()
        self.distances = List[Float32]()
        self.metadata = List[String]()
        self.ids.reserve(len(results))
        self.distances.reserve(len(results))
        self.metadata.reserve(len(results))
        for i in range(len(results)):
            self.ids.append(results[i].id)
            self.distances.append(results[i].distance)
            if results[i].metadata:
                self.metadata.append(results[i].metadata.value())
            else:
                self.metadata.append("{}")


struct OmenDBVectorBuffer(Movable):
    var vectors: List[Float32]
    var vector_count: Int
    var metadata: String

    def __init__(
        out self,
        var vectors: List[Float32],
        vector_count: Int,
        metadata: String,
    ):
        self.vectors = vectors^
        self.vector_count = vector_count
        self.metadata = metadata


def _copy_string(
    value: String,
    dest: OptionalUnsafePointer[UInt8, MutAnyOrigin],
    capacity: Int64,
) -> Int64:
    var length = value.byte_length()
    if not dest:
        return Int64(length)
    if capacity <= 0:
        return Int64(length)

    var out = dest.value()
    var src = value.unsafe_ptr()
    var n = min(length, Int(capacity))
    for i in range(n):
        out[i] = UInt8(src[i])
    return Int64(length)


def _store_2_from_handle(
    handle: Int64,
) raises -> UnsafePointer[MultiVectorExactStore[2], MutExternalOrigin]:
    if handle == 0:
        raise Error("null multivector handle")
    return UnsafePointer[MultiVectorExactStore[2], MutExternalOrigin](
        unsafe_from_address=Int(handle)
    )


def _store_48_from_handle(
    handle: Int64,
) raises -> UnsafePointer[MultiVectorExactStore[48], MutExternalOrigin]:
    if handle == 0:
        raise Error("null multivector handle")
    return UnsafePointer[MultiVectorExactStore[48], MutExternalOrigin](
        unsafe_from_address=Int(handle)
    )


def _results_from_handle(
    handle: Int64,
) raises -> UnsafePointer[OmenDBVectorResults, MutExternalOrigin]:
    if handle == 0:
        raise Error("null multivector results handle")
    return UnsafePointer[OmenDBVectorResults, MutExternalOrigin](
        unsafe_from_address=Int(handle)
    )


def _vectors_from_handle(
    handle: Int64,
) raises -> UnsafePointer[OmenDBVectorBuffer, MutExternalOrigin]:
    if handle == 0:
        raise Error("null multivector vectors handle")
    return UnsafePointer[OmenDBVectorBuffer, MutExternalOrigin](
        unsafe_from_address=Int(handle)
    )


def _alloc_store_2(
    var store: MultiVectorExactStore[2],
) -> Int64:
    var ptr = alloc[MultiVectorExactStore[2]](1)
    ptr.init_pointee_move(store^)
    return Int64(UnsafePointer(to=ptr).bitcast[Int]()[])


def _alloc_store_48(
    var store: MultiVectorExactStore[48],
) -> Int64:
    var ptr = alloc[MultiVectorExactStore[48]](1)
    ptr.init_pointee_move(store^)
    return Int64(UnsafePointer(to=ptr).bitcast[Int]()[])


def _alloc_results(
    var results: List[MultiVectorResult],
) -> Int64:
    var ptr = alloc[OmenDBVectorResults](1)
    ptr.init_pointee_move(OmenDBVectorResults(results^))
    return Int64(UnsafePointer(to=ptr).bitcast[Int]()[])


def _alloc_vector_buffer(
    var vectors: List[Float32], vector_count: Int, metadata: String
) -> Int64:
    var ptr = alloc[OmenDBVectorBuffer](1)
    ptr.init_pointee_move(OmenDBVectorBuffer(vectors^, vector_count, metadata))
    return Int64(UnsafePointer(to=ptr).bitcast[Int]()[])


@export("omendb_vector_mv2_create", ABI="C")
def omendb_vector_mv2_create(
    path: CStringSlice[ImmutAnyOrigin],
) -> Int64:
    try:
        return _alloc_store_2(MultiVectorExactStore[2].create(String(path)))
    except:
        return 0


@export("omendb_vector_mv2_open", ABI="C")
def omendb_vector_mv2_open(
    path: CStringSlice[ImmutAnyOrigin],
) -> Int64:
    try:
        return _alloc_store_2(MultiVectorExactStore[2].open(String(path)))
    except:
        return 0


@export("omendb_vector_mv48_create", ABI="C")
def omendb_vector_mv48_create(
    path: CStringSlice[ImmutAnyOrigin],
) -> Int64:
    try:
        return _alloc_store_48(MultiVectorExactStore[48].create(String(path)))
    except:
        return 0


@export("omendb_vector_mv48_open", ABI="C")
def omendb_vector_mv48_open(
    path: CStringSlice[ImmutAnyOrigin],
) -> Int64:
    try:
        return _alloc_store_48(MultiVectorExactStore[48].open(String(path)))
    except:
        return 0


@export("omendb_vector_mv2_free", ABI="C")
def omendb_vector_mv2_free(handle: Int64):
    if handle != 0:
        try:
            var ptr = _store_2_from_handle(handle)
            ptr.destroy_pointee()
            ptr.free()
        except:
            pass


@export("omendb_vector_mv48_free", ABI="C")
def omendb_vector_mv48_free(handle: Int64):
    if handle != 0:
        try:
            var ptr = _store_48_from_handle(handle)
            ptr.destroy_pointee()
            ptr.free()
        except:
            pass


@export("omendb_vector_mv2_len", ABI="C")
def omendb_vector_mv2_len(handle: Int64) -> Int64:
    try:
        return Int64(_store_2_from_handle(handle)[].len())
    except:
        return -1


@export("omendb_vector_mv48_len", ABI="C")
def omendb_vector_mv48_len(handle: Int64) -> Int64:
    try:
        return Int64(_store_48_from_handle(handle)[].len())
    except:
        return -1


@export("omendb_vector_mv2_flush", ABI="C")
def omendb_vector_mv2_flush(handle: Int64) -> Int32:
    try:
        _store_2_from_handle(handle)[].flush()
        return 0
    except:
        return -1


@export("omendb_vector_mv48_flush", ABI="C")
def omendb_vector_mv48_flush(handle: Int64) -> Int32:
    try:
        _store_48_from_handle(handle)[].flush()
        return 0
    except:
        return -1


@export("omendb_vector_mv2_delete", ABI="C")
def omendb_vector_mv2_delete(
    handle: Int64,
    id: CStringSlice[ImmutAnyOrigin],
) -> Int32:
    try:
        if _store_2_from_handle(handle)[].delete(String(id)):
            return 1
        return 0
    except:
        return -1


@export("omendb_vector_mv48_delete", ABI="C")
def omendb_vector_mv48_delete(
    handle: Int64,
    id: CStringSlice[ImmutAnyOrigin],
) -> Int32:
    try:
        if _store_48_from_handle(handle)[].delete(String(id)):
            return 1
        return 0
    except:
        return -1


@export("omendb_vector_mv2_set_vectors_text", ABI="C")
def omendb_vector_mv2_set_vectors_text(
    handle: Int64,
    id: CStringSlice[ImmutAnyOrigin],
    vectors: UnsafePointer[Float32, ImmutAnyOrigin],
    vector_count: Int64,
    text: CStringSlice[ImmutAnyOrigin],
    metadata: CStringSlice[ImmutAnyOrigin],
) -> Int32:
    if vector_count <= 0:
        return -1
    try:
        _ = _store_2_from_handle(handle)[].set_vectors_text(
            String(id),
            Span(ptr=vectors, length=Int(vector_count) * 2),
            Int(vector_count),
            String(text),
            metadata=String(metadata),
        )
        return 0
    except:
        return -1


@export("omendb_vector_mv48_set_vectors_text", ABI="C")
def omendb_vector_mv48_set_vectors_text(
    handle: Int64,
    id: CStringSlice[ImmutAnyOrigin],
    vectors: UnsafePointer[Float32, ImmutAnyOrigin],
    vector_count: Int64,
    text: CStringSlice[ImmutAnyOrigin],
    metadata: CStringSlice[ImmutAnyOrigin],
) -> Int32:
    if vector_count <= 0:
        return -1
    try:
        _ = _store_48_from_handle(handle)[].set_vectors_text(
            String(id),
            Span(ptr=vectors, length=Int(vector_count) * 48),
            Int(vector_count),
            String(text),
            metadata=String(metadata),
        )
        return 0
    except:
        return -1


@export("omendb_vector_mv2_search_vectors", ABI="C")
def omendb_vector_mv2_search_vectors(
    handle: Int64,
    vectors: UnsafePointer[Float32, ImmutAnyOrigin],
    vector_count: Int64,
    k: Int64,
) -> Int64:
    if vector_count <= 0:
        return 0
    try:
        return _alloc_results(
            _store_2_from_handle(handle)[].search_vectors_with_options(
                Span(ptr=vectors, length=Int(vector_count) * 2),
                Int(vector_count),
                k=Int(k),
            )
        )
    except:
        return 0


@export("omendb_vector_mv48_search_vectors", ABI="C")
def omendb_vector_mv48_search_vectors(
    handle: Int64,
    vectors: UnsafePointer[Float32, ImmutAnyOrigin],
    vector_count: Int64,
    k: Int64,
) -> Int64:
    if vector_count <= 0:
        return 0
    try:
        return _alloc_results(
            _store_48_from_handle(handle)[].search_vectors_with_options(
                Span(ptr=vectors, length=Int(vector_count) * 48),
                Int(vector_count),
                k=Int(k),
            )
        )
    except:
        return 0


@export("omendb_vector_mv2_search_hybrid_vectors", ABI="C")
def omendb_vector_mv2_search_hybrid_vectors(
    handle: Int64,
    text: CStringSlice[ImmutAnyOrigin],
    vectors: UnsafePointer[Float32, ImmutAnyOrigin],
    vector_count: Int64,
    k: Int64,
    text_candidate_k: Int64,
) -> Int64:
    if vector_count <= 0:
        return 0
    try:
        return _alloc_results(
            _store_2_from_handle(handle)[].search_hybrid_vectors(
                String(text),
                Span(ptr=vectors, length=Int(vector_count) * 2),
                Int(vector_count),
                k=Int(k),
                text_candidate_k=Int(text_candidate_k),
            )
        )
    except:
        return 0


@export("omendb_vector_mv48_search_hybrid_vectors", ABI="C")
def omendb_vector_mv48_search_hybrid_vectors(
    handle: Int64,
    text: CStringSlice[ImmutAnyOrigin],
    vectors: UnsafePointer[Float32, ImmutAnyOrigin],
    vector_count: Int64,
    k: Int64,
    text_candidate_k: Int64,
) -> Int64:
    if vector_count <= 0:
        return 0
    try:
        return _alloc_results(
            _store_48_from_handle(handle)[].search_hybrid_vectors(
                String(text),
                Span(ptr=vectors, length=Int(vector_count) * 48),
                Int(vector_count),
                k=Int(k),
                text_candidate_k=Int(text_candidate_k),
            )
        )
    except:
        return 0


@export("omendb_vector_mv2_get_vectors", ABI="C")
def omendb_vector_mv2_get_vectors(
    handle: Int64,
    id: CStringSlice[ImmutAnyOrigin],
) -> Int64:
    try:
        var owned_id = String(id)
        var store = _store_2_from_handle(handle)
        var vector_count = store[].get_vector_count(owned_id)
        if vector_count == 0:
            return 0
        var vectors = store[].get_vectors(owned_id)
        var metadata = store[].get_metadata(owned_id)
        if metadata:
            return _alloc_vector_buffer(
                vectors^, vector_count, metadata.value()
            )
        return _alloc_vector_buffer(vectors^, vector_count, "{}")
    except:
        return 0


@export("omendb_vector_mv48_get_vectors", ABI="C")
def omendb_vector_mv48_get_vectors(
    handle: Int64,
    id: CStringSlice[ImmutAnyOrigin],
) -> Int64:
    try:
        var owned_id = String(id)
        var store = _store_48_from_handle(handle)
        var vector_count = store[].get_vector_count(owned_id)
        if vector_count == 0:
            return 0
        var vectors = store[].get_vectors(owned_id)
        var metadata = store[].get_metadata(owned_id)
        if metadata:
            return _alloc_vector_buffer(
                vectors^, vector_count, metadata.value()
            )
        return _alloc_vector_buffer(vectors^, vector_count, "{}")
    except:
        return 0


@export("omendb_vector_mv_results_free", ABI="C")
def omendb_vector_mv_results_free(results: Int64):
    if results != 0:
        try:
            var ptr = _results_from_handle(results)
            ptr.destroy_pointee()
            ptr.free()
        except:
            pass


@export("omendb_vector_mv_results_len", ABI="C")
def omendb_vector_mv_results_len(results: Int64) -> Int64:
    try:
        return Int64(len(_results_from_handle(results)[].ids))
    except:
        return -1


@export("omendb_vector_mv_result_distance", ABI="C")
def omendb_vector_mv_result_distance(results: Int64, index: Int64) -> Float32:
    try:
        var ptr = _results_from_handle(results)
        if index < 0 or index >= Int64(len(ptr[].distances)):
            return 0.0
        return ptr[].distances[Int(index)]
    except:
        return 0.0


@export("omendb_vector_mv_result_id_copy", ABI="C")
def omendb_vector_mv_result_id_copy(
    results: Int64,
    index: Int64,
    dest: OptionalUnsafePointer[UInt8, MutAnyOrigin],
    capacity: Int64,
) -> Int64:
    try:
        var ptr = _results_from_handle(results)
        if index < 0 or index >= Int64(len(ptr[].ids)):
            return -1
        return _copy_string(ptr[].ids[Int(index)], dest, capacity)
    except:
        return -1


@export("omendb_vector_mv_result_metadata_copy", ABI="C")
def omendb_vector_mv_result_metadata_copy(
    results: Int64,
    index: Int64,
    dest: OptionalUnsafePointer[UInt8, MutAnyOrigin],
    capacity: Int64,
) -> Int64:
    try:
        var ptr = _results_from_handle(results)
        if index < 0 or index >= Int64(len(ptr[].metadata)):
            return -1
        return _copy_string(ptr[].metadata[Int(index)], dest, capacity)
    except:
        return -1


@export("omendb_vector_mv_vectors_free", ABI="C")
def omendb_vector_mv_vectors_free(vectors: Int64):
    if vectors != 0:
        try:
            var ptr = _vectors_from_handle(vectors)
            ptr.destroy_pointee()
            ptr.free()
        except:
            pass


@export("omendb_vector_mv_vectors_count", ABI="C")
def omendb_vector_mv_vectors_count(vectors: Int64) -> Int64:
    try:
        return Int64(_vectors_from_handle(vectors)[].vector_count)
    except:
        return -1


@export("omendb_vector_mv_vectors_float_len", ABI="C")
def omendb_vector_mv_vectors_float_len(vectors: Int64) -> Int64:
    try:
        return Int64(len(_vectors_from_handle(vectors)[].vectors))
    except:
        return -1


@export("omendb_vector_mv_vectors_copy", ABI="C")
def omendb_vector_mv_vectors_copy(
    vectors: Int64,
    dest: OptionalUnsafePointer[Float32, MutAnyOrigin],
    capacity: Int64,
) -> Int64:
    try:
        var ptr = _vectors_from_handle(vectors)
        var length = len(ptr[].vectors)
        if not dest:
            return Int64(length)
        if capacity <= 0:
            return Int64(length)
        var out = dest.value()
        var n = min(length, Int(capacity))
        for i in range(n):
            out[i] = ptr[].vectors[i]
        return Int64(length)
    except:
        return -1


@export("omendb_vector_mv_vectors_metadata_copy", ABI="C")
def omendb_vector_mv_vectors_metadata_copy(
    vectors: Int64,
    dest: OptionalUnsafePointer[UInt8, MutAnyOrigin],
    capacity: Int64,
) -> Int64:
    try:
        return _copy_string(
            _vectors_from_handle(vectors)[].metadata, dest, capacity
        )
    except:
        return -1
