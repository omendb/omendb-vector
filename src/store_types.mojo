"""
Type definitions for VectorStore.
"""
from std.python import Python, PythonObject


def _bool_flag(value: Bool) -> String:
    if value:
        return "1"
    return "0"


def _config_fingerprint(
    dim: Int,
    metric: String,
    M: Int,
    ef_construction: Int,
    alpha: Float32,
    text_enabled: Bool,
    graph_enabled: Bool,
) -> String:
    return (
        "single_segment_v1|dim="
        + String(dim)
        + "|metric="
        + metric
        + "|M="
        + String(M)
        + "|efc="
        + String(ef_construction)
        + "|alpha="
        + String(alpha)
        + "|text="
        + _bool_flag(text_enabled)
        + "|graph="
        + _bool_flag(graph_enabled)
    )


def _sha256_file(path: String) raises -> String:
    var hashlib = Python.import_module("hashlib")
    var builtins = Python.import_module("builtins")
    var digest = hashlib.sha256()
    var f = builtins.open(path, "rb")
    digest.update(f.read())
    _ = f.close()
    return String(digest.hexdigest())


def _snapshot_checksums(path: String) raises -> PythonObject:
    var os = Python.import_module("os")
    var builtins = Python.import_module("builtins")
    var checksums = Python.dict()
    var names = builtins.sorted(os.listdir(path))
    for i in range(Int(py=builtins.len(names))):
        var name = String(names[i])
        if name == "manifest.json":
            continue
        var file_path = path + "/" + name
        if Bool(os.path.isfile(file_path)):
            checksums[name] = _sha256_file(file_path)
    return checksums


def _validate_snapshot_checksums(path: String, manifest: PythonObject) raises:
    var builtins = Python.import_module("builtins")
    var checksums = manifest["checksums"]
    var names = builtins.sorted(checksums.keys())
    for i in range(Int(py=builtins.len(names))):
        var name = String(names[i])
        var expected = String(checksums[name])
        var actual = _sha256_file(path + "/" + name)
        if actual != expected:
            raise Error(
                "store integrity error: data checksum does not match manifest"
            )


@fieldwise_init
struct Metric(Equatable, ImplicitlyCopyable):
    var _value: Int

    comptime L2 = Metric(_value=0)
    comptime COSINE = Metric(_value=1)
    comptime DOT = Metric(_value=2)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)


struct HNSWParams(ImplicitlyCopyable):
    var M: Int
    var ef_construction: Int
    var ef_search: Int
    var alpha: Float32

    def __init__(
        out self,
        M: Int = 16,
        ef_construction: Int = 100,
        ef_search: Int = 100,
        alpha: Float32 = 1.0,
    ):
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.alpha = alpha


struct SearchOptions(ImplicitlyCopyable):
    var k: Int
    var ef_search: Int
    var max_distance: Optional[Float32]

    def __init__(
        out self,
        k: Int = 10,
        ef_search: Int = 100,
        max_distance: Optional[Float32] = None,
    ):
        self.k = k
        self.ef_search = ef_search
        self.max_distance = max_distance


struct VectorStoreOptions(ImplicitlyCopyable):
    var metric: Metric
    var hnsw: HNSWParams
    var text_enabled: Bool
    var graph_enabled: Bool
    var sparse_enabled: Bool
    var backend: Int  # 0=HNSW, 1=SymphonyQG

    def __init__(
        out self,
        metric: Metric = Metric.L2,
        hnsw: HNSWParams = HNSWParams(),
        text_enabled: Bool = False,
        graph_enabled: Bool = False,
        sparse_enabled: Bool = False,
        backend: Int = 0,
    ):
        self.metric = metric
        self.hnsw = hnsw
        self.text_enabled = text_enabled
        self.graph_enabled = graph_enabled
        self.sparse_enabled = sparse_enabled
        self.backend = backend


struct SearchResult(Copyable, Movable):
    var id: String
    var vector_id: UInt64
    var distance: Float32
    var rrf_score: Float32  # Only meaningful for hybrid search
    var has_rrf_score: Bool  # True if rrf_score was explicitly set
    var metadata: Optional[String]
    var source_span: Optional[String]

    def __init__(
        out self,
        id: String,
        vector_id: UInt64,
        distance: Float32,
        metadata: Optional[String] = None,
        source_span: Optional[String] = None,
        rrf_score: Float32 = 0.0,
        has_rrf_score: Bool = False,
    ):
        self.id = id
        self.vector_id = vector_id
        self.distance = distance
        self.rrf_score = rrf_score
        self.has_rrf_score = has_rrf_score
        self.metadata = metadata
        self.source_span = source_span

    def __init__(out self, *, copy: Self):
        self.id = copy.id
        self.vector_id = copy.vector_id
        self.distance = copy.distance
        self.rrf_score = copy.rrf_score
        self.has_rrf_score = copy.has_rrf_score
        self.metadata = copy.metadata
        self.source_span = copy.source_span

    def __init__(out self, *, deinit take: Self):
        self.id = take.id^
        self.vector_id = take.vector_id
        self.distance = take.distance
        self.rrf_score = take.rrf_score
        self.has_rrf_score = take.has_rrf_score
        self.metadata = take.metadata^
        self.source_span = take.source_span^
