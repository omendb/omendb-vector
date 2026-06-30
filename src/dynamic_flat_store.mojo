"""Dynamic flat vector store for arbitrary dimensions.

This module provides a vector store that uses a dynamic flat index
for exact search, supporting any dimension up to 8192.
"""

from std.collections import Dict, List
from std.python import Python, PythonObject
from dynamic_flat import DynamicFlatIndex


struct DynamicFlatVectorStore(Movable):
    """A vector store using dynamic flat index for exact search.

    Supports arbitrary dimensions at runtime, up to a maximum of 8192.
    Uses linear scan for search - correct but not optimized for large datasets.
    """

    var dim: Int
    var index: DynamicFlatIndex
    var external_to_vector: Dict[String, UInt64]
    var vector_external_ids: List[String]
    var metadata: List[Optional[String]]
    var source_spans: List[Optional[String]]
    var deleted: List[Bool]
    var deleted_count: Int
    var persistent_path: Optional[String]

    def __init__(out self, dim: Int):
        self.dim = dim
        self.index = DynamicFlatIndex(dim)
        self.external_to_vector = Dict[String, UInt64]()
        self.vector_external_ids = List[String]()
        self.metadata = List[Optional[String]]()
        self.source_spans = List[Optional[String]]()
        self.deleted = List[Bool]()
        self.deleted_count = 0
        self.persistent_path = None

    def __init__(out self, *, deinit take: Self):
        self.dim = take.dim
        self.index = take.index^
        self.external_to_vector = take.external_to_vector^
        self.vector_external_ids = take.vector_external_ids^
        self.metadata = take.metadata^
        self.source_spans = take.source_spans^
        self.deleted = take.deleted^
        self.deleted_count = take.deleted_count
        self.persistent_path = take.persistent_path^

    def is_persistent(ref self) -> Bool:
        return self.persistent_path != None

    @staticmethod
    def create(path: String, dim: Int) raises -> Self:
        var os = Python.import_module("os")
        if Bool(os.path.exists(path)) and Bool(
            os.path.exists(path + "/manifest.json")
        ):
            raise Error("persistent dynamic flat store already exists")
        _ = os.makedirs(path, exist_ok=True)

        var store = Self(dim)
        store.persistent_path = Optional[String](path)
        store.flush()
        return store^

    @staticmethod
    def open(path: String) raises -> Self:
        var json = Python.import_module("json")
        var builtins = Python.import_module("builtins")
        var os = Python.import_module("os")
        if not Bool(os.path.exists(path + "/manifest.json")):
            raise Error("persistent dynamic flat store manifest not found")

        var manifest_file = builtins.open(path + "/manifest.json", "r")
        var manifest = json.loads(manifest_file.read())
        _ = manifest_file.close()

        var manifest_dim = Int(py=manifest["dim"])
        var store = Self(manifest_dim)
        store.persistent_path = Optional[String](path)
        store._load_records(path)
        store._load_tombstones(path)
        store._load_vectors(path)
        return store^

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
            tempfile.mkdtemp(
                prefix=".omendb-dynamic-" + basename + "-", dir=parent
            )
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
        self._write_vectors(path)
        self._write_records(path)
        self._write_tombstones(path)
        self._write_manifest(path)

    def _write_vectors(ref self, path: String) raises:
        var builtins = Python.import_module("builtins")
        var struct_module = Python.import_module("struct")
        var num_vectors = len(self.vector_external_ids)
        var num_floats = num_vectors * self.dim
        var vectors_file = builtins.open(path + "/vectors.bin", "wb")
        for i in range(num_floats):
            var packed = struct_module.pack("f", self.index.data[i])
            _ = vectors_file.write(packed)
        _ = vectors_file.close()

    def _load_vectors(mut self, path: String) raises:
        var builtins = Python.import_module("builtins")
        var struct_module = Python.import_module("struct")
        var os = Python.import_module("os")
        if not Bool(os.path.exists(path + "/vectors.bin")):
            raise Error("persistent dynamic flat store vectors not found")

        var vectors_file = builtins.open(path + "/vectors.bin", "rb")
        var data = vectors_file.read()
        _ = vectors_file.close()

        var num_vectors = len(self.vector_external_ids)
        var expected_bytes = num_vectors * self.dim * 4
        if Int(py=builtins.len(data)) != expected_bytes:
            raise Error("persistent dynamic flat store vectors size mismatch")

        self.index = DynamicFlatIndex(self.dim)
        self.index.reserve(num_vectors)
        for i in range(num_vectors * self.dim):
            var offset = i * 4
            var chunk = data[offset : offset + 4]
            var unpacked = struct_module.unpack("f", chunk)
            self.index.data.append(Float32(py=unpacked[0]))
        self.index.num_elements = num_vectors

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
            records.append(record)

        var records_file = builtins.open(path + "/records.json", "w")
        _ = records_file.write(json.dumps(records))
        _ = records_file.close()

    def _load_records(mut self, path: String) raises:
        var json = Python.import_module("json")
        var builtins = Python.import_module("builtins")
        var os = Python.import_module("os")
        if not Bool(os.path.exists(path + "/records.json")):
            raise Error("persistent dynamic flat store records not found")

        var records_file = builtins.open(path + "/records.json", "r")
        var records = json.loads(records_file.read())
        _ = records_file.close()

        self.external_to_vector = Dict[String, UInt64]()
        self.vector_external_ids = List[String]()
        self.metadata = List[Optional[String]]()
        self.source_spans = List[Optional[String]]()
        self.deleted = List[Bool]()
        self.deleted_count = 0

        for i in range(Int(py=builtins.len(records))):
            var record = records[i]
            var id = String(record["id"])
            var vector_id = UInt64(Int(py=record["vector_id"]))
            if Int(vector_id) != i:
                raise Error("persistent dynamic flat store vector ID mismatch")
            self.external_to_vector[id] = vector_id
            self.vector_external_ids.append(id)
            self.deleted.append(False)
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

    def _load_tombstones(mut self, path: String) raises:
        var json = Python.import_module("json")
        var builtins = Python.import_module("builtins")
        var os = Python.import_module("os")
        if not Bool(os.path.exists(path + "/tombstones.json")):
            raise Error("persistent dynamic flat store tombstones not found")

        var tombstones_file = builtins.open(path + "/tombstones.json", "r")
        var tombstones = json.loads(tombstones_file.read())
        _ = tombstones_file.close()

        if Int(py=builtins.len(tombstones)) != len(self.vector_external_ids):
            raise Error(
                "persistent dynamic flat store tombstone count mismatch"
            )
        self.external_to_vector = Dict[String, UInt64]()
        self.deleted_count = 0
        for i in range(Int(py=builtins.len(tombstones))):
            var tombstone = tombstones[i]
            var vector_id = Int(py=tombstone["vector_id"])
            if vector_id != i:
                raise Error(
                    "persistent dynamic flat store tombstone vector ID mismatch"
                )
            if Bool(tombstone["deleted"]):
                self.deleted[i] = True
                self.deleted_count += 1
            else:
                self.external_to_vector[self.vector_external_ids[i]] = UInt64(i)

    def _write_manifest(ref self, path: String) raises:
        var json = Python.import_module("json")
        var builtins = Python.import_module("builtins")
        var manifest = Python.dict()
        manifest["format_version"] = 1
        manifest["store_type"] = "dynamic_flat"
        manifest["dim"] = self.dim
        manifest["metric"] = "l2"
        manifest["num_vectors"] = len(self.vector_external_ids)
        manifest["record_count"] = len(self.vector_external_ids)
        manifest["tombstone_count"] = self.deleted_count
        manifest["index_mode"] = "flat"

        var files = Python.dict()
        files["vectors"] = "vectors.bin"
        files["records"] = "records.json"
        files["tombstones"] = "tombstones.json"
        manifest["files"] = files

        var manifest_file = builtins.open(path + "/manifest.json", "w")
        _ = manifest_file.write(json.dumps(manifest, sort_keys=True))
        _ = manifest_file.close()

    def len(ref self) -> Int:
        return len(self.vector_external_ids) - self.deleted_count

    def set(
        mut self,
        id: String,
        vector: Span[Float32, _],
        metadata: Optional[String] = None,
        source_span: Optional[String] = None,
    ) raises -> UInt64:
        if id.byte_length() == 0:
            raise Error("id must not be empty")
        if len(vector) != self.dim:
            raise Error("vector dimension mismatch")
        if id in self.external_to_vector:
            _ = self.delete(id)

        var vector_id = UInt64(len(self.vector_external_ids))
        self.index.insert(vector)
        self.external_to_vector[id] = vector_id
        self.vector_external_ids.append(id)
        self.metadata.append(metadata)
        self.source_spans.append(source_span)
        self.deleted.append(False)
        return vector_id

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
        return True

    def supersede(mut self, old_id: String, new_id: String) raises -> Bool:
        """Mark old_id as superseded by new_id."""
        if old_id not in self.external_to_vector:
            return False
        if new_id not in self.external_to_vector:
            raise Error("replacement item '" + new_id + "' not found")
        var vector_id = self.external_to_vector.pop(old_id)
        var idx = Int(vector_id)
        if idx >= len(self.deleted):
            raise Error("supersede vector ID out of range")
        if self.deleted[idx]:
            return False
        self.deleted[idx] = True
        self.deleted_count += 1
        return True

    def get(ref self, id: String) raises -> List[Float32]:
        if id not in self.external_to_vector:
            raise Error("id not found")
        var vector_id = Int(self.external_to_vector[id])
        if self.deleted[vector_id]:
            raise Error("id not found")
        var vec = self.index.get_vector(vector_id)
        var result = List[Float32]()
        result.reserve(self.dim)
        for i in range(self.dim):
            result.append(vec[i])
        return result^

    def search(
        ref self,
        query: Span[Float32, _],
        k: Int,
    ) raises -> List[DynamicFlatSearchResult]:
        """Search for k nearest neighbors using L2 distance."""
        var indices = self.index.search_with_distances(query, k)

        var results = List[DynamicFlatSearchResult]()
        results.reserve(len(indices))
        for i in range(len(indices)):
            var vector_id = indices[i]
            if Int(vector_id) >= len(self.deleted):
                continue
            if self.deleted[Int(vector_id)]:
                continue
            # Compute distance for this vector
            var vec = self.index.get_vector(Int(vector_id))
            var distance: Float32 = 0.0
            for j in range(self.dim):
                var delta = query[j] - vec[j]
                distance += delta * delta
            results.append(
                DynamicFlatSearchResult(
                    id=self.vector_external_ids[Int(vector_id)],
                    vector_id=UInt64(vector_id),
                    distance=distance,
                    metadata=self.metadata[Int(vector_id)],
                    source_span=self.source_spans[Int(vector_id)],
                )
            )
        return results^

    def search_exact_ids(
        ref self,
        query: Span[Float32, _],
        ids: List[String],
    ) raises -> List[DynamicFlatSearchResult]:
        """Search among specific IDs."""
        var results = List[DynamicFlatSearchResult]()
        for id in ids:
            if id not in self.external_to_vector:
                continue
            var vector_id = Int(self.external_to_vector[id])
            if self.deleted[vector_id]:
                continue
            var vec = self.index.get_vector(vector_id)
            var distance: Float32 = 0.0
            for j in range(self.dim):
                var delta = query[j] - vec[j]
                distance += delta * delta
            results.append(
                DynamicFlatSearchResult(
                    id=id,
                    vector_id=UInt64(vector_id),
                    distance=distance,
                    metadata=self.metadata[vector_id],
                    source_span=self.source_spans[vector_id],
                )
            )

        # Sort by distance
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                if results[i].distance > results[j].distance:
                    # Swap by rebuilding results
                    var id_i = results[i].id
                    var vec_id_i = results[i].vector_id
                    var dist_i = results[i].distance
                    var meta_i = results[i].metadata
                    var src_i = results[i].source_span
                    results[i] = DynamicFlatSearchResult(
                        id=results[j].id,
                        vector_id=results[j].vector_id,
                        distance=results[j].distance,
                        metadata=results[j].metadata,
                        source_span=results[j].source_span,
                    )
                    results[j] = DynamicFlatSearchResult(
                        id=id_i,
                        vector_id=vec_id_i,
                        distance=dist_i,
                        metadata=meta_i,
                        source_span=src_i,
                    )
        return results^


struct DynamicFlatSearchResult(Copyable, Movable):
    """Search result from dynamic flat index."""

    var id: String
    var vector_id: UInt64
    var distance: Float32
    var metadata: Optional[String]
    var source_span: Optional[String]

    def __init__(
        out self,
        id: String,
        vector_id: UInt64,
        distance: Float32,
        metadata: Optional[String] = None,
        source_span: Optional[String] = None,
    ):
        self.id = id
        self.vector_id = vector_id
        self.distance = distance
        self.metadata = metadata
        self.source_span = source_span

    def __init__(out self, *, copy: Self):
        self.id = copy.id
        self.vector_id = copy.vector_id
        self.distance = copy.distance
        self.metadata = copy.metadata
        self.source_span = copy.source_span

    def __init__(out self, *, deinit take: Self):
        self.id = take.id^
        self.vector_id = take.vector_id
        self.distance = take.distance
        self.metadata = take.metadata^
        self.source_span = take.source_span^
