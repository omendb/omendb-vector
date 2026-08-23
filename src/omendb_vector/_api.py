from __future__ import annotations

import ctypes
import hashlib
import json
import os
import shutil
import struct
import tempfile
from collections.abc import Set as AbstractSet
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, Literal, cast

from . import _native
from ._dimensions import (
    dense_dim_error,
    is_supported_dense_dim,
    is_supported_multivector_dim,
    multivector_dim_error,
    vector_mismatch_error,
)
from ._native import EngineUnavailableError

_fcntl: Any | None = import_module("fcntl") if os.name == "posix" else None

class _UNSET:
    """Sentinel for distinguishing 'not provided' from None in keyword args."""

    def __repr__(self) -> str:
        return "<UNSET>"


_UNSET = _UNSET()

type IndexMode = Literal["hnsw", "flat", "symphonyqg"]
type Metric = Literal["l2", "cosine", "dot"]
type SearchMode = Literal["vector", "text", "hybrid"]


# --- Native cJSON batch filter matching ---
_filter_lib: Any | None = None

def _get_filter_lib() -> Any:
    global _filter_lib
    if _filter_lib is None:
        lib_path = Path(__file__).parent / "libfilter.dylib"
        if not lib_path.exists():
            lib_path = Path(__file__).parent / "libfilter.so"
        if lib_path.exists():
            _filter_lib = ctypes.CDLL(str(lib_path))
            _filter_lib.filter_batch.argtypes = [
                ctypes.POINTER(ctypes.c_char_p),  # metadata_strings
                ctypes.POINTER(ctypes.c_int),      # deleted
                ctypes.c_int,                       # n
                ctypes.c_char_p,                    # filter_json
                ctypes.POINTER(ctypes.c_int),      # out_indices
            ]
            _filter_lib.filter_batch.restype = ctypes.c_int
    return _filter_lib

def _batch_filter_ids(
    all_ids: list[str],
    metadata_raw: list[Any],  # PythonObject from native
    deleted: list[bool],
    filter_json: str,
) -> list[str]:
    """Evaluate filter in one C call, return matching IDs."""
    lib = _get_filter_lib()
    if lib is None:
        return all_ids  # fallback
    n = len(all_ids)
    if n == 0:
        return []
    # Build C arrays
    meta_arr = (ctypes.c_char_p * n)()
    del_arr = (ctypes.c_int * n)()
    for i in range(n):
        raw = metadata_raw[i]
        if raw is not None:
            meta_arr[i] = str(raw).encode("utf-8")
        else:
            meta_arr[i] = b"{}"
        del_arr[i] = 1 if deleted[i] else 0
    out_arr = (ctypes.c_int * n)()
    count = lib.filter_batch(
        meta_arr, del_arr, n,
        filter_json.encode("utf-8"),
        out_arr,
    )
    if count < 0:
        return all_ids
    return [all_ids[out_arr[i]] for i in range(count)]


type GraphDirection = Literal["out", "in", "both"]
type VectorMode = Literal["single", "multi"]
type EncodingMode = Literal["none", "muvera"]

_MOJO_INT_BYTES = 8
_MOJO_F32_BYTES = 4
_MULTIVECTOR_MANIFEST_META_WIDTH = 6
_MULTIVECTOR_RECORD_META_WIDTH = 8
_MULTIVECTOR_SOURCE_META_WIDTH = 2
_MULTIVECTOR_TEXT_META_WIDTH = 3
_PUBLIC_INDEX_MODE_KEY = "public_index_mode"


class StoreBusyError(RuntimeError):
    """Raised when a persistent collection write is already in progress."""


def _collection_backup_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.bak")


def _collection_lock_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.write.lock")


@contextmanager
def _collection_write_lock(path: Path):
    lock_path = _collection_lock_path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if _fcntl is not None:
        lock_file = lock_path.open("a+", encoding="utf-8")
        try:
            try:
                _fcntl.flock(lock_file.fileno(), _fcntl.LOCK_EX | _fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise StoreBusyError(
                    f"collection write already in progress: {path}"
                ) from exc
            lock_file.seek(0)
            lock_file.truncate()
            lock_file.write(f"pid={os.getpid()}\n")
            lock_file.flush()
            yield
        finally:
            try:
                _fcntl.flock(lock_file.fileno(), _fcntl.LOCK_UN)
            finally:
                lock_file.close()
        return

    lock_fd: int | None = None
    try:
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise StoreBusyError(
                f"collection write already in progress: {path}"
            ) from exc
        os.write(lock_fd, f"pid={os.getpid()}\n".encode())
        yield
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass


def _recover_collection_path(path: Path) -> None:
    if path.exists():
        return
    backup_path = _collection_backup_path(path)
    if not backup_path.exists():
        return
    with _collection_write_lock(path):
        if not path.exists() and backup_path.exists():
            shutil.move(str(backup_path), str(path))


@dataclass(frozen=True, slots=True)
class HNSWConfig:
    m: int = 16
    ef_construction: int = 100
    ef_search: int = 100
    alpha: float = 1.0
    metric: str = "l2"

    def __post_init__(self) -> None:
        if self.m <= 0:
            raise ValueError("hnsw.m must be positive")
        if self.ef_construction <= 0:
            raise ValueError("hnsw.ef_construction must be positive")
        if self.ef_search <= 0:
            raise ValueError("hnsw.ef_search must be positive")
        if self.alpha <= 0.0:
            raise ValueError("hnsw.alpha must be positive")


@dataclass(frozen=True, slots=True)
class CollectionConfig:
    dim: int
    vector_mode: VectorMode = "single"
    index: IndexMode = "hnsw"
    metric: Metric = "l2"
    encoding: EncodingMode = "none"
    hnsw: HNSWConfig = field(default_factory=HNSWConfig)
    text: bool = False
    graph: bool = False
    sparse: bool = False

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.vector_mode not in ("single", "multi"):
            raise ValueError("vector_mode must be 'single' or 'multi'")
        if self.index not in ("hnsw", "flat", "symphonyqg"):
            raise ValueError("index must be 'hnsw', 'flat', or 'symphonyqg'")
        if self.index == "symphonyqg":
            # QG backend is available via VectorIndex(backend=1) in Mojo.
            # Python API passthrough pending competitive speed (currently 11.6x slower).
            if self.encoding != "none":
                raise ValueError("symphonyqg does not support muvera encoding")
        if self.metric not in ("l2", "cosine", "dot"):
            raise ValueError("metric must be 'l2', 'cosine', or 'dot'")
        if self.encoding not in ("none", "muvera"):
            raise ValueError("encoding must be 'none' or 'muvera'")
        if self.vector_mode == "single" and self.index == "flat" and self.metric not in ("l2", "cosine", "dot"):
            raise ValueError("flat index supports 'l2', 'cosine', and 'dot'")
        if self.vector_mode == "single" and self.encoding != "none":
            raise ValueError("single-vector collections require encoding='none'")
        if self.vector_mode == "single" and not is_supported_dense_dim(
            self.dim, self.index
        ):
            raise ValueError(dense_dim_error(self.dim, self.index))
        if self.vector_mode == "multi":
            if self.graph:
                raise ValueError("multi-vector collections do not support graph yet")
            if self.metric != "dot":
                raise ValueError(
                    "multi-vector collections currently require metric='dot'"
                )
            if self.encoding == "none" and self.index != "flat":
                raise ValueError("exact multi-vector collections require index='flat'")
            if self.encoding == "muvera" and self.index != "hnsw":
                raise ValueError("MuVERA multi-vector collections require index='hnsw'")
            if not is_supported_multivector_dim(self.dim, self.encoding):
                raise ValueError(multivector_dim_error(self.dim, self.encoding))


@dataclass(frozen=True, slots=True)
class SearchResult:
    id: str
    score: float
    distance: float | None = None
    metadata: dict[str, Any] | None = None
    source: dict[str, Any] | None = None
    text: str | None = None
    explanation: dict[str, Any] | None = None
    evidence: SearchEvidence | None = None


@dataclass(frozen=True, slots=True)
class SearchEvidence:
    method: str
    candidate_source: str | None = None
    candidate_k: int | None = None
    ef_search: int | None = None
    filter: dict[str, Any] | None = None
    score_parts: dict[str, Any] = field(default_factory=dict)
    rerank: str | None = None
    source: dict[str, Any] | None = None
    relationship: RelationshipEvidence | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RelationshipStep:
    edge_id: int
    from_id: str
    to_id: str
    type: str
    depth: int
    traversal_direction: GraphDirection
    weight: float | None = None


@dataclass(frozen=True, slots=True)
class RelationshipEvidence:
    from_id: str
    to_id: str
    included: bool
    reason: str
    direction: GraphDirection
    max_depth: int
    type: str | None
    path: tuple[str, ...] = ()
    steps: tuple[RelationshipStep, ...] = ()
    depth: int | None = None


@dataclass(frozen=True, slots=True)
class _RelationshipEdge:
    id: int
    from_id: str
    to_id: str
    type: str
    weight: float | None


@dataclass(frozen=True, slots=True)
class CheckIssue:
    collection: str | None
    code: str
    message: str
    path: str | None = None


@dataclass(frozen=True, slots=True)
class CheckResult:
    collections_checked: int
    issues: tuple[CheckIssue, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.issues


@dataclass(frozen=True, slots=True)
class MaintenanceStats:
    collection: str
    persistent: bool
    live_count: int
    record_count: int | None = None
    tombstone_count: int | None = None
    stale_candidate_count: int | None = None
    disk_bytes: int | None = None
    estimated_reclaimable_bytes: int | None = None
    store_layout: str | None = None
    index_mode: str | None = None
    encoding: str | None = None
    text_enabled: bool | None = None
    graph_enabled: bool | None = None
    needs_rebuild: bool = False
    notes: tuple[str, ...] = ()


@dataclass(slots=True)
class Collection:
    name: str
    config: CollectionConfig
    path: Path | None = None
    create: bool = True
    _native_collection: Any | None = field(default=None, repr=False)
    _relationship_edges: dict[tuple[str, str, str], _RelationshipEdge] | None = field(
        default=None, repr=False
    )
    _metadata_cache: dict[str, dict[str, Any] | None] | None = field(
        default=None, repr=False
    )
    _sparse_cache: dict[str, dict[int, float]] = field(default_factory=dict, repr=False)
    _text_only_store: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)

    def _invalidate_metadata_cache(self) -> None:
        self._metadata_cache = None

    def _load_text_only_store(self) -> None:
        """Load text-only records from the persisted sidecar file."""
        if self.path is None:
            return
        store_path = self.path / "text_only_store.json"
        if store_path.exists():
            self._text_only_store = json.loads(store_path.read_text())

    def _get_parsed_metadata(self, id: str) -> dict[str, Any] | None:
        if self._metadata_cache is None:
            self._metadata_cache = {}
        if id in self._metadata_cache:
            return self._metadata_cache[id]
        raw = self._native_handle().metadata_for_id(id)
        parsed = _metadata_from_native(raw)
        self._metadata_cache[id] = parsed
        return parsed

    def _native_handle(self) -> Any:
        if self.path is not None:
            _recover_collection_path(self.path)
        if self.config.vector_mode == "multi":
            if self.config.metric != "dot":
                raise EngineUnavailableError(
                    "Mojo multi-vector bindings currently support metric='dot' only"
                )
            if self.config.encoding == "none" and self.config.index != "flat":
                raise EngineUnavailableError(
                    "Mojo exact multi-vector bindings require index='flat'"
                )
            if self.config.encoding == "muvera" and self.config.index != "hnsw":
                raise EngineUnavailableError(
                    "Mojo MuVERA multi-vector bindings require index='hnsw'"
                )
            if self._native_collection is None:
                if self.path is not None and self.create:
                    with _collection_write_lock(self.path):
                        self._native_collection = _native.multivector_collection(
                            self.config.dim,
                            self.path,
                            create=self.create,
                            encoding=self.config.encoding,
                        )
                else:
                    self._native_collection = _native.multivector_collection(
                        self.config.dim,
                        self.path,
                        create=self.create,
                        encoding=self.config.encoding,
                    )
            return self._native_collection
        if self._native_collection is None:
            # Pass metric from CollectionConfig to HNSWConfig for native layer
            hnsw_with_metric = HNSWConfig(
                m=self.config.hnsw.m,
                ef_construction=self.config.hnsw.ef_construction,
                ef_search=self.config.hnsw.ef_search,
                alpha=self.config.hnsw.alpha,
                metric=self.config.metric,
            )
            if self.path is not None and self.create:
                with _collection_write_lock(self.path):
                    self._native_collection = _native.dense_collection(
                        self.config.dim,
                        self.path,
                        create=self.create,
                        hnsw=hnsw_with_metric,
                        index=self.config.index,
                    )
                    self._sync_public_manifest_index()
            else:
                self._native_collection = _native.dense_collection(
                    self.config.dim,
                    self.path,
                    create=self.create,
                    hnsw=hnsw_with_metric,
                    index=self.config.index,
                )
            if self.config.graph:
                self._native_collection.enable_graph()
        return self._native_collection

    def _check_vector(self, vector: list[float]) -> None:
        if len(vector) != self.config.dim:
            raise ValueError(vector_mismatch_error(self.config.dim, len(vector)))

    def _check_vectors(self, vectors: list[list[float]]) -> None:
        if not vectors:
            raise ValueError("vectors must contain at least one vector")
        for vector in vectors:
            if len(vector) != self.config.dim:
                raise ValueError(vector_mismatch_error(self.config.dim, len(vector)))

    def needs_compaction(self) -> bool:
        """True when tombstone ratio >= 25%%. Call vacuum() to reclaim space."""
        return bool(self._native_handle().needs_compaction())

    def set_sparse(self, id: str, sparse: dict[int, float]) -> None:
        """Set sparse vector {dim: weight} for an existing item.

        Call after set() or set_text(). Auto-enables sparse on the store.
        """
        self._native_handle().set_sparse(id, sparse)
        self._sparse_cache[id] = {int(dim): float(weight) for dim, weight in sparse.items()}

    def set(
        self,
        id: str,
        *,
        vector: list[float] | None = None,
        vectors: list[list[float]] | None = None,
        text: str | None = None,
        metadata: dict[str, Any] | None = None,
        source: dict[str, Any] | None = None,
        relationships: list[dict[str, Any]] | None = None,
        sparse: dict[int, float] | None = None,
    ) -> None:
        if text is not None and not self.config.text:
            raise EngineUnavailableError("text indexing requires config.text=True")
        metadata_json = (
            json.dumps(metadata, separators=(",", ":"))
            if metadata is not None
            else None
        )
        source_json = _source_to_native(source)
        if self.config.vector_mode == "multi":
            if vector is not None or vectors is None:
                raise ValueError("multi-vector collections require vectors")
            if relationships:
                raise EngineUnavailableError(
                    "relationships are not wired for multi-vector collections"
                )
            self._check_vectors(vectors)
            if self.get(id) is not None:
                self.delete(id)
            if text is not None:
                self._native_handle().set_text(
                    id, vectors, text, metadata_json, source_json
                )
            else:
                self._native_handle().set(id, vectors, metadata_json, source_json)
            # Index metadata for multi-vector path
            if metadata is not None:
                try:
                    record = self._native_handle().get(id, False)
                    if record is not None:
                        self._native_handle().index_metadata(
                            int(record['vector_id']), metadata
                        )
                except AttributeError:
                    pass  # Engine variant doesn't support metadata indexing
            return
        # Text-only items: store entirely in Python, never touch the engine.
        # The engine's placeholder-vector approach pollutes the HNSW graph.
        if text is not None and vector is None:
            # Force-initialize the native engine's text subsystem
            native = self._native_handle()
            if self.config.text and not self._text_only_store:
                # Seed the engine with a dummy so text search is enabled
                # (the engine requires at least one vector-backed text doc)
                try:
                    native.set_text(
                        "__omendb_vector_text_seed__",
                        [0.0] * self.config.dim,
                        "__seed__",
                        None,
                        None,
                    )
                    native.delete("__omendb_vector_text_seed__")
                except Exception:
                    pass
            self._text_only_store[id] = {
                "text": text,
                "metadata": metadata,
                "source": source,
                "relationships": relationships or [],
            }
            return
        if vectors is not None or vector is None:
            raise ValueError("single-vector collections require vector")
        self._check_vector(vector)
        if relationships:
            for relationship in relationships:
                if "to" not in relationship or "type" not in relationship:
                    raise ValueError("relationships require 'to' and 'type'")
        # Delete existing record before inserting (engine is append-only)
        existing = self.get(id)
        if existing is not None:
            self.delete(id)
        if text is not None:
            self._native_handle().set_text(
                id, vector, text, metadata_json, source_json
            )
            # Bridge may not return vector_id; look it up from record
            record = self._native_handle().get(id, False)
            vector_id = (
                int(record['vector_id']) if record is not None else None
            )
        else:
            result = self._native_handle().set(
                id, vector, metadata_json, source_json
            )
            vector_id = int(result) if result is not None else None
        # Index metadata for filtered search
        if metadata is not None and vector_id is not None:
            try:
                self._native_handle().index_metadata(
                    int(vector_id), metadata
                )
            except AttributeError:
                pass  # Engine variant doesn't support metadata indexing
        if relationships:
            for relationship in relationships:
                self.add_relationship(
                    id,
                    str(relationship["to"]),
                    type=str(relationship["type"]),
                    weight=relationship.get("weight"),
                )
        self._invalidate_metadata_cache()
        if sparse is not None:
            self.set_sparse(id, sparse)

    def set_many(
        self, records: list[dict[str, Any]], *, batch_size: int | None = None
    ) -> None:
        del batch_size
        # Fast path: single-vector, no text, no relationships — native batch insert
        if (
            self.config.vector_mode == "single"
            and not self.config.text
            and self.config.index == "hnsw"
            and all(
                r.get("vector") is not None
                and r.get("vectors") is None
                and r.get("text") is None
                and r.get("relationships") is None
                and r.get("source") is None
                for r in records
            )
        ):
            ids = [str(r["id"]) for r in records]
            # Flatten vectors into a single list
            flat: list[float] = []
            for r in records:
                flat.extend(r["vector"])
            metadata = [r.get("metadata") for r in records]
            vector_ids = self._native_handle().set_batch(ids, flat, metadata)
            # Batch index metadata — single round-trip instead of N calls
            if any(r.get("metadata") is not None for r in records):
                vid_list = [int(vector_ids[i]) for i in range(len(records))]
                meta_list = [r.get("metadata") for r in records]
                self._native_handle().index_metadata_batch(vid_list, meta_list)
            # Update metadata cache
            for r in records:
                rid = str(r["id"])
                meta = r.get("metadata")
                if meta is not None and self._metadata_cache is not None:
                    self._metadata_cache[rid] = meta
            return

        # Slow path: text, multi-vector, relationships, or source
        for record in records:
            self.set(
                str(record["id"]),
                vector=record.get("vector"),
                vectors=record.get("vectors"),
                text=record.get("text"),
                metadata=record.get("metadata"),
                source=record.get("source"),
                relationships=record.get("relationships"),
            )

    def get(
        self,
        id: str,
        *,
        include_vector: bool | None = None,
        include_vectors: bool | None = None,
        include_text: bool | None = None,
        include_source: bool | None = None,
        include_timestamps: bool = False,
    ) -> dict[str, Any] | None:
        # Default to returning what the collection stores
        if include_vector is None and include_vectors is None:
            if self.config.vector_mode == "multi":
                include_vectors = True
                include_vector = False
            else:
                include_vector = True
                include_vectors = False
        if include_text is None:
            include_text = self.config.text
        if include_source is None:
            include_source = True
        if include_vector and include_vectors:
            raise ValueError("choose include_vector or include_vectors, not both")
        # Check text-only store first
        if id in self._text_only_store:
            entry = self._text_only_store[id]
            result: dict[str, Any] = {"id": id, "metadata": entry["metadata"]}
            if include_text:
                result["text"] = entry["text"]
            if include_source:
                result["source"] = entry["source"]
            return result
        if self.config.vector_mode == "multi":
            if include_vector:
                raise ValueError("multi-vector collections use include_vectors")
            record = self._native_handle().get(id, include_vectors)
        else:
            if include_vectors:
                raise ValueError("single-vector collections use include_vector")
            record = self._native_handle().get(id, include_vector)
        if record is None:
            return None
        result: dict[str, Any] = {
            "id": str(record["id"]),
            "metadata": _metadata_from_native(record["metadata"]),
        }
        if include_timestamps:
            result["created_at"] = float(record.get("created_at", 0.0))
            result["updated_at"] = float(record.get("updated_at", 0.0))
            superseded_at = float(record.get("superseded_at", 0.0))
            if superseded_at > 0.0:
                result["superseded_at"] = superseded_at
        if include_source:
            result["source"] = _source_from_native(record["source"])
        if include_vector:
            result["vector"] = [float(value) for value in record["vector"]]
        if include_vectors:
            result["vectors"] = [
                [float(value) for value in vector] for vector in record["vectors"]
            ]
        if include_text:
            if not self.config.text:
                result["text"] = None
            else:
                text = self._native_handle().get_text(id)
                result["text"] = None if text is None else str(text)
        return result

    def count(self) -> int:
        """Return the number of live records in this collection."""
        return len(self._native_handle().live_ids()) + len(self._text_only_store)

    def multi_get(
        self,
        ids: list[str],
        *,
        include_vector: bool = False,
        include_vectors: bool = False,
        include_text: bool = False,
        include_source: bool = False,
        include_timestamps: bool = False,
    ) -> list[dict[str, Any] | None]:
        """Batch retrieve multiple items by ID. Returns list in same order as ids."""
        return [
            self.get(
                id,
                include_vector=include_vector,
                include_vectors=include_vectors,
                include_text=include_text,
                include_source=include_source,
                include_timestamps=include_timestamps,
            )
            for id in ids
        ]

    def search(
        self,
        *,
        vector: list[float] | None = None,
        vectors: list[list[float]] | None = None,
        text: str | None = None,
        k: int = 10,
        filter: dict[str, Any] | None = None,
        include_ids: list[str] | None = None,
        exclude_ids: list[str] | None = None,
        mode: SearchMode | None = None,
        explain: bool = False,
        ef: int | None = None,
        hybrid_alpha: float = 0.5,
        rrf_k: int = 60,
        sparse: dict[int, float] | None = None,
    ) -> list[SearchResult]:
        _check_positive_int("k", k)
        if ef is not None:
            _check_positive_int("ef", ef)
        _check_search_mode(mode)
        _validate_filter(filter)
        include_set = _normalize_id_constraint("include_ids", include_ids)
        exclude_set = _normalize_id_constraint("exclude_ids", exclude_ids)
        has_constraints = (
            filter is not None or include_set is not None or exclude_set is not None
        )
        supplied = sum(value is not None for value in (vector, vectors, text))
        if supplied > 1 and mode is None:
            raise ValueError(
                "mode is required when multiple query modalities are supplied"
            )
        if vector is not None and vectors is not None:
            raise ValueError("choose vector or vectors, not both")
        if self.config.vector_mode == "multi" and vector is not None:
            raise ValueError("multi-vector collections use vectors")
        if self.config.vector_mode == "single" and vectors is not None:
            raise ValueError("single-vector collections use vector")
        candidate_k = _candidate_count(k, has_constraints)
        if mode == "hybrid":
            if text is None or (vector is None and vectors is None):
                raise ValueError(
                    "hybrid search requires vector and text or vectors and text"
                )
            if not self.config.text:
                raise EngineUnavailableError("hybrid search requires config.text=True")
            if vectors is not None:
                if ef is not None:
                    raise EngineUnavailableError(
                        "ef is only supported for dense HNSW hybrid search"
                    )
                if hybrid_alpha != 0.5 or rrf_k != 60:
                    raise EngineUnavailableError(
                        "hybrid_alpha and rrf_k are only supported for dense RRF "
                        "hybrid search"
                    )
                return self._search_multivector(
                    vectors=vectors,
                    text=text,
                    k=k,
                    candidate_k=candidate_k,
                    filter=filter,
                    include_ids=include_set,
                    exclude_ids=exclude_set,
                    explain=explain,
                )
            if vector is None:
                raise ValueError("hybrid search requires vector")
            self._check_vector(vector)
            return self._search_hybrid(
                vector=vector,
                text=text,
                k=k,
                candidate_k=candidate_k,
                filter=filter,
                include_ids=include_set,
                exclude_ids=exclude_set,
                explain=explain,
                ef=ef,
                hybrid_alpha=hybrid_alpha,
                rrf_k=rrf_k,
                sparse=sparse,
            )
        if mode in (None, "vector") and vectors is not None:
            return self._search_multivector(
                vectors=vectors,
                text=None,
                k=k,
                candidate_k=candidate_k,
                filter=filter,
                include_ids=include_set,
                exclude_ids=exclude_set,
                explain=explain,
            )
        if (
            mode in (None, "text")
            and text is not None
            and (vector is None or mode == "text")
        ):
            if self.config.vector_mode == "multi":
                raise EngineUnavailableError(
                    "text-only search is not wired for multi-vector collections; "
                    "use hybrid search with vectors and text"
                )
            if not self.config.text:
                raise EngineUnavailableError("text search requires config.text=True")
            text_candidate_k = self._candidate_count_for_constraints(
                filter, include_set, exclude_set, candidate_k
            )
            results = self._search_results_from_native(
                self._native_handle().search_text(text, text_candidate_k),
                include_distance=False,
            )
            # Merge text-only store results (Python-side simple match)
            if self._text_only_store:
                query_lower = text.lower().split()
                for tid, entry in list(self._text_only_store.items()):
                    entry_text = entry.get("text", "").lower()
                    if any(w in entry_text for w in query_lower):
                        results.append(
                            SearchResult(
                                id=tid,
                                score=-1.0,  # simple match score
                                metadata=entry.get("metadata"),
                                source=entry.get("source"),
                            )
                        )
            filtered = _apply_filter(results, filter, k, include_set, exclude_set)
            if explain:
                return _with_explanation(
                    filtered,
                    {
                        "method": "bm25",
                        "candidate_k": text_candidate_k,
                        "filter": self._constraint_evidence(
                            filter,
                            include_set,
                            exclude_set,
                            strategy="all_live_candidates",
                        ),
                    },
                )
            return filtered
        if mode in (None, "vector") and vector is not None:
            self._check_vector(vector)
            # Bitmap path: Mojo metadata index + ACORN 2-hop (primary)
            if filter is not None and self.config.index == "hnsw":
                bitmap_results = self._native_handle().search_filtered_by_bitmap(
                    vector,
                    filter,
                    k,
                    ef if ef is not None else self.config.hnsw.ef_search,
                )
                results = self._search_results_from_native(
                    bitmap_results,
                    include_distance=True,
                )
                # Apply include/exclude constraints
                results = _apply_filter(results, None, k, include_set, exclude_set)
                # Fall back to exact if HNSW+bitmap returns too few or wrong results
                hnsw_count = len(results)
                eligible_ids = self._allowlist_ids(
                    filter, include_set, exclude_set
                )
                if eligible_ids and (
                    hnsw_count < k
                    or self._hnsw_results_might_be_wrong(
                        results, vector, eligible_ids
                    )
                ):
                    exact_results = self._search_results_from_native(
                        self._native_handle().search_exact_ids(
                            vector, eligible_ids
                        ),
                        include_distance=True,
                    )
                    exact_results.sort(
                        key=lambda result: result.distance or 0.0
                    )
                    results = exact_results[:k]
                    if explain:
                        return _with_explanation(
                            results,
                            {
                                "method": "exact_l2",
                                "candidates_generated": len(eligible_ids),
                                "candidates_scored": len(eligible_ids),
                                "fallback_reason": (
                                    "hnsw_bitmap_underrun_exact_fallback"
                                    if hnsw_count < k
                                    else "hnsw_bitmap_quality_exact_fallback"
                                ),
                                "hnsw_bitmap_candidates": hnsw_count,
                                "filter": self._constraint_evidence(
                                    filter,
                                    include_set,
                                    exclude_set,
                                    strategy="exact_allowlist",
                                    allowlist_count=len(eligible_ids),
                                ),
                            },
                        )
                        return results
                if explain:
                    return _with_explanation(
                        results,
                        {
                            "method": "hnsw_filtered_bitmap",
                            "candidates_generated": len(results),
                            "candidates_scored": len(results),
                            "vector_ef": ef if ef is not None else self.config.hnsw.ef_search,
                            "filter": self._constraint_evidence(
                                filter,
                                include_set,
                                exclude_set,
                                strategy="engine_bitmap_filter",
                            ),
                        },
                    )
                return results
            # Include/exclude only (no metadata filter)
            allowlist = self._allowlist_ids(None, include_set, exclude_set)
            if allowlist is not None:
                if self.config.index == "hnsw":
                    results = self._search_results_from_native(
                        self._native_handle().search_filtered_ids(
                            vector,
                            allowlist,
                            k,
                            ef if ef is not None else self.config.hnsw.ef_search,
                        ),
                        include_distance=True,
                    )
                    # Fall back to exact if HNSW overfetch returns too few
                    hnsw_count = len(results)
                    if len(allowlist) > 0 and hnsw_count < k:
                        exact_results = self._search_results_from_native(
                            self._native_handle().search_exact_ids(
                                vector, allowlist
                            ),
                            include_distance=True,
                        )
                        exact_results.sort(
                            key=lambda result: result.distance or 0.0
                        )
                        results = exact_results[:k]
                        if explain:
                            return _with_explanation(
                                results,
                                {
                                    "method": "exact_l2",
                                    "candidates_generated": len(allowlist),
                                    "candidates_scored": len(allowlist),
                                    "fallback_reason": "hnsw_underrun_exact_fallback",
                                    "hnsw_candidates": hnsw_count,
                                    "filter": self._constraint_evidence(
                                        filter,
                                        include_set,
                                        exclude_set,
                                        strategy="exact_allowlist",
                                        allowlist_count=len(allowlist),
                                    ),
                                },
                            )
                        return results
                    if explain:
                        return _with_explanation(
                            results,
                            {
                                "method": "hnsw_filtered_l2",
                                "candidates_generated": len(results),
                                "candidates_scored": len(results),
                                "vector_ef": ef if ef is not None else self.config.hnsw.ef_search,
                                "filter": self._constraint_evidence(
                                    filter,
                                    include_set,
                                    exclude_set,
                                    strategy="engine_bitmap_filter",
                                    allowlist_count=len(allowlist),
                                ),
                            },
                        )
                    return results
                exact_results = self._search_results_from_native(
                    self._native_handle().search_exact_ids(vector, allowlist),
                    include_distance=True,
                )
                exact_results.sort(key=lambda result: result.distance or 0.0)
                filtered = exact_results[:k]
                if explain:
                    return _with_explanation(
                        filtered,
                        {
                            "method": "exact_l2",
                            "candidates_generated": len(allowlist),
                            "candidates_scored": len(allowlist),
                            "fallback_reason": "selective_filter_exact_fallback",
                            "filter": self._constraint_evidence(
                                filter,
                                include_set,
                                exclude_set,
                                strategy="exact_allowlist",
                                allowlist_count=len(allowlist),
                            ),
                        },
                    )
                return filtered
            if self.config.index == "flat":
                if ef is not None:
                    raise ValueError("ef is only supported for hnsw vector search")
                all_live_ids = self._all_live_ids()
                exact_results = self._search_results_from_native(
                    self._native_handle().search_exact_ids(vector, all_live_ids),
                    include_distance=True,
                )
                exact_results.sort(key=lambda result: result.distance or 0.0)
                filtered = exact_results[:k]
                if explain:
                    return _with_explanation(
                        filtered,
                        {
                            "method": "exact_l2",
                            "candidate_source": "flat_all_live",
                            "candidate_k": len(all_live_ids),
                        },
                    )
                return filtered
            results = self._search_results_from_native(
                self._native_handle().search(
                    vector,
                    candidate_k,
                    ef if ef is not None else self.config.hnsw.ef_search,
                ),
                include_distance=True,
            )
            filtered = _apply_filter(results, filter, k, include_set, exclude_set)
            if explain:
                return _with_explanation(
                    filtered,
                    {
                        "method": "hnsw_l2",
                        "candidate_k": candidate_k,
                        "ef_search": ef
                        if ef is not None
                        else self.config.hnsw.ef_search,
                        "filter": self._constraint_evidence(
                            filter, include_set, exclude_set, strategy="post_filter"
                        ),
                    },
                )
            return filtered
        raise EngineUnavailableError(
            f"selected search mode '{mode}' does not support the requested "
            f"combination of parameters"
        )

    def _search_multivector(
        self,
        *,
        vectors: list[list[float]],
        text: str | None,
        k: int,
        candidate_k: int,
        filter: dict[str, Any] | None,
        include_ids: AbstractSet[str] | None,
        exclude_ids: AbstractSet[str] | None,
        explain: bool,
    ) -> list[SearchResult]:
        if self.config.vector_mode != "multi":
            raise ValueError("vectors search requires vector_mode='multi'")
        self._check_vectors(vectors)
        has_constraints = (
            filter is not None or include_ids is not None or exclude_ids is not None
        )
        if text is None and has_constraints:
            allowlist = self._allowlist_ids(filter, include_ids, exclude_ids)
            if allowlist is None:
                return []
            return self._search_multivector_exact_ids(
                vectors=vectors,
                ids=allowlist,
                k=k,
                filter=filter,
                include_ids=include_ids,
                exclude_ids=exclude_ids,
                explain=explain,
            )
        native_k = (
            self._candidate_count_for_constraints(
                filter, include_ids, exclude_ids, candidate_k
            )
            if text is not None
            else candidate_k
        )
        native_results = (
            self._native_handle().search_text(text, vectors, native_k)
            if text is not None
            else self._native_handle().search(vectors, native_k)
        )
        results: list[SearchResult] = []
        for item in native_results:
            maxsim = float(item["maxsim"])
            explanation: dict[str, Any] | None = (
                {"method": "maxsim", "maxsim": maxsim} if explain else None
            )
            if text is not None and explanation is not None:
                explanation["candidate_source"] = "bm25"
                if has_constraints:
                    explanation["candidate_k"] = native_k
                    explanation["filter"] = self._constraint_evidence(
                        filter,
                        include_ids,
                        exclude_ids,
                        strategy="all_live_bm25_candidates",
                    )
            source = _source_from_native(item["source"])
            score = float(item["score"])
            results.append(
                SearchResult(
                    id=str(item["id"]),
                    score=score,
                    distance=None,
                    metadata=_metadata_from_native(item["metadata"]),
                    source=source,
                    explanation=explanation,
                    evidence=_search_evidence_from_explanation(
                        explanation,
                        score=score,
                        distance=None,
                        source=source,
                    )
                    if explanation is not None
                    else None,
                )
            )
        return _apply_filter(results, filter, k, include_ids, exclude_ids)

    def _search_multivector_exact_ids(
        self,
        *,
        vectors: list[list[float]],
        ids: list[str],
        k: int,
        filter: dict[str, Any] | None,
        include_ids: AbstractSet[str] | None,
        exclude_ids: AbstractSet[str] | None,
        explain: bool,
    ) -> list[SearchResult]:
        results: list[SearchResult] = []
        for id in ids:
            record = self.get(id, include_vectors=True, include_source=True)
            if record is None:
                continue
            doc_vectors = cast(list[list[float]], record["vectors"])
            maxsim = _maxsim_score(vectors, doc_vectors)
            explanation = (
                {
                    "method": "maxsim",
                    "maxsim": maxsim,
                    "candidate_source": "exact_filter_allowlist",
                    "filter": self._constraint_evidence(
                        filter,
                        include_ids,
                        exclude_ids,
                        strategy="exact_allowlist",
                        allowlist_count=len(ids),
                    ),
                }
                if explain
                else None
            )
            source = cast(dict[str, Any] | None, record.get("source"))
            results.append(
                SearchResult(
                    id=id,
                    score=maxsim,
                    distance=None,
                    metadata=cast(dict[str, Any] | None, record["metadata"]),
                    source=source,
                    explanation=explanation,
                    evidence=_search_evidence_from_explanation(
                        explanation,
                        score=maxsim,
                        distance=None,
                        source=source,
                    )
                    if explanation is not None
                    else None,
                )
            )
        results.sort(key=lambda result: (-result.score, result.id))
        return results[:k]

    def _search_hybrid(
        self,
        *,
        vector: list[float],
        text: str,
        k: int,
        candidate_k: int,
        filter: dict[str, Any] | None,
        include_ids: AbstractSet[str] | None,
        exclude_ids: AbstractSet[str] | None,
        explain: bool,
        ef: int | None,
        hybrid_alpha: float,
        rrf_k: int,
        sparse: dict[int, float] | None,
    ) -> list[SearchResult]:
        if not 0.0 <= hybrid_alpha <= 1.0:
            raise ValueError("hybrid_alpha must be between 0.0 and 1.0")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")

        allowlist = self._allowlist_ids(filter, include_ids, exclude_ids)

        # Fast path: no filters, HNSW index → native Mojo RRF fusion
        if (
            allowlist is None
            and self.config.index == "hnsw"
            and not explain
            and sparse is None
        ):
            native_results = self._native_handle().search_hybrid(
                vector,
                text,
                {
                    "k": candidate_k,
                    "alpha": hybrid_alpha,
                    "rrf_k": rrf_k,
                    "ef_search": ef if ef is not None else self.config.hnsw.ef_search,
                },
            )
            results = self._search_results_from_native(
                native_results, include_distance=True
            )
            return _apply_filter(results, filter, k, include_ids, exclude_ids)

        # Slow path: filters or explain mode → Python-layer RRF
        vector_filter_strategy = "post_filter"
        if allowlist is not None:
            if self.config.index == "hnsw":
                vector_results = self._search_results_from_native(
                    self._native_handle().search_filtered_ids(
                        vector,
                        allowlist,
                        candidate_k,
                        ef if ef is not None else self.config.hnsw.ef_search,
                    ),
                    include_distance=True,
                )
                vector_filter_strategy = "engine_bitmap_filter"
            else:
                vector_results = self._search_results_from_native(
                    self._native_handle().search_exact_ids(vector, allowlist),
                    include_distance=True,
                )
                vector_results.sort(key=lambda result: result.distance or 0.0)
                vector_results = vector_results[:candidate_k]
                vector_filter_strategy = "exact_allowlist"
        elif self.config.index == "flat":
            if ef is not None:
                raise ValueError("ef is only supported for hnsw vector search")
            all_live_ids = self._all_live_ids()
            vector_results = self._search_results_from_native(
                self._native_handle().search_exact_ids(vector, all_live_ids),
                include_distance=True,
            )
            vector_results.sort(key=lambda result: result.distance or 0.0)
            vector_results = vector_results[:candidate_k]
            vector_filter_strategy = "flat_all_live"
        else:
            vector_results = self._search_results_from_native(
                self._native_handle().search(
                    vector,
                    candidate_k,
                    ef if ef is not None else self.config.hnsw.ef_search,
                ),
                include_distance=True,
            )
        text_candidate_k = self._candidate_count_for_constraints(
            filter, include_ids, exclude_ids, candidate_k
        )
        text_results = self._search_results_from_native(
            self._native_handle().search_text(text, text_candidate_k),
            include_distance=False,
        )

        sparse_results = self._search_sparse_python(sparse, text_candidate_k)

        fused: dict[str, SearchResult] = {}
        vector_ranks: dict[str, int] = {}
        text_ranks: dict[str, int] = {}
        sparse_ranks: dict[str, int] = {}
        vector_scores: dict[str, float] = {}
        text_scores: dict[str, float] = {}
        sparse_scores: dict[str, float] = {}
        distances: dict[str, float | None] = {}
        metadata: dict[str, dict[str, Any] | None] = {}
        sources: dict[str, dict[str, Any] | None] = {}

        for rank, result in enumerate(vector_results, start=1):
            vector_ranks[result.id] = rank
            vector_scores[result.id] = result.score
            distances[result.id] = result.distance
            metadata[result.id] = result.metadata
            sources[result.id] = result.source
            fused[result.id] = result
        for rank, result in enumerate(text_results, start=1):
            text_ranks[result.id] = rank
            text_scores[result.id] = result.score
            metadata.setdefault(result.id, result.metadata)
            sources.setdefault(result.id, result.source)
            fused.setdefault(result.id, result)
        for rank, result in enumerate(sparse_results, start=1):
            sparse_ranks[result.id] = rank
            sparse_scores[result.id] = result.score
            metadata.setdefault(result.id, result.metadata)
            sources.setdefault(result.id, result.source)
            fused.setdefault(result.id, result)

        results: list[SearchResult] = []
        for id, base in fused.items():
            vector_rank = vector_ranks.get(id)
            text_rank = text_ranks.get(id)
            score = 0.0
            if vector_rank is not None:
                score += hybrid_alpha / (rrf_k + vector_rank)
            sparse_rank = sparse_ranks.get(id)
            if text_rank is not None:
                text_weight = 1.0 - hybrid_alpha
                if sparse is not None:
                    text_weight = text_weight / 2.0
                score += text_weight / (rrf_k + text_rank)
            if sparse_rank is not None:
                score += ((1.0 - hybrid_alpha) / 2.0) / (rrf_k + sparse_rank)

            explanation: dict[str, Any] | None = None
            if explain:
                explanation = {
                    "method": "rrf",
                    "rrf_k": rrf_k,
                    "hybrid_alpha": hybrid_alpha,
                    "vector": {
                        "rank": vector_rank,
                        "score": vector_scores.get(id),
                        "distance": distances.get(id),
                    },
                    "text": {
                        "rank": text_rank,
                        "score": text_scores.get(id),
                    },
                    "sparse": {
                        "rank": sparse_rank,
                        "score": sparse_scores.get(id),
                    },
                }
                if (
                    filter is not None
                    or include_ids is not None
                    or exclude_ids is not None
                ):
                    explanation["filter"] = {
                        "vector": self._constraint_evidence(
                            filter,
                            include_ids,
                            exclude_ids,
                            strategy=vector_filter_strategy,
                            allowlist_count=len(allowlist)
                            if allowlist is not None
                            else None,
                        ),
                        "text": self._constraint_evidence(
                            filter,
                            include_ids,
                            exclude_ids,
                            strategy="all_live_bm25_candidates",
                            candidate_k=text_candidate_k,
                        ),
                    }

            results.append(
                SearchResult(
                    id=id,
                    score=score,
                    distance=distances.get(id, base.distance),
                    metadata=metadata.get(id, base.metadata),
                    source=sources.get(id, base.source),
                    explanation=explanation,
                    evidence=_search_evidence_from_explanation(
                        explanation,
                        score=score,
                        distance=distances.get(id, base.distance),
                        source=sources.get(id, base.source),
                    )
                    if explanation is not None
                    else None,
                )
            )

        results.sort(key=lambda result: (-result.score, result.id))
        return _apply_filter(results, filter, k, include_ids, exclude_ids)

    def _search_sparse_python(
        self, sparse: dict[int, float] | None, k: int
    ) -> list[SearchResult]:
        if sparse is None or k <= 0:
            return []
        query = {int(dim): float(weight) for dim, weight in sparse.items() if weight != 0.0}
        if not query:
            return []

        scored: list[tuple[float, str]] = []
        for id, doc_sparse in self._sparse_cache.items():
            score = 0.0
            for dim, query_weight in query.items():
                score += query_weight * doc_sparse.get(dim, 0.0)
            if score != 0.0:
                scored.append((score, id))
        scored.sort(key=lambda item: (-item[0], item[1]))

        results: list[SearchResult] = []
        for score, id in scored[:k]:
            record = self.get(id, include_source=True)
            if record is None:
                continue
            results.append(
                SearchResult(
                    id=id,
                    score=score,
                    distance=None,
                    metadata=cast(dict[str, Any] | None, record["metadata"]),
                    source=cast(dict[str, Any] | None, record["source"]),
                )
            )
        return results

    def _allowlist_ids(
        self,
        filter: dict[str, Any] | None,
        include_ids: AbstractSet[str] | None,
        exclude_ids: AbstractSet[str] | None,
    ) -> list[str] | None:
        if filter is None and include_ids is None and exclude_ids is None:
            return None
        native = self._native_handle()
        # Always iterate native live IDs. The metadata cache may be partial.
        all_ids = native.live_ids()
        cache = self._metadata_cache
        if filter is not None:
            result = []
            # Pre-extract filter key/operator pairs for fast evaluation
            filter_ops = []
            for key, expected in filter.items():
                if isinstance(expected, dict):
                    for op, val in expected.items():
                        filter_ops.append((key, op, val))
                else:
                    filter_ops.append((key, "", expected))
            for id in all_ids:
                if include_ids is not None and id not in include_ids:
                    continue
                if exclude_ids is not None and id in exclude_ids:
                    continue
                m = (
                    cache.get(id)
                    if cache is not None and id in cache
                    else self._get_parsed_metadata(id)
                )
                if m is None:
                    continue
                match = True
                for key, op, val in filter_ops:
                    if key not in m:
                        match = False
                        break
                    actual = m[key]
                    if op == "$exists":
                        if (key in m) != bool(val):
                            match = False
                            break
                    elif op == "$in":
                        if actual not in val:
                            match = False
                            break
                    elif op == "$gt":
                        if not (actual > val):
                            match = False
                            break
                    elif op == "$lt":
                        if not (actual < val):
                            match = False
                            break
                    elif op == "$gte":
                        if not (actual >= val):
                            match = False
                            break
                    elif op == "$lte":
                        if not (actual <= val):
                            match = False
                            break
                    elif op == "":
                        if actual != val:
                            match = False
                            break
                if match:
                    result.append(id)
            return result
        # No filter, just apply include/exclude
        result = []
        include_set_normalized = include_ids
        exclude_set_normalized = exclude_ids
        for id in all_ids:
            if include_set_normalized is not None and id not in include_set_normalized:
                continue
            if exclude_set_normalized is not None and id in exclude_set_normalized:
                continue
            result.append(id)
        return result

    def _hnsw_results_might_be_wrong(
        self,
        results: list[SearchResult],
        vector: list[float],
        eligible_ids: list[str],
    ) -> bool:
        """Check if HNSW bitmap results likely missed nearer candidates.

        Samples a few IDs from the eligible set, computes exact distances,
        and checks if any sample is closer than the worst HNSW result.
        If so, HNSW with the current ef is unreliable — fall back to exact.
        """
        if not results or not eligible_ids:
            return False
        worst_distance = max(
            (r.distance if r.distance is not None else float("inf"))
            for r in results
        )
        # Sample up to 10 IDs from eligible set
        sample_size = min(10, len(eligible_ids))
        import random
        sampled = random.sample(eligible_ids, sample_size)
        # Compute exact distances for sampled IDs
        native = self._native_handle()
        exact_sample = native.search_exact_ids(vector, sampled)
        for item in exact_sample[:sample_size]:
            dist = item.get("distance")
            if dist is not None and dist < worst_distance:
                return True  # Sampled ID is closer than HNSW worst result
        return False

    def _all_live_ids(self) -> list[str]:
        return self._native_handle().live_ids()

    def _matching_records(
        self,
        filter: dict[str, Any] | None,
        include_ids: AbstractSet[str] | None,
        exclude_ids: AbstractSet[str] | None,
    ) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        for record in self._native_handle().records():
            id = str(record["id"])
            metadata = _metadata_from_native(record["metadata"])
            if _matches_constraints(id, metadata, filter, include_ids, exclude_ids):
                matches.append(cast(dict[str, Any], record))
        return matches

    def _candidate_count_for_constraints(
        self,
        filter: dict[str, Any] | None,
        include_ids: AbstractSet[str] | None,
        exclude_ids: AbstractSet[str] | None,
        default: int,
    ) -> int:
        if filter is None and include_ids is None and exclude_ids is None:
            return default
        return max(default, len(self._native_handle().records()))

    def _constraint_evidence(
        self,
        filter: dict[str, Any] | None,
        include_ids: AbstractSet[str] | None,
        exclude_ids: AbstractSet[str] | None,
        *,
        strategy: str,
        allowlist_count: int | None = None,
        candidate_k: int | None = None,
        candidates_generated: int | None = None,
        candidates_scored: int | None = None,
        vector_ef: int | None = None,
        fallback_reason: str | None = None,
    ) -> dict[str, Any] | None:
        if filter is None and include_ids is None and exclude_ids is None:
            return None
        evidence: dict[str, Any] = {"strategy": strategy}
        if include_ids is not None:
            evidence["include_count"] = len(include_ids)
        if exclude_ids is not None:
            evidence["exclude_count"] = len(exclude_ids)
        if allowlist_count is not None:
            evidence["eligible_count"] = allowlist_count
            total = self._native_handle().len()
            if total > 0:
                evidence["selectivity"] = round(allowlist_count / total, 4)
        if candidate_k is not None:
            evidence["candidate_k"] = candidate_k
        if candidates_generated is not None:
            evidence["candidates_generated"] = candidates_generated
        if candidates_scored is not None:
            evidence["candidates_scored"] = candidates_scored
        if vector_ef is not None:
            evidence["vector_ef"] = vector_ef
        if fallback_reason is not None:
            evidence["fallback_reason"] = fallback_reason
        return evidence

    def _search_results_from_native(
        self, native_results: list[dict[str, Any]], *, include_distance: bool
    ) -> list[SearchResult]:
        results: list[SearchResult] = []
        for item in native_results:
            results.append(
                SearchResult(
                    id=str(item["id"]),
                    score=float(item["score"]),
                    distance=float(item["distance"]) if include_distance else None,
                    metadata=_metadata_from_native(item["metadata"]),
                    source=_source_from_native(item["source"]),
                )
            )
        return results

    def search_vector(
        self,
        vector: list[float],
        *,
        k: int = 10,
        filter: dict[str, Any] | None = None,
        include_ids: list[str] | None = None,
        exclude_ids: list[str] | None = None,
        explain: bool = False,
        ef: int | None = None,
    ) -> list[SearchResult]:
        return self.search(
            vector=vector,
            k=k,
            filter=filter,
            include_ids=include_ids,
            exclude_ids=exclude_ids,
            mode="vector",
            explain=explain,
            ef=ef,
        )

    def search_vectors(
        self,
        vectors: list[list[float]],
        *,
        k: int = 10,
        filter: dict[str, Any] | None = None,
        include_ids: list[str] | None = None,
        exclude_ids: list[str] | None = None,
        explain: bool = False,
    ) -> list[SearchResult]:
        return self.search(
            vectors=vectors,
            k=k,
            filter=filter,
            include_ids=include_ids,
            exclude_ids=exclude_ids,
            mode="vector",
            explain=explain,
        )

    def search_text(
        self,
        text: str,
        *,
        k: int = 10,
        filter: dict[str, Any] | None = None,
        include_ids: list[str] | None = None,
        exclude_ids: list[str] | None = None,
        explain: bool = False,
    ) -> list[SearchResult]:
        return self.search(
            text=text,
            k=k,
            filter=filter,
            include_ids=include_ids,
            exclude_ids=exclude_ids,
            mode="text",
            explain=explain,
        )

    def search_hybrid(
        self,
        *,
        vector: list[float] | None = None,
        vectors: list[list[float]] | None = None,
        text: str,
        k: int = 10,
        filter: dict[str, Any] | None = None,
        include_ids: list[str] | None = None,
        exclude_ids: list[str] | None = None,
        explain: bool = False,
        ef: int | None = None,
        hybrid_alpha: float = 0.5,
        rrf_k: int = 60,
        sparse: dict[int, float] | None = None,
    ) -> list[SearchResult]:
        return self.search(
            vector=vector,
            vectors=vectors,
            text=text,
            k=k,
            filter=filter,
            include_ids=include_ids,
            exclude_ids=exclude_ids,
            mode="hybrid",
            explain=explain,
            ef=ef,
            hybrid_alpha=hybrid_alpha,
            rrf_k=rrf_k,
            sparse=sparse,
        )

    def add_relationship(
        self, from_id: str, to_id: str, *, type: str, weight: float | None = None
    ) -> int:
        if not self.config.graph:
            raise EngineUnavailableError("relationships require config.graph=True")
        if not type:
            raise ValueError("relationship type must not be empty")
        edge_id = int(self._native_handle().add_edge(from_id, to_id, type, weight))
        catalog = self._relationship_edge_catalog()
        catalog[(from_id, to_id, type)] = _RelationshipEdge(
            id=edge_id,
            from_id=from_id,
            to_id=to_id,
            type=type,
            weight=weight,
        )
        self._renumber_relationship_edges()
        return edge_id

    def remove_relationship(self, from_id: str, to_id: str, *, type: str) -> bool:
        if not self.config.graph:
            raise EngineUnavailableError("relationships require config.graph=True")
        removed = bool(self._native_handle().remove_edge(from_id, to_id, type))
        if removed:
            self._relationship_edge_catalog().pop((from_id, to_id, type), None)
            self._renumber_relationship_edges()
        return removed

    def neighbors(
        self,
        id: str,
        *,
        direction: GraphDirection = "out",
        type: str | None = None,
    ) -> list[str]:
        if not self.config.graph:
            raise EngineUnavailableError("relationships require config.graph=True")
        _check_graph_direction(direction)
        return [
            str(value) for value in self._native_handle().neighbors(id, direction, type)
        ]

    def traverse(
        self,
        id: str,
        *,
        direction: GraphDirection = "out",
        max_depth: int = 10,
        type: str | None = None,
    ) -> list[str]:
        if not self.config.graph:
            raise EngineUnavailableError("relationships require config.graph=True")
        _check_graph_direction(direction)
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        return [
            str(value)
            for value in self._native_handle().traverse(id, direction, max_depth, type)
        ]

    def has_path(
        self,
        from_id: str,
        to_id: str,
        *,
        direction: GraphDirection = "out",
        max_depth: int = 10,
        type: str | None = None,
    ) -> bool:
        if not self.config.graph:
            raise EngineUnavailableError("relationships require config.graph=True")
        _check_graph_direction(direction)
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        return bool(
            self._native_handle().has_path(from_id, to_id, direction, max_depth, type)
        )

    def shortest_path(
        self,
        from_id: str,
        to_id: str,
        *,
        direction: GraphDirection = "out",
        max_depth: int = 10,
        type: str | None = None,
    ) -> list[str]:
        if not self.config.graph:
            raise EngineUnavailableError("relationships require config.graph=True")
        _check_graph_direction(direction)
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        return [
            str(value)
            for value in self._native_handle().shortest_path(
                from_id, to_id, direction, max_depth, type
            )
        ]

    def relationship_evidence(
        self,
        from_id: str,
        to_id: str,
        *,
        direction: GraphDirection = "out",
        max_depth: int = 10,
        type: str | None = None,
    ) -> RelationshipEvidence:
        if not self.config.graph:
            raise EngineUnavailableError("relationships require config.graph=True")
        _check_graph_direction(direction)
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative")

        path = tuple(
            self.shortest_path(
                from_id,
                to_id,
                direction=direction,
                max_depth=max_depth,
                type=type,
            )
        )
        if not path:
            return RelationshipEvidence(
                from_id=from_id,
                to_id=to_id,
                included=False,
                reason="no_path_within_constraints",
                direction=direction,
                max_depth=max_depth,
                type=type,
            )
        if len(path) == 1:
            return RelationshipEvidence(
                from_id=from_id,
                to_id=to_id,
                included=True,
                reason="same_record",
                direction=direction,
                max_depth=max_depth,
                type=type,
                path=path,
                depth=0,
            )

        steps: list[RelationshipStep] = []
        for index in range(len(path) - 1):
            steps.extend(
                self._relationship_steps_for_hop(
                    path[index],
                    path[index + 1],
                    depth=index + 1,
                    direction=direction,
                    type=type,
                )
            )
        reason = (
            "shortest_path_within_constraints"
            if steps
            else "path_found_without_edge_metadata"
        )
        return RelationshipEvidence(
            from_id=from_id,
            to_id=to_id,
            included=True,
            reason=reason,
            direction=direction,
            max_depth=max_depth,
            type=type,
            path=path,
            steps=tuple(steps),
            depth=len(path) - 1,
        )

    def flush(self) -> None:
        if self._native_collection is not None:
            if self.path is None:
                self._native_collection.flush()
                if self._text_only_store:
                    pass  # memory-only: text-only store stays in memory
                return
            with _collection_write_lock(self.path):
                self._native_collection.flush()
                self._sync_public_manifest_index()
        if self.path is not None and self._text_only_store:
            self.path.mkdir(parents=True, exist_ok=True)
            store_path = self.path / "text_only_store.json"
            store_path.write_text(
                json.dumps(
                    {
                        tid: {
                            "text": entry["text"],
                            "metadata": entry["metadata"],
                            "source": entry["source"],
                        }
                        for tid, entry in self._text_only_store.items()
                    },
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
        elif self.path is not None and not self._text_only_store:
            pass  # No text-only records to persist

    def _sync_public_manifest_index(self) -> None:
        if (
            self.path is None
            or self.config.vector_mode != "single"
            or self.config.index != "flat"
        ):
            return
        manifest_path = self.path / "manifest.json"
        if not manifest_path.exists():
            return
        manifest = _read_json_object(manifest_path)
        if manifest.get(_PUBLIC_INDEX_MODE_KEY) == "flat":
            return
        manifest[_PUBLIC_INDEX_MODE_KEY] = "flat"
        manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    def vacuum(self) -> MaintenanceStats:
        if self.path is None:
            if not self.needs_compaction():
                return self.maintenance_stats()
            records = self._live_records_for_rebuild()
            relationships = self._live_relationships_for_rebuild(
                {str(record["id"]) for record in records}
            )
            replacement = Collection(
                name=self.name,
                config=self.config,
                path=None,
                create=True,
            )
            self._copy_live_records_into(replacement, records, relationships)
            self._native_collection = replacement._native_handle()
            self._relationship_edges = replacement._relationship_edges
            self._metadata_cache = None
            self._sparse_cache = replacement._sparse_cache.copy()
            return self.maintenance_stats()

        self.flush()
        before = self.maintenance_stats()
        if not before.needs_rebuild:
            return before

        records = self._live_records_for_rebuild()
        relationships = self._live_relationships_for_rebuild(
            {str(record["id"]) for record in records}
        )
        parent = self.path.parent
        tmp_path = Path(
            tempfile.mkdtemp(prefix=f".omendb_vector-{self.name}-vacuum-", dir=parent)
        )
        backup_path = _collection_backup_path(self.path)
        replacement = Collection(
            name=self.name,
            config=self.config,
            path=tmp_path,
            create=True,
        )
        try:
            self._copy_live_records_into(replacement, records, relationships)
            replacement.flush()
            with _collection_write_lock(self.path):
                if backup_path.exists():
                    shutil.rmtree(backup_path)
                shutil.move(str(self.path), str(backup_path))
                shutil.move(str(tmp_path), str(self.path))
        except Exception:
            if tmp_path.exists():
                shutil.rmtree(tmp_path)
            if backup_path.exists() and not self.path.exists():
                shutil.move(str(backup_path), str(self.path))
            raise
        else:
            if backup_path.exists():
                shutil.rmtree(backup_path)
            self._native_collection = None
            self.create = False
            self._relationship_edges = None
            self._metadata_cache = None
            self._sparse_cache = replacement._sparse_cache.copy()
            return self.maintenance_stats()

    def _copy_live_records_into(
        self,
        replacement: Collection,
        records: list[dict[str, Any]],
        relationships: list[_RelationshipEdge],
    ) -> None:
        for record in records:
            id = str(record["id"])
            replacement.set(
                id,
                vector=cast(list[float] | None, record.get("vector")),
                vectors=cast(list[list[float]] | None, record.get("vectors")),
                text=cast(str | None, record.get("text")),
                metadata=cast(dict[str, Any] | None, record.get("metadata")),
                source=cast(dict[str, Any] | None, record.get("source")),
            )
            sparse = self._sparse_cache.get(id)
            if sparse is not None:
                replacement.set_sparse(id, sparse)
        for edge in relationships:
            replacement.add_relationship(
                edge.from_id,
                edge.to_id,
                type=edge.type,
                weight=edge.weight,
            )

    def _live_records_for_rebuild(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for item in self._native_handle().records():
            id = str(item["id"])
            if self.config.vector_mode == "multi":
                record = self.get(
                    id,
                    include_vectors=True,
                    include_text=self.config.text,
                    include_source=True,
                )
            else:
                record = self.get(
                    id,
                    include_vector=True,
                    include_text=self.config.text,
                    include_source=True,
                )
            if record is not None:
                records.append(record)
        return records

    def _live_relationships_for_rebuild(
        self, live_ids: AbstractSet[str]
    ) -> list[_RelationshipEdge]:
        if not self.config.graph:
            return []
        return [
            edge
            for edge in sorted(
                self._relationship_edge_catalog().values(),
                key=lambda item: item.id,
            )
            if edge.from_id in live_ids and edge.to_id in live_ids
        ]

    def maintenance_stats(self) -> MaintenanceStats:
        live_count = int(self._native_handle().len())
        notes: list[str] = []
        if self.path is None:
            notes.append("memory_collection_has_no_persisted_manifest")
            return MaintenanceStats(
                collection=self.name,
                persistent=False,
                live_count=live_count,
                record_count=live_count,
                tombstone_count=None,
                stale_candidate_count=None,
                needs_rebuild=False,
                notes=tuple(notes),
            )

        manifest_path = self.path / "manifest.json"
        manifest: dict[str, Any] | None = None
        if manifest_path.exists():
            manifest = _read_json_object(manifest_path)
        else:
            notes.append("persistent_manifest_not_flushed")

        disk_bytes = _path_size_bytes(self.path)
        record_count = (
            _nonnegative_int_or_none(manifest.get("record_count"))
            if manifest is not None
            else None
        )
        tombstone_count = (
            _nonnegative_int_or_none(manifest.get("tombstone_count"))
            if manifest is not None
            else None
        )
        stale_candidate_count = tombstone_count
        estimated_reclaimable_bytes = _estimated_reclaimable_bytes(
            disk_bytes, record_count, tombstone_count
        )
        return MaintenanceStats(
            collection=self.name,
            persistent=True,
            live_count=live_count,
            record_count=record_count,
            tombstone_count=tombstone_count,
            stale_candidate_count=stale_candidate_count,
            disk_bytes=disk_bytes,
            estimated_reclaimable_bytes=estimated_reclaimable_bytes,
            store_layout=str(manifest.get("store_layout"))
            if manifest is not None
            else None,
            index_mode=str(
                manifest.get(_PUBLIC_INDEX_MODE_KEY, manifest.get("index_mode"))
            )
            if manifest is not None
            else None,
            encoding=str(manifest.get("encoding_mode", self.config.encoding))
            if manifest is not None
            else self.config.encoding,
            text_enabled=bool(manifest.get("text_enabled"))
            if manifest is not None
            else self.config.text,
            graph_enabled=bool(manifest.get("graph_enabled"))
            if manifest is not None
            else self.config.graph,
            needs_rebuild=bool(tombstone_count),
            notes=tuple(notes),
        )

    def check(self) -> CheckResult:
        """Verify this collection's on-disk state without modifying it.

        Call flush() first when you want to check the current in-memory state.
        """
        if self.path is None or not self.path.exists():
            raise EngineUnavailableError(
                "collection check requires a persisted collection"
            )
        issues = _check_collection_path(self.path)
        return CheckResult(collections_checked=1, issues=tuple(issues))

    def rebuild(self) -> MaintenanceStats:
        """Rebuild this collection from live records, resetting all
        tombstone-inflated state. Equivalent to vacuum with a fresh
        path when persisted."""
        stats = self.maintenance_stats()
        if not stats.needs_rebuild and self.path is not None:
            return stats
        return self.vacuum()

    def export_to(self, path: str | Path) -> None:
        """Export this collection to a standalone directory for backup,
        migration, or inspection."""
        if self.path is None or not self.path.exists():
            raise EngineUnavailableError(
                "collection export requires a persisted collection"
            )
        self.flush()
        export_path = Path(path)
        if export_path.exists():
            raise FileExistsError(export_path)
        shutil.copytree(self.path, export_path)

    def delete(self, id: str) -> None:
        if id in self._text_only_store:
            del self._text_only_store[id]
            return
        self._native_handle().delete(id)
        self._remove_incident_relationship_edges(id)
        self._invalidate_metadata_cache()

    def supersede(self, old_id: str, new_id: str) -> None:
        """Mark old_id as superseded by new_id. Both items must exist."""
        if old_id == new_id:
            raise ValueError("cannot supersede an item with itself")
        try:
            result = self._native_handle().supersede(old_id, new_id)
        except Exception as e:
            if "not found" in str(e):
                raise ValueError(f"replacement item '{new_id}' not found") from e
            raise
        if not result:
            raise KeyError(f"item '{old_id}' not found or already deleted")
        self._remove_incident_relationship_edges(old_id)
        self._invalidate_metadata_cache()

    def update(
        self,
        id: str,
        *,
        vector: list[float] | None = _UNSET,
        vectors: list[list[float]] | None = _UNSET,
        text: str | None = _UNSET,
        metadata: dict[str, Any] | None = _UNSET,
        source: dict[str, Any] | None = _UNSET,
    ) -> None:
        record = self.get(
            id,
            include_vector=self.config.vector_mode == "single",
            include_vectors=self.config.vector_mode == "multi",
            include_text=self.config.text,
            include_source=True,
        )
        if record is None:
            raise KeyError(id)
        merged_vector = vector if vector is not _UNSET else record.get("vector")
        merged_vectors = vectors if vectors is not _UNSET else record.get("vectors")
        if merged_vector is None and merged_vectors is None:
            raise ValueError("update requires vector or vectors")
        merged_text = text if text is not _UNSET else record.get("text")
        merged_metadata = metadata if metadata is not _UNSET else record.get("metadata")
        merged_source = source if source is not _UNSET else record.get("source")
        self.delete(id)
        self.set(
            id,
            vector=merged_vector,
            vectors=merged_vectors,
            text=merged_text,
            metadata=merged_metadata,
            source=merged_source,
        )

    def _relationship_steps_for_hop(
        self,
        from_id: str,
        to_id: str,
        *,
        depth: int,
        direction: GraphDirection,
        type: str | None,
    ) -> list[RelationshipStep]:
        steps: list[RelationshipStep] = []
        for edge in sorted(
            self._relationship_edge_catalog().values(), key=lambda item: item.id
        ):
            if type is not None and edge.type != type:
                continue
            if direction in ("out", "both") and (
                edge.from_id,
                edge.to_id,
            ) == (from_id, to_id):
                steps.append(
                    RelationshipStep(
                        edge_id=edge.id,
                        from_id=edge.from_id,
                        to_id=edge.to_id,
                        type=edge.type,
                        depth=depth,
                        traversal_direction="out",
                        weight=edge.weight,
                    )
                )
            if direction in ("in", "both") and (
                edge.from_id,
                edge.to_id,
            ) == (to_id, from_id):
                steps.append(
                    RelationshipStep(
                        edge_id=edge.id,
                        from_id=edge.from_id,
                        to_id=edge.to_id,
                        type=edge.type,
                        depth=depth,
                        traversal_direction="in",
                        weight=edge.weight,
                    )
                )
        return steps

    def _relationship_edge_catalog(
        self,
    ) -> dict[tuple[str, str, str], _RelationshipEdge]:
        if self._relationship_edges is None:
            self._relationship_edges = self._load_relationship_edge_catalog()
        return self._relationship_edges

    def _load_relationship_edge_catalog(
        self,
    ) -> dict[tuple[str, str, str], _RelationshipEdge]:
        catalog: dict[tuple[str, str, str], _RelationshipEdge] = {}
        if self.path is None:
            return catalog
        edges_path = self.path / "graph_edges.json"
        if not edges_path.exists():
            return catalog
        edges = json.loads(edges_path.read_text(encoding="utf-8"))
        if not isinstance(edges, list):
            return catalog
        for index, item in enumerate(edges):
            if not isinstance(item, dict):
                continue
            edge_record = cast(dict[str, object], item)
            from_id = str(edge_record.get("from", ""))
            to_id = str(edge_record.get("to", ""))
            edge_type = str(edge_record.get("edge_type", ""))
            if not from_id or not to_id or not edge_type:
                continue
            weight_value = edge_record.get("weight")
            weight = (
                float(cast(str | float | int, weight_value))
                if bool(edge_record.get("has_weight")) and weight_value is not None
                else None
            )
            catalog[(from_id, to_id, edge_type)] = _RelationshipEdge(
                id=index,
                from_id=from_id,
                to_id=to_id,
                type=edge_type,
                weight=weight,
            )
        return catalog

    def _remove_incident_relationship_edges(self, id: str) -> None:
        if self._relationship_edges is None:
            return
        self._relationship_edges = {
            key: edge
            for key, edge in self._relationship_edges.items()
            if edge.from_id != id and edge.to_id != id
        }
        self._renumber_relationship_edges()

    def _renumber_relationship_edges(self) -> None:
        if self._relationship_edges is None:
            return
        renumbered: dict[tuple[str, str, str], _RelationshipEdge] = {}
        for new_id, edge in enumerate(
            sorted(self._relationship_edges.values(), key=lambda item: item.id)
        ):
            renumbered[(edge.from_id, edge.to_id, edge.type)] = _RelationshipEdge(
                id=new_id,
                from_id=edge.from_id,
                to_id=edge.to_id,
                type=edge.type,
                weight=edge.weight,
            )
        self._relationship_edges = renumbered


@dataclass(slots=True)
class Database:
    path: Path | None = None
    _collections: dict[str, Collection] = field(default_factory=dict)

    def collection(
        self,
        name: str,
        *,
        config: CollectionConfig | None = None,
        create: bool = True,
    ) -> Collection:
        _check_collection_name(name)
        if name in self._collections:
            collection = self._collections[name]
            if config is not None:
                _check_collection_config_matches(name, config, collection.config)
            return collection
        collection_path = self.path / name if self.path else None
        if collection_path is not None:
            _recover_collection_path(collection_path)
        if not create and collection_path is None:
            raise KeyError(name)
        manifest_exists = (
            collection_path is not None and (collection_path / "manifest.json").exists()
        )
        if not create and collection_path is not None and not manifest_exists:
            raise FileNotFoundError(collection_path / "manifest.json")
        if config is None:
            if collection_path is None:
                raise ValueError("config is required when creating a memory collection")
            if not manifest_exists:
                raise ValueError(
                    f"config is required when creating collection {name!r}"
                )
            config = _config_from_manifest(collection_path)
        elif manifest_exists:
            persisted_config = _config_from_manifest(collection_path)
            _check_collection_config_matches(name, config, persisted_config)
        native_create = create
        if manifest_exists:
            native_create = False
        collection = Collection(
            name=name,
            config=config,
            path=collection_path,
            create=native_create,
        )
        collection._load_text_only_store()
        self._collections[name] = collection
        return collection

    def collections(self) -> list[str]:
        names = set(self._collections)
        if self.path is not None and self.path.exists():
            for child in self.path.iterdir():
                if child.is_dir() and (child / "manifest.json").exists():
                    names.add(child.name)
        return sorted(names)

    def flush(self) -> None:
        for collection in self._collections.values():
            collection.flush()

    def close(self) -> None:
        self.flush()

    def snapshot(self, path: str | Path) -> None:
        if self.path is None:
            raise EngineUnavailableError("memory databases cannot be snapshotted")
        snapshot_path = Path(path)
        if snapshot_path.exists():
            raise FileExistsError(snapshot_path)
        self.flush()
        shutil.copytree(self.path, snapshot_path)

    def import_snapshot(self, path: str | Path, *, replace: bool = False) -> None:
        if self.path is None:
            raise EngineUnavailableError("memory databases cannot import snapshots")
        snapshot_path = Path(path)
        if not snapshot_path.exists():
            raise FileNotFoundError(snapshot_path)
        if self.path.exists() and any(self.path.iterdir()):
            if not replace:
                raise FileExistsError(self.path)
            shutil.rmtree(self.path)
        shutil.copytree(snapshot_path, self.path, dirs_exist_ok=True)
        self._collections.clear()

    def check(self) -> CheckResult:
        if self.path is None:
            raise EngineUnavailableError("memory databases cannot be checked")
        issues: list[CheckIssue] = []
        if not self.path.exists():
            _add_check_issue(
                issues,
                None,
                "missing_database",
                "database path does not exist",
                self.path,
            )
            return CheckResult(collections_checked=0, issues=tuple(issues))

        collection_paths = [
            child
            for child in sorted(self.path.iterdir())
            if child.is_dir() and (child / "manifest.json").exists()
        ]
        for collection_path in collection_paths:
            issues.extend(_check_collection_path(collection_path))
        return CheckResult(
            collections_checked=len(collection_paths), issues=tuple(issues)
        )


def check(path: str | Path) -> CheckResult:
    return open(path, create=False).check()


def open(path: str | Path, *, create: bool = True) -> Database:
    db_path = Path(path)
    if not db_path.exists() and not create:
        raise FileNotFoundError(db_path)
    return Database(path=db_path)


def create(path: str | Path, *, exist_ok: bool = False) -> Database:
    db_path = Path(path)
    if db_path.exists() and not exist_ok:
        raise FileExistsError(db_path)
    db_path.mkdir(parents=True, exist_ok=True)
    return Database(path=db_path)


def memory() -> Database:
    return Database()


def _config_from_manifest(path: Path | None) -> CollectionConfig:
    if path is None:
        raise ValueError("config is required when creating a memory collection")
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    try:
        manifest_value = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"collection manifest is not valid JSON: {manifest_path}"
        ) from exc
    if not isinstance(manifest_value, dict):
        raise ValueError(f"collection manifest must be a JSON object: {manifest_path}")
    manifest = cast(dict[str, Any], manifest_value)
    is_multivector = manifest.get("store_layout") == "multivector_exact_v1"
    encoding: EncodingMode = "none"
    if is_multivector:
        manifest_encoding = str(manifest.get("encoding_mode", "none"))
        if manifest_encoding == "muvera":
            encoding = "muvera"
        elif manifest_encoding != "none":
            raise ValueError("persistent multivector encoding mismatch")
    index: IndexMode = "hnsw" if is_multivector and encoding == "muvera" else "flat"
    if not is_multivector:
        manifest_index = str(
            manifest.get(_PUBLIC_INDEX_MODE_KEY, manifest.get("index_mode", "hnsw"))
        )
        if manifest_index not in ("hnsw", "flat", "symphonyqg"):
            raise ValueError("persistent single-vector index mode mismatch")
        index = cast(IndexMode, manifest_index)
    return CollectionConfig(
        dim=int(manifest["dim"]),
        vector_mode="multi" if is_multivector else "single",
        index=index,
        metric="dot" if is_multivector else str(manifest.get("metric", "l2")),
        encoding=encoding,
        hnsw=HNSWConfig(
            m=int(manifest.get("M", 16)),
            ef_construction=int(manifest.get("ef_construction", 100)),
            ef_search=int(manifest.get("ef_search", 100)),
            alpha=float(manifest.get("alpha", 1.0)),
        ),
        text=bool(manifest.get("text_enabled", False)),
        graph=bool(manifest.get("graph_enabled", False)),
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return cast(dict[str, Any], value)


def _nonnegative_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    if parsed < 0:
        return None
    return parsed


def _path_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def _estimated_reclaimable_bytes(
    disk_bytes: int, record_count: int | None, tombstone_count: int | None
) -> int | None:
    if record_count is None or tombstone_count is None:
        return None
    if record_count <= 0 or tombstone_count <= 0:
        return 0
    return int(disk_bytes * (tombstone_count / record_count))


def _check_collection_config_matches(
    name: str, requested: CollectionConfig, persisted: CollectionConfig
) -> None:
    mismatches: list[str] = []
    _add_config_mismatch(mismatches, "dim", requested.dim, persisted.dim)
    _add_config_mismatch(
        mismatches, "vector_mode", requested.vector_mode, persisted.vector_mode
    )
    _add_config_mismatch(mismatches, "index", requested.index, persisted.index)
    _add_config_mismatch(mismatches, "metric", requested.metric, persisted.metric)
    _add_config_mismatch(mismatches, "encoding", requested.encoding, persisted.encoding)
    _add_config_mismatch(mismatches, "text", requested.text, persisted.text)
    _add_config_mismatch(mismatches, "graph", requested.graph, persisted.graph)
    if requested.vector_mode == "single" and persisted.vector_mode == "single":
        _add_config_mismatch(mismatches, "hnsw.m", requested.hnsw.m, persisted.hnsw.m)
        _add_config_mismatch(
            mismatches,
            "hnsw.ef_construction",
            requested.hnsw.ef_construction,
            persisted.hnsw.ef_construction,
        )
        _add_config_mismatch(
            mismatches, "hnsw.alpha", requested.hnsw.alpha, persisted.hnsw.alpha
        )
    if mismatches:
        raise ValueError(
            f"collection {name!r} config does not match persisted manifest: "
            + "; ".join(mismatches)
        )


def _add_config_mismatch(
    mismatches: list[str], field_name: str, requested: object, persisted: object
) -> None:
    if requested != persisted:
        mismatches.append(
            f"{field_name} requested {requested!r}, persisted {persisted!r}"
        )


def _check_collection_path(collection_path: Path) -> list[CheckIssue]:
    issues: list[CheckIssue] = []
    manifest = _read_manifest_for_check(collection_path, issues)
    if manifest is None:
        return issues
    if manifest.get("store_layout") == "multivector_exact_v1":
        _check_multivector_collection(collection_path, manifest, issues)
    else:
        _check_dense_collection(collection_path, manifest, issues)
    _check_native_reopen(collection_path, manifest, issues)
    return issues


def _read_manifest_for_check(
    collection_path: Path, issues: list[CheckIssue]
) -> dict[str, Any] | None:
    manifest_path = collection_path / "manifest.json"
    try:
        value = json.loads(manifest_path.read_text())
    except OSError as exc:
        _add_check_issue(
            issues,
            collection_path.name,
            "manifest_unreadable",
            f"cannot read manifest: {exc}",
            manifest_path,
        )
        return None
    except json.JSONDecodeError as exc:
        _add_check_issue(
            issues,
            collection_path.name,
            "manifest_invalid_json",
            f"manifest is not valid JSON: {exc}",
            manifest_path,
        )
        return None
    if not isinstance(value, dict):
        _add_check_issue(
            issues,
            collection_path.name,
            "manifest_invalid",
            "manifest must be a JSON object",
            manifest_path,
        )
        return None
    return cast(dict[str, Any], value)


def _check_dense_collection(
    collection_path: Path, manifest: dict[str, Any], issues: list[CheckIssue]
) -> None:
    required = ["records.json", "tombstones.json", "hnsw.meta"]
    if bool(manifest.get("text_enabled", False)):
        required.append("text_docs.json")
    if bool(manifest.get("graph_enabled", False)):
        required.append("graph_edges.json")
    _check_required_files(collection_path, required, issues)

    record_count = _manifest_nonnegative_int(
        collection_path, manifest, "record_count", issues
    )
    tombstone_count = _manifest_nonnegative_int(
        collection_path, manifest, "tombstone_count", issues
    )
    records = _read_json_list_for_check(
        collection_path, "records.json", "records_invalid", issues
    )
    if (
        records is not None
        and record_count is not None
        and len(records) != record_count
    ):
        _add_check_issue(
            issues,
            collection_path.name,
            "count_mismatch",
            f"records.json has {len(records)} rows; manifest has {record_count}",
            collection_path / "records.json",
        )

    tombstones = _read_json_list_for_check(
        collection_path, "tombstones.json", "tombstones_invalid", issues
    )
    if tombstones is not None:
        if record_count is not None and len(tombstones) != record_count:
            _add_check_issue(
                issues,
                collection_path.name,
                "count_mismatch",
                f"tombstones.json has {len(tombstones)} rows; manifest has "
                f"{record_count} records",
                collection_path / "tombstones.json",
            )
        deleted_count = sum(
            1
            for tombstone in tombstones
            if isinstance(tombstone, dict) and bool(tombstone.get("deleted"))
        )
        if tombstone_count is not None and deleted_count != tombstone_count:
            _add_check_issue(
                issues,
                collection_path.name,
                "count_mismatch",
                f"tombstones.json has {deleted_count} deleted rows; manifest has "
                f"{tombstone_count}",
                collection_path / "tombstones.json",
            )

    _check_dense_checksums(collection_path, manifest, issues)


def _check_multivector_collection(
    collection_path: Path, manifest: dict[str, Any], issues: list[CheckIssue]
) -> None:
    required = [
        "manifest.meta",
        "records.meta",
        "records.strings",
        "vectors.f32",
    ]
    if bool(manifest.get("text_enabled", False)):
        required.extend(["text_docs.meta", "text_docs.strings"])
    if bool(manifest.get("source_spans", False)):
        required.extend(["source_spans.meta", "source_spans.strings"])
    _check_required_files(collection_path, required, issues)

    record_count = _manifest_nonnegative_int(
        collection_path, manifest, "record_count", issues
    )
    tombstone_count = _manifest_nonnegative_int(
        collection_path, manifest, "tombstone_count", issues
    )
    dim = _manifest_nonnegative_int(collection_path, manifest, "dim", issues)

    manifest_meta = _read_mojo_ints(
        collection_path / "manifest.meta", collection_path.name, issues
    )
    if manifest_meta is not None:
        if len(manifest_meta) < _MULTIVECTOR_MANIFEST_META_WIDTH:
            _add_check_issue(
                issues,
                collection_path.name,
                "manifest_meta_invalid",
                "manifest.meta has too few fields",
                collection_path / "manifest.meta",
            )
        else:
            if record_count is not None and manifest_meta[2] != record_count:
                _add_check_issue(
                    issues,
                    collection_path.name,
                    "count_mismatch",
                    f"manifest.meta has {manifest_meta[2]} records; "
                    f"manifest.json has {record_count}",
                    collection_path / "manifest.meta",
                )
            if tombstone_count is not None and manifest_meta[3] != tombstone_count:
                _add_check_issue(
                    issues,
                    collection_path.name,
                    "count_mismatch",
                    f"manifest.meta has {manifest_meta[3]} tombstones; "
                    f"manifest.json has {tombstone_count}",
                    collection_path / "manifest.meta",
                )

    records_meta = _read_mojo_ints(
        collection_path / "records.meta", collection_path.name, issues
    )
    if records_meta is not None:
        if len(records_meta) % _MULTIVECTOR_RECORD_META_WIDTH != 0:
            _add_check_issue(
                issues,
                collection_path.name,
                "records_meta_invalid",
                "records.meta length is not a whole record metadata table",
                collection_path / "records.meta",
            )
        else:
            _check_multivector_record_counts(
                collection_path,
                records_meta,
                dim,
                record_count,
                tombstone_count,
                issues,
            )

    if bool(manifest.get("source_spans", False)):
        source_meta = _read_mojo_ints(
            collection_path / "source_spans.meta", collection_path.name, issues
        )
        if (
            source_meta is not None
            and record_count is not None
            and len(source_meta) // _MULTIVECTOR_SOURCE_META_WIDTH != record_count
        ):
            _add_check_issue(
                issues,
                collection_path.name,
                "count_mismatch",
                "source_spans.meta count does not match manifest record_count",
                collection_path / "source_spans.meta",
            )

    if bool(manifest.get("text_enabled", False)):
        text_meta = _read_mojo_ints(
            collection_path / "text_docs.meta", collection_path.name, issues
        )
        if text_meta is not None and len(text_meta) % _MULTIVECTOR_TEXT_META_WIDTH != 0:
            _add_check_issue(
                issues,
                collection_path.name,
                "text_meta_invalid",
                "text_docs.meta length is not a whole text metadata table",
                collection_path / "text_docs.meta",
            )


def _check_multivector_record_counts(
    collection_path: Path,
    records_meta: list[int],
    dim: int | None,
    record_count: int | None,
    tombstone_count: int | None,
    issues: list[CheckIssue],
) -> None:
    actual_record_count = len(records_meta) // _MULTIVECTOR_RECORD_META_WIDTH
    if record_count is not None and actual_record_count != record_count:
        _add_check_issue(
            issues,
            collection_path.name,
            "count_mismatch",
            f"records.meta has {actual_record_count} rows; manifest has {record_count}",
            collection_path / "records.meta",
        )
    deleted_count = 0
    expected_vectors = 0
    for row in range(actual_record_count):
        base = row * _MULTIVECTOR_RECORD_META_WIDTH
        if records_meta[base + 3] == 1:
            deleted_count += 1
        expected_vectors += records_meta[base + 1]
    if tombstone_count is not None and deleted_count != tombstone_count:
        _add_check_issue(
            issues,
            collection_path.name,
            "count_mismatch",
            f"records.meta has {deleted_count} deleted rows; manifest has "
            f"{tombstone_count}",
            collection_path / "records.meta",
        )

    vector_value_count = _file_element_count(
        collection_path / "vectors.f32",
        _MOJO_F32_BYTES,
        "vectors_invalid",
        collection_path.name,
        issues,
    )
    if dim is not None and vector_value_count is not None:
        expected_values = expected_vectors * dim
        if vector_value_count != expected_values:
            _add_check_issue(
                issues,
                collection_path.name,
                "count_mismatch",
                f"vectors.f32 has {vector_value_count} floats; records.meta expects "
                f"{expected_values}",
                collection_path / "vectors.f32",
            )


def _check_required_files(
    collection_path: Path, relative_names: list[str], issues: list[CheckIssue]
) -> None:
    for relative_name in relative_names:
        file_path = collection_path / relative_name
        if not file_path.is_file():
            _add_check_issue(
                issues,
                collection_path.name,
                "required_file_missing",
                f"required file is missing: {relative_name}",
                file_path,
            )


def _check_dense_checksums(
    collection_path: Path, manifest: dict[str, Any], issues: list[CheckIssue]
) -> None:
    checksums = manifest.get("checksums")
    if not isinstance(checksums, dict):
        _add_check_issue(
            issues,
            collection_path.name,
            "checksums_missing",
            "dense manifest is missing checksums",
            collection_path / "manifest.json",
        )
        return
    for relative_name, expected in checksums.items():
        if not isinstance(relative_name, str) or not isinstance(expected, str):
            _add_check_issue(
                issues,
                collection_path.name,
                "checksums_invalid",
                "checksum entries must map file names to sha256 strings",
                collection_path / "manifest.json",
            )
            continue
        file_path = collection_path / relative_name
        if not file_path.is_file():
            _add_check_issue(
                issues,
                collection_path.name,
                "checksum_file_missing",
                f"checksummed file is missing: {relative_name}",
                file_path,
            )
            continue
        actual = _sha256_file(file_path)
        if actual != expected:
            _add_check_issue(
                issues,
                collection_path.name,
                "checksum_mismatch",
                f"checksum mismatch for {relative_name}",
                file_path,
            )


def _check_native_reopen(
    collection_path: Path, manifest: dict[str, Any], issues: list[CheckIssue]
) -> None:
    try:
        config = _config_from_manifest(collection_path)
        collection = Collection(
            name=collection_path.name,
            config=config,
            path=collection_path,
            create=False,
        )
        live_count = int(collection._native_handle().len())
    except Exception as exc:
        _add_check_issue(
            issues,
            collection_path.name,
            "native_open_failed",
            f"native reopen failed: {exc}",
            collection_path,
        )
        return

    record_count = _manifest_nonnegative_int(
        collection_path, manifest, "record_count", issues
    )
    tombstone_count = _manifest_nonnegative_int(
        collection_path, manifest, "tombstone_count", issues
    )
    if record_count is not None and tombstone_count is not None:
        expected_live_count = record_count - tombstone_count
        if live_count != expected_live_count:
            _add_check_issue(
                issues,
                collection_path.name,
                "count_mismatch",
                f"native live count is {live_count}; manifest expects "
                f"{expected_live_count}",
                collection_path,
            )


def _manifest_nonnegative_int(
    collection_path: Path,
    manifest: dict[str, Any],
    key: str,
    issues: list[CheckIssue],
) -> int | None:
    value = manifest.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        _add_check_issue(
            issues,
            collection_path.name,
            "manifest_invalid",
            f"manifest field {key!r} must be a non-negative integer",
            collection_path / "manifest.json",
        )
        return None
    return value


def _read_json_list_for_check(
    collection_path: Path,
    relative_name: str,
    code: str,
    issues: list[CheckIssue],
) -> list[Any] | None:
    file_path = collection_path / relative_name
    if not file_path.exists():
        return None
    try:
        value = json.loads(file_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        _add_check_issue(
            issues,
            collection_path.name,
            code,
            f"{relative_name} is not readable JSON: {exc}",
            file_path,
        )
        return None
    if not isinstance(value, list):
        _add_check_issue(
            issues,
            collection_path.name,
            code,
            f"{relative_name} must contain a JSON list",
            file_path,
        )
        return None
    return cast(list[Any], value)


def _read_mojo_ints(
    file_path: Path, collection_name: str, issues: list[CheckIssue]
) -> list[int] | None:
    if not file_path.exists():
        return None
    try:
        data = file_path.read_bytes()
    except OSError as exc:
        _add_check_issue(
            issues,
            collection_name,
            "binary_metadata_unreadable",
            f"cannot read {file_path.name}: {exc}",
            file_path,
        )
        return None
    if len(data) % _MOJO_INT_BYTES != 0:
        _add_check_issue(
            issues,
            collection_name,
            "binary_metadata_invalid",
            f"{file_path.name} byte length is not a whole Mojo Int array",
            file_path,
        )
        return None
    count = len(data) // _MOJO_INT_BYTES
    if count == 0:
        return []
    return list(struct.unpack(f"<{count}q", data))


def _file_element_count(
    file_path: Path,
    element_size: int,
    code: str,
    collection_name: str,
    issues: list[CheckIssue],
) -> int | None:
    if not file_path.exists():
        return None
    try:
        byte_count = file_path.stat().st_size
    except OSError as exc:
        _add_check_issue(
            issues,
            collection_name,
            code,
            f"cannot stat {file_path.name}: {exc}",
            file_path,
        )
        return None
    if byte_count % element_size != 0:
        _add_check_issue(
            issues,
            collection_name,
            code,
            f"{file_path.name} byte length is not divisible by {element_size}",
            file_path,
        )
        return None
    return byte_count // element_size


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _add_check_issue(
    issues: list[CheckIssue],
    collection: str | None,
    code: str,
    message: str,
    path: Path | None = None,
) -> None:
    issues.append(
        CheckIssue(
            collection=collection,
            code=code,
            message=message,
            path=str(path) if path is not None else None,
        )
    )


def _metadata_from_native(value: Any) -> dict[str, Any] | None:
    return None if value is None else json.loads(str(value))


def _source_to_native(source: dict[str, Any] | None) -> str | None:
    if source is None:
        return None
    _validate_source(source)
    return json.dumps(source, separators=(",", ":"))


def _source_from_native(value: Any) -> dict[str, Any] | None:
    return None if value is None else json.loads(str(value))


def _validate_source(source: dict[str, Any]) -> None:
    if not isinstance(source, dict):
        raise ValueError("source must be a JSON object")
    if not any(
        isinstance(source.get(key), str) and source.get(key)
        for key in ("path", "uri", "url")
    ):
        raise ValueError("source requires one of 'path', 'uri', or 'url'")
    _check_optional_span_start_end(source, "line", one_based=True)
    _check_optional_span_start_end(source, "page", one_based=True)
    _check_optional_span_start_end(source, "byte", one_based=False)


def _check_optional_span_start_end(
    source: dict[str, Any], prefix: str, *, one_based: bool
) -> None:
    start_key = f"{prefix}_start"
    end_key = f"{prefix}_end"
    start = source.get(start_key)
    end = source.get(end_key)
    floor = 1 if one_based else 0
    if start is not None:
        _check_source_int(start_key, start, floor)
    if end is not None:
        _check_source_int(end_key, end, floor)
    if start is not None and end is not None and int(end) < int(start):
        raise ValueError(f"source {end_key} must be >= {start_key}")


def _check_source_int(name: str, value: Any, floor: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < floor:
        raise ValueError(f"source {name} must be an integer >= {floor}")


def _check_graph_direction(direction: GraphDirection) -> None:
    if direction not in ("out", "in", "both"):
        raise ValueError("direction must be 'out', 'in', or 'both'")


def _candidate_count(k: int, constrained: bool) -> int:
    _check_positive_int("k", k)
    if not constrained:
        return k
    return max(k * 16, k + 32)


def _check_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _check_search_mode(mode: SearchMode | None) -> None:
    if mode is not None and mode not in ("vector", "text", "hybrid"):
        raise ValueError("mode must be 'vector', 'text', or 'hybrid'")


def _check_collection_name(name: str) -> None:
    if not name:
        raise ValueError("collection name must not be empty")
    if name in (".", ".."):
        raise ValueError("collection name must not be '.' or '..'")
    if "/" in name or "\\" in name:
        raise ValueError("collection name must not contain path separators")


def _apply_filter(
    results: list[SearchResult],
    filter: dict[str, Any] | None,
    k: int,
    include_ids: AbstractSet[str] | None = None,
    exclude_ids: AbstractSet[str] | None = None,
) -> list[SearchResult]:
    if filter is None and include_ids is None and exclude_ids is None:
        return results[:k]
    return [
        result
        for result in results
        if _matches_constraints(
            result.id, result.metadata, filter, include_ids, exclude_ids
        )
    ][:k]


def _with_explanation(
    results: list[SearchResult], explanation: dict[str, Any]
) -> list[SearchResult]:
    return [
        SearchResult(
            id=result.id,
            score=result.score,
            distance=result.distance,
            metadata=result.metadata,
            source=result.source,
            text=result.text,
            explanation=explanation,
            evidence=_search_evidence_from_explanation(
                explanation,
                score=result.score,
                distance=result.distance,
                source=result.source,
            ),
        )
        for result in results
    ]


def _search_evidence_from_explanation(
    explanation: dict[str, Any],
    *,
    score: float,
    distance: float | None,
    source: dict[str, Any] | None,
    relationship: RelationshipEvidence | None = None,
) -> SearchEvidence:
    method = str(explanation.get("method", "unknown"))
    candidate_source = explanation.get("candidate_source")
    if candidate_source is None:
        candidate_source = method
    candidate_k_value = explanation.get("candidate_k")
    ef_search_value = explanation.get("ef_search")
    score_parts = _score_parts_from_explanation(
        method, explanation, score=score, distance=distance
    )
    return SearchEvidence(
        method=method,
        candidate_source=str(candidate_source),
        candidate_k=int(candidate_k_value)
        if isinstance(candidate_k_value, int)
        else None,
        ef_search=int(ef_search_value) if isinstance(ef_search_value, int) else None,
        filter=cast(dict[str, Any] | None, explanation.get("filter")),
        score_parts=score_parts,
        rerank=_rerank_from_explanation(method, explanation),
        source=source,
        relationship=relationship,
        raw=explanation,
    )


def _score_parts_from_explanation(
    method: str,
    explanation: dict[str, Any],
    *,
    score: float,
    distance: float | None,
) -> dict[str, Any]:
    if method == "rrf":
        return {
            "score": score,
            "vector": explanation.get("vector"),
            "text": explanation.get("text"),
            "hybrid_alpha": explanation.get("hybrid_alpha"),
            "rrf_k": explanation.get("rrf_k"),
        }
    if method == "maxsim":
        return {"score": score, "maxsim": explanation.get("maxsim")}
    if distance is not None:
        return {"score": score, "distance": distance}
    return {"score": score}


def _rerank_from_explanation(method: str, explanation: dict[str, Any]) -> str | None:
    if method == "rrf":
        return "rrf"
    if method == "maxsim":
        return "maxsim"
    if explanation.get("candidate_source") == "exact_filter_allowlist":
        return "exact_allowlist"
    return None


def _maxsim_score(
    query_vectors: list[list[float]], doc_vectors: list[list[float]]
) -> float:
    total = 0.0
    for query in query_vectors:
        best = float("-inf")
        for doc in doc_vectors:
            dot = sum(q * d for q, d in zip(query, doc, strict=True))
            if dot > best:
                best = dot
        total += best
    return total


def _matches_constraints(
    id: str,
    metadata: dict[str, Any] | None,
    filter: dict[str, Any] | None,
    include_ids: AbstractSet[str] | None,
    exclude_ids: AbstractSet[str] | None,
) -> bool:
    if include_ids is not None and id not in include_ids:
        return False
    if exclude_ids is not None and id in exclude_ids:
        return False
    if filter is not None and not _matches_filter(metadata, filter):
        return False
    return True


def _matches_filter(metadata: dict[str, Any] | None, filter: dict[str, Any]) -> bool:
    # Handle $and/$or/$not at top level
    if "$and" in filter:
        return all(_matches_filter(metadata, sub) for sub in filter["$and"])
    if "$or" in filter:
        return any(_matches_filter(metadata, sub) for sub in filter["$or"])
    if "$not" in filter:
        return not _matches_filter(metadata, filter["$not"])
    for key, expected in filter.items():
        exists = metadata is not None and key in metadata
        actual = metadata[key] if exists and metadata is not None else None
        if isinstance(expected, dict):
            if "$exists" in expected and exists != bool(expected["$exists"]):
                return False
            if "$in" in expected and actual not in expected["$in"]:
                return False
            for op in ("$gt", "$lt", "$gte", "$lte"):
                if op in expected:
                    if not exists or actual is None:
                        return False
                    threshold = expected[op]
                    if op == "$gt" and not (actual > threshold):
                        return False
                    if op == "$lt" and not (actual < threshold):
                        return False
                    if op == "$gte" and not (actual >= threshold):
                        return False
                    if op == "$lte" and not (actual <= threshold):
                        return False
            if "$ne" in expected and actual == expected["$ne"]:
                return False
            unknown = set(expected) - {"$exists", "$in", "$gt", "$lt", "$gte", "$lte", "$ne"}
            if unknown:
                raise ValueError(f"unsupported filter operator: {sorted(unknown)[0]}")
        elif not exists or actual != expected:
            return False
    return True


def _validate_filter(filter: dict[str, Any] | None) -> None:
    if filter is None:
        return
    # Handle $and/$or/$not at top level
    if "$and" in filter:
        for sub in filter["$and"]:
            _validate_filter(sub)
        return
    if "$or" in filter:
        for sub in filter["$or"]:
            _validate_filter(sub)
        return
    if "$not" in filter:
        _validate_filter(filter["$not"])
        return
    for expected in filter.values():
        if isinstance(expected, dict):
            unknown = set(expected) - {"$exists", "$in", "$gt", "$lt", "$gte", "$lte", "$ne"}
            if unknown:
                raise ValueError(f"unsupported filter operator: {sorted(unknown)[0]}")


def _normalize_id_constraint(name: str, ids: list[str] | None) -> set[str] | None:
    if ids is None:
        return None
    normalized: set[str] = set()
    for id in ids:
        if not isinstance(id, str) or id == "":
            raise ValueError(f"{name} must contain non-empty strings")
        normalized.add(id)
    return normalized
