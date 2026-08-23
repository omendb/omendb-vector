from __future__ import annotations

import json
import shutil
import sys
from typing import Any, cast

import pytest

import omendb_vector


def test_import_and_collection_config() -> None:
    config = omendb_vector.CollectionConfig(dim=2)
    db = omendb_vector.memory()
    docs = db.collection("docs", config=config)

    assert docs.name == "docs"
    assert docs.config.index == "hnsw"
    assert db.collections() == ["docs"]


def test_python_boundary_validation() -> None:
    with pytest.raises(ValueError, match="hnsw.m"):
        omendb_vector.HNSWConfig(m=0)
    with pytest.raises(ValueError, match="ef_search"):
        omendb_vector.HNSWConfig(ef_search=0)
    with pytest.raises(ValueError, match="alpha"):
        omendb_vector.HNSWConfig(alpha=0.0)

    db = omendb_vector.memory()
    with pytest.raises(ValueError, match="collection name"):
        db.collection("", config=omendb_vector.CollectionConfig(dim=2))
    with pytest.raises(ValueError, match="path separators"):
        db.collection("../outside", config=omendb_vector.CollectionConfig(dim=2))

    docs = db.collection("docs", config=omendb_vector.CollectionConfig(dim=2))
    with pytest.raises(ValueError, match="k"):
        docs.search_vector([0.0, 0.0], k=0)
    with pytest.raises(ValueError, match="ef"):
        docs.search_vector([0.0, 0.0], ef=0)


def test_collection_config_and_missing_path_errors_are_explicit(tmp_path) -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")

    db_path = tmp_path / "db"
    db = omendb_vector.create(db_path)
    with pytest.raises(ValueError, match="config is required"):
        db.collection("missing")
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        db.collection("missing", create=False)

    docs = db.collection("docs", config=omendb_vector.CollectionConfig(dim=2))
    docs.set("origin", vector=[0.0, 0.0])
    docs.flush()

    reopened = omendb_vector.open(db_path, create=False)
    with pytest.raises(ValueError, match="config does not match persisted manifest"):
        reopened.collection("docs", config=omendb_vector.CollectionConfig(dim=128))
    with pytest.raises(ValueError, match="text requested True, persisted False"):
        reopened.collection("docs", config=omendb_vector.CollectionConfig(dim=2, text=True))

    cached = reopened.collection("docs", create=False)
    assert reopened.collection("docs", config=omendb_vector.CollectionConfig(dim=2)) is cached
    with pytest.raises(ValueError, match="dim requested 128, persisted 2"):
        reopened.collection("docs", config=omendb_vector.CollectionConfig(dim=128))


def test_corrupt_collection_manifest_has_clear_python_error(tmp_path) -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")

    db_path = tmp_path / "db"
    docs = omendb_vector.create(db_path).collection(
        "docs", config=omendb_vector.CollectionConfig(dim=2)
    )
    docs.set("origin", vector=[0.0, 0.0])
    docs.flush()
    (db_path / "docs" / "manifest.json").write_text("{")

    with pytest.raises(ValueError, match="manifest is not valid JSON"):
        omendb_vector.open(db_path, create=False).collection("docs", create=False)


def test_open_recovers_collection_backup_after_interrupted_swap(tmp_path) -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")

    db_path = tmp_path / "db"
    docs = omendb_vector.create(db_path).collection(
        "docs", config=omendb_vector.CollectionConfig(dim=2, text=True)
    )
    docs.set("old", vector=[0.0, 0.0], text="last committed snapshot")
    docs.flush()

    collection_path = db_path / "docs"
    backup_path = db_path / "docs.bak"
    shutil.move(str(collection_path), str(backup_path))

    reopened = omendb_vector.open(db_path, create=False).collection("docs", create=False)

    assert collection_path.exists()
    assert not backup_path.exists()
    assert [row.id for row in reopened.search_text("snapshot", k=1)] == ["old"]


def test_open_prefers_current_collection_when_backup_is_stale(tmp_path) -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")

    db_path = tmp_path / "db"
    docs = omendb_vector.create(db_path).collection(
        "docs", config=omendb_vector.CollectionConfig(dim=2, text=True)
    )
    docs.set("old", vector=[0.0, 0.0], text="old snapshot")
    docs.flush()
    old_snapshot = tmp_path / "old-snapshot"
    shutil.copytree(db_path / "docs", old_snapshot)

    docs.set("new", vector=[1.0, 0.0], text="current snapshot")
    docs.delete("old")
    docs.flush()
    shutil.copytree(old_snapshot, db_path / "docs.bak")

    reopened = omendb_vector.open(db_path, create=False).collection("docs", create=False)

    assert [row.id for row in reopened.search_text("current", k=1)] == ["new"]
    assert reopened.search_text("old", k=1) == []


def test_flush_reports_collection_write_lock_conflict(tmp_path) -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")

    try:
        import fcntl
    except ImportError:
        pytest.skip("advisory lock test requires fcntl")

    db_path = tmp_path / "db"
    docs = omendb_vector.create(db_path).collection(
        "docs", config=omendb_vector.CollectionConfig(dim=2)
    )
    docs.set("origin", vector=[0.0, 0.0])
    lock_path = db_path / "docs.write.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with pytest.raises(omendb_vector.StoreBusyError, match="write already"):
                docs.flush()
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def test_search_requires_explicit_mode_for_text_and_vector() -> None:
    docs = omendb_vector.memory().collection("docs", config=omendb_vector.CollectionConfig(dim=2))

    with pytest.raises(ValueError, match="mode is required"):
        docs.search(vector=[0.0, 1.0], text="query")


def test_search_validates_mode_filter_and_explain_before_native_dispatch() -> None:
    docs = omendb_vector.memory().collection(
        "docs", config=omendb_vector.CollectionConfig(dim=2, text=True)
    )
    source = {"path": "guide.md", "line_start": 1, "line_end": 2}
    docs.set("guide", vector=[0.0, 1.0], text="query guide", source=source)

    with pytest.raises(ValueError, match="mode must be"):
        docs.search(vector=[0.0, 1.0], mode=cast(Any, "graph"))
    with pytest.raises(ValueError, match="unsupported filter operator"):
        docs.search_vector([0.0, 1.0], filter={"kind": {"$regex": "task"}})
    vector_result = docs.search_vector([0.0, 1.0], explain=True)[0]
    vector_explanation = vector_result.explanation
    assert vector_explanation is not None
    assert vector_explanation == {
        "method": "hnsw_l2",
        "candidate_k": 10,
        "ef_search": 100,
        "filter": None,
    }
    assert vector_result.evidence == omendb_vector.SearchEvidence(
        method="hnsw_l2",
        candidate_source="hnsw_l2",
        candidate_k=10,
        ef_search=100,
        filter=None,
        score_parts={"score": vector_result.score, "distance": vector_result.distance},
        source=source,
        raw=vector_explanation,
    )
    text_result = docs.search_text("query", explain=True)[0]
    text_explanation = text_result.explanation
    assert text_explanation is not None
    assert text_explanation == {
        "method": "bm25",
        "candidate_k": 10,
        "filter": None,
    }
    assert text_result.evidence == omendb_vector.SearchEvidence(
        method="bm25",
        candidate_source="bm25",
        candidate_k=10,
        filter=None,
        score_parts={"score": text_result.score},
        source=source,
        raw=text_explanation,
    )


def test_set_update_delete_vector_lifecycle() -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")

    docs = omendb_vector.memory().collection("docs", config=omendb_vector.CollectionConfig(dim=2))

    docs.set("a", vector=[0.0, 0.0], metadata={"version": 1})
    docs.update("a", vector=[10.0, 0.0], metadata={"version": 2})
    docs.set("b", vector=[1.0, 0.0])
    docs.delete("a")

    assert docs.get("a") is None
    assert docs.get("b", include_vector=True) == {
        "id": "b",
        "metadata": None,
        "vector": [1.0, 0.0],
        "source": None,
    }
    assert [row.id for row in docs.search_vector([0.0, 0.0], k=5)] == ["b"]
    # Test timestamps
    result = docs.get("b", include_timestamps=True)
    assert "created_at" in result
    assert "updated_at" in result
    assert result["created_at"] > 0

    with pytest.raises(KeyError):
        docs.update("missing", vector=[0.0, 0.0])
    with pytest.raises(ValueError, match="require vector"):
        docs.set("missing")


def test_set_duplicate_id_is_explicit_upsert() -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")

    docs = omendb_vector.memory().collection(
        "docs", config=omendb_vector.CollectionConfig(dim=2, text=True)
    )
    docs.set(
        "guide",
        vector=[0.0, 0.0],
        text="old install guide",
        metadata={"version": 1},
    )
    docs.set(
        "guide",
        vector=[10.0, 0.0],
        text="new install guide",
        metadata={"version": 2},
        source={"path": "docs/new.md", "line_start": 1, "line_end": 2},
    )

    assert docs.get(
        "guide",
        include_vector=True,
        include_text=True,
        include_source=True,
    ) == {
        "id": "guide",
        "metadata": {"version": 2},
        "source": {"path": "docs/new.md", "line_start": 1, "line_end": 2},
        "vector": [10.0, 0.0],
        "text": "new install guide",
    }
    # Test timestamps
    result = docs.get("guide", include_timestamps=True)
    assert "created_at" in result
    assert "updated_at" in result
    assert docs.search_text("old", k=1) == []
    assert [row.id for row in docs.search_text("new", k=1)] == ["guide"]

    docs.delete("missing")
    assert docs.get("missing") is None


def test_vector_set_and_search_when_native_extension_is_available() -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")

    docs = omendb_vector.memory().collection("docs", config=omendb_vector.CollectionConfig(dim=2))

    docs.set("origin", vector=[0.0, 0.0], metadata={"kind": "anchor"})
    docs.set_many(
        [
            {"id": "right", "vector": [1.0, 0.0], "metadata": {"kind": "axis"}},
            {"id": "up", "vector": [0.0, 1.0]},
        ]
    )

    results = docs.search_vector([0.1, 0.0], k=2)

    assert [result.id for result in results] == ["origin", "right"]
    assert results[0].metadata == {"kind": "anchor"}

    assert docs.get("missing") is None
    assert docs.get("origin") == {
        "id": "origin",
        "metadata": {"kind": "anchor"},
        "vector": [0.0, 0.0],
        "source": None,
    }
    assert docs.get("right", include_vector=True) == {
        "id": "right",
        "metadata": {"kind": "axis"},
        "vector": [1.0, 0.0],
        "source": None,
    }


def test_source_span_lifecycle_search_and_reopen(tmp_path) -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")

    with pytest.raises(ValueError, match="source requires"):
        omendb_vector.memory().collection("bad", config=omendb_vector.CollectionConfig(dim=2)).set(
            "bad", vector=[0.0, 0.0], source={"line_start": 1}
        )

    source = {
        "repo": "omendb-vector",
        "path": "docs/install.md",
        "line_start": 3,
        "line_end": 9,
    }
    db_path = tmp_path / "db"
    docs = omendb_vector.create(db_path).collection(
        "docs", config=omendb_vector.CollectionConfig(dim=2, text=True)
    )

    docs.set(
        "install",
        vector=[0.0, 0.0],
        text="install source span",
        metadata={"kind": "guide"},
        source=source,
    )

    assert docs.search_text("install", k=1)[0].source == source
    assert docs.get("install", include_vector=True, include_source=True, include_text=False) == {
        "id": "install",
        "metadata": {"kind": "guide"},
        "source": source,
        "vector": [0.0, 0.0],
    }

    docs.update("install", vector=[1.0, 0.0], metadata={"kind": "updated"})
    assert docs.get("install", include_vector=True, include_source=True, include_text=False) == {
        "id": "install",
        "metadata": {"kind": "updated"},
        "source": source,
        "vector": [1.0, 0.0],
    }

    docs.flush()
    reopened = omendb_vector.open(db_path, create=False).collection("docs", create=False)
    assert reopened.search_vector([0.0, 0.0], k=1)[0].source == source
    assert reopened.get("install", include_source=True, include_text=False, include_vector=False) == {
        "id": "install",
        "metadata": {"kind": "updated"},
        "source": source,
    }

    reopened.update("install", source=None)
    cleared = reopened.get("install", include_source=True)
    assert cleared is not None
    assert cleared["source"] is None


def test_text_only_items_without_placeholder_vectors(tmp_path) -> None:
    """Test that text-only items can be stored without vectors."""
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")

    db_path = tmp_path / "db"
    docs = omendb_vector.create(db_path).collection(
        "docs", config=omendb_vector.CollectionConfig(dim=2, text=True)
    )

    # Add text-only item (no vector)
    docs.set(
        "readme",
        text="This is the readme file",
        metadata={"type": "doc"},
        source={"path": "README.md", "line_start": 1, "line_end": 5},
    )

    # Add item with vector and text
    docs.set(
        "guide",
        vector=[1.0, 0.0],
        text="This is the guide",
        metadata={"type": "doc"},
    )

    # Text search should find both
    results = docs.search_text("readme", k=1)
    assert len(results) == 1
    assert results[0].id == "readme"

    results = docs.search_text("guide", k=1)
    assert len(results) == 1
    assert results[0].id == "guide"

    # Vector search should only find items with vectors
    results = docs.search_vector([1.0, 0.0], k=10)
    assert len(results) == 1
    assert results[0].id == "guide"

    # Get should work for both
    readme = docs.get("readme", include_source=True)
    assert readme is not None
    assert readme["id"] == "readme"
    assert readme["source"] == {"path": "README.md", "line_start": 1, "line_end": 5}

    guide = docs.get("guide", include_vector=True)
    assert guide is not None
    assert guide["id"] == "guide"
    assert guide["vector"] == [1.0, 0.0]

    # Flush and reopen
    docs.flush()
    reopened = omendb_vector.open(db_path, create=False).collection("docs", create=False)

    # Text search should still find both
    results = reopened.search_text("readme", k=1)
    assert len(results) == 1
    assert results[0].id == "readme"

    # Vector search should still only find items with vectors
    results = reopened.search_vector([1.0, 0.0], k=10)
    assert len(results) == 1
    assert results[0].id == "guide"

    # Delete text-only item
    reopened.delete("readme")

    # Verify deletion
    assert reopened.get("readme") is None
    results = reopened.search_text("readme", k=1)
    assert len(results) == 0


def test_vector_search_dim128_when_native_extension_is_available() -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")

    docs = omendb_vector.memory().collection("docs", config=omendb_vector.CollectionConfig(dim=128))
    origin = [0.0] * 128
    right = [0.0] * 128
    right[0] = 1.0
    far = [10.0] * 128

    docs.set("origin", vector=origin, metadata={"kind": "anchor"})
    docs.set("right", vector=right)
    docs.set("far", vector=far)

    results = docs.search_vector([0.1, *([0.0] * 127)], k=2)

    assert [result.id for result in results] == ["origin", "right"]
    assert docs.get("origin", include_vector=True) == {
        "id": "origin",
        "metadata": {"kind": "anchor"},
        "vector": origin,
        "source": None,
    }


def test_vector_search_dim384_when_native_extension_is_available() -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")

    docs = omendb_vector.memory().collection("docs", config=omendb_vector.CollectionConfig(dim=384))
    origin = [0.0] * 384
    right = [0.0] * 384
    right[0] = 1.0

    docs.set("origin", vector=origin)
    docs.set("right", vector=right)

    results = docs.search_vector([0.1, *([0.0] * 383)], k=2)

    assert [result.id for result in results] == ["origin", "right"]


def test_persistent_vector_collection_reopens(tmp_path) -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")

    db_path = tmp_path / "db"
    docs = omendb_vector.create(db_path).collection(
        "docs", config=omendb_vector.CollectionConfig(dim=2)
    )
    docs.set("origin", vector=[0.0, 0.0], metadata={"kind": "anchor"})
    docs.set("right", vector=[1.0, 0.0])
    docs.flush()

    reopened = omendb_vector.open(db_path).collection(
        "docs", config=omendb_vector.CollectionConfig(dim=2)
    )
    results = reopened.search_vector([0.05, 0.0], k=2)

    assert [result.id for result in results] == ["origin", "right"]
    assert results[0].metadata == {"kind": "anchor"}

    inferred_config = omendb_vector.open(db_path, create=False).collection(
        "docs", create=False
    )
    assert inferred_config.config.dim == 2
    assert omendb_vector.open(db_path, create=False).collections() == ["docs"]


def test_dense_flat_vector_collection_uses_exact_search_and_reopens(
    tmp_path,
) -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")

    db_path = tmp_path / "db"
    docs = omendb_vector.create(db_path).collection(
        "docs", config=omendb_vector.CollectionConfig(dim=2, index="flat", text=True)
    )
    docs.set(
        "origin",
        vector=[0.0, 0.0],
        text="anchor point",
        metadata={"kind": "anchor"},
    )
    docs.set("right", vector=[1.0, 0.0], text="edge point", metadata={"kind": "edge"})
    docs.set("up", vector=[0.0, 2.0], text="vertical point")

    results = docs.search_vector([0.2, 0.0], k=3, explain=True)
    assert [row.id for row in results] == ["origin", "right", "up"]
    assert results[0].evidence is not None
    assert results[0].evidence.method == "exact_l2"
    assert results[0].evidence.candidate_source == "flat_all_live"

    with pytest.raises(ValueError, match="ef is only supported for hnsw"):
        docs.search_vector([0.2, 0.0], k=1, ef=10)

    hybrid = docs.search_hybrid(vector=[0.2, 0.0], text="edge", k=2, explain=True)
    assert [row.id for row in hybrid] == ["right", "origin"]
    assert hybrid[0].evidence is not None
    assert hybrid[0].evidence.method == "rrf"

    with pytest.raises(ValueError, match="ef is only supported for hnsw"):
        docs.search_hybrid(vector=[0.2, 0.0], text="edge", k=1, ef=10)

    docs.flush()
    manifest = json.loads((db_path / "docs" / "manifest.json").read_text())
    assert manifest["index_mode"] == "hnsw"
    assert manifest["public_index_mode"] == "flat"

    reopened = omendb_vector.open(db_path, create=False).collection("docs", create=False)
    assert reopened.config.index == "flat"
    assert [row.id for row in reopened.search_vector([0.2, 0.0], k=2)] == [
        "origin",
        "right",
    ]
    assert omendb_vector.check(db_path).ok

    with pytest.raises(ValueError, match="index requested 'hnsw', persisted 'flat'"):
        omendb_vector.open(db_path, create=False).collection(
            "docs", config=omendb_vector.CollectionConfig(dim=2, index="hnsw")
        )


def test_database_check_passes_for_dense_and_muvera_collections(tmp_path) -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")

    db_path = tmp_path / "db"
    db = omendb_vector.create(db_path)
    docs = db.collection("docs", config=omendb_vector.CollectionConfig(dim=2, text=True))
    docs.set(
        "install",
        vector=[0.0, 0.0],
        text="install guide",
        source={"path": "docs/install.md", "line_start": 1, "line_end": 4},
    )
    code = db.collection(
        "code",
        config=omendb_vector.CollectionConfig(
            dim=2,
            vector_mode="multi",
            index="hnsw",
            metric="dot",
            encoding="muvera",
            text=True,
        ),
    )
    code.set(
        "fn",
        vectors=[[1.0, 0.0], [0.0, 1.0]],
        text="function body",
        source={"path": "src/fn.mojo", "line_start": 2, "line_end": 9},
    )

    db.flush()
    result = db.check()

    assert result.ok
    assert result.collections_checked == 2
    assert result.issues == ()
    assert omendb_vector.check(db_path).ok


def test_maintenance_stats_expose_tombstones_and_reopen(tmp_path) -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")

    db_path = tmp_path / "db"
    docs = omendb_vector.create(db_path).collection(
        "docs", config=omendb_vector.CollectionConfig(dim=2, text=True, graph=True)
    )
    docs.set(
        "keep-a",
        vector=[0.0, 0.0],
        text="keep",
        source={"path": "docs/a.md", "line_start": 1, "line_end": 2},
    )
    docs.set("remove", vector=[1.0, 0.0], text="remove")
    docs.set("keep-b", vector=[2.0, 0.0], text="keep")
    docs.add_relationship("keep-a", "keep-b", type="related")
    docs.flush()

    initial = docs.maintenance_stats()
    assert initial == omendb_vector.MaintenanceStats(
        collection="docs",
        persistent=True,
        live_count=3,
        record_count=3,
        tombstone_count=0,
        stale_candidate_count=0,
        disk_bytes=initial.disk_bytes,
        estimated_reclaimable_bytes=0,
        store_layout="single_segment_v1",
        index_mode="hnsw",
        encoding="none",
        text_enabled=True,
        graph_enabled=True,
        needs_rebuild=False,
    )
    assert initial.disk_bytes is not None and initial.disk_bytes > 0

    docs.delete("remove")
    docs.flush()

    stats = docs.maintenance_stats()
    assert stats.live_count == 2
    assert stats.record_count == 3
    assert stats.tombstone_count == 1
    assert stats.stale_candidate_count == 1
    assert stats.needs_rebuild
    assert stats.estimated_reclaimable_bytes is not None
    assert stats.estimated_reclaimable_bytes > 0

    reopened = omendb_vector.open(db_path, create=False).collection("docs", create=False)
    reopened_stats = reopened.maintenance_stats()
    assert reopened_stats.live_count == 2
    assert reopened_stats.record_count == 3
    assert reopened_stats.tombstone_count == 1
    assert reopened_stats.stale_candidate_count == 1
    assert reopened_stats.needs_rebuild

    vacuumed = reopened.vacuum()
    assert vacuumed.live_count == 2
    assert vacuumed.record_count == 2
    assert vacuumed.tombstone_count == 0
    assert vacuumed.stale_candidate_count == 0
    assert vacuumed.estimated_reclaimable_bytes == 0
    assert not vacuumed.needs_rebuild
    assert reopened.get(
        "keep-a",
        include_vector=True,
        include_text=True,
        include_source=True,
    ) == {
        "id": "keep-a",
        "metadata": None,
        "source": {"path": "docs/a.md", "line_start": 1, "line_end": 2},
        "vector": [0.0, 0.0],
        "text": "keep",
    }
    assert [row.id for row in reopened.search_text("remove", k=3)] == []
    assert reopened.neighbors("keep-a", type="related") == ["keep-b"]

    post_vacuum = omendb_vector.open(db_path, create=False).collection("docs", create=False)
    post_stats = post_vacuum.maintenance_stats()
    assert post_stats.live_count == 2
    assert post_stats.record_count == 2
    assert post_stats.tombstone_count == 0
    assert post_stats.stale_candidate_count == 0
    assert not post_stats.needs_rebuild
    assert post_vacuum.neighbors("keep-a", type="related") == ["keep-b"]


def test_vacuum_works_on_memory_collection() -> None:
    """vacuum() on memory collections should compact deleted records."""
    docs = omendb_vector.memory().collection("docs", config=omendb_vector.CollectionConfig(dim=2))
    docs.set("a", vector=[0.0, 0.0])
    docs.set("b", vector=[1.0, 0.0])
    docs.set("c", vector=[0.0, 1.0])
    docs.set("d", vector=[1.0, 1.0])

    # No deletions → no compaction needed
    assert not docs.needs_compaction()
    stats = docs.vacuum()
    assert stats.live_count == 4
    assert not docs.needs_compaction()

    # Delete 2 of 4 → 50%% tombstones → needs compaction
    docs.delete("a")
    docs.delete("b")
    assert docs.needs_compaction()

    stats = docs.vacuum()
    assert stats.live_count == 2
    assert not docs.needs_compaction()

    # Deleted items are gone from results
    results = docs.search_vector([0.0, 0.0], k=10)
    result_ids = {r.id for r in results}
    assert "a" not in result_ids
    assert "b" not in result_ids
    assert "c" in result_ids
    assert "d" in result_ids


def test_database_check_reports_dense_checksum_corruption(tmp_path) -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")

    db_path = tmp_path / "db"
    docs = omendb_vector.create(db_path).collection(
        "docs", config=omendb_vector.CollectionConfig(dim=2)
    )
    docs.set("origin", vector=[0.0, 0.0])
    docs.flush()
    records_path = db_path / "docs" / "records.json"
    records_path.write_text(records_path.read_text().replace("origin", "corrupt", 1))

    result = omendb_vector.check(db_path)

    assert not result.ok
    assert result.collections_checked == 1
    assert any(
        issue.collection == "docs" and issue.code == "checksum_mismatch"
        for issue in result.issues
    )
    assert any(
        issue.collection == "docs" and issue.code == "native_open_failed"
        for issue in result.issues
    )


def test_database_check_reports_multivector_missing_sidecar(tmp_path) -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")

    db_path = tmp_path / "db"
    docs = omendb_vector.create(db_path).collection(
        "code",
        config=omendb_vector.CollectionConfig(
            dim=2,
            vector_mode="multi",
            index="flat",
            metric="dot",
            text=True,
        ),
    )
    docs.set("fn", vectors=[[1.0, 0.0]], text="function")
    docs.flush()
    (db_path / "code" / "source_spans.strings").unlink()

    result = omendb_vector.open(db_path, create=False).check()

    assert not result.ok
    assert result.collections_checked == 1
    assert any(
        issue.collection == "code" and issue.code == "required_file_missing"
        for issue in result.issues
    )
    assert any(
        issue.collection == "code" and issue.code == "native_open_failed"
        for issue in result.issues
    )


def test_database_check_rejects_memory_databases() -> None:
    with pytest.raises(omendb_vector.EngineUnavailableError, match="memory databases"):
        omendb_vector.memory().check()


def test_persistent_dim128_collection_reopens(tmp_path) -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")

    db_path = tmp_path / "db"
    origin = [0.0] * 128
    right = [0.0] * 128
    right[0] = 1.0
    docs = omendb_vector.create(db_path).collection(
        "docs", config=omendb_vector.CollectionConfig(dim=128)
    )
    docs.set("origin", vector=origin)
    docs.set("right", vector=right)
    docs.flush()

    reopened = omendb_vector.open(db_path, create=False).collection("docs", create=False)

    assert reopened.config.dim == 128
    assert [row.id for row in reopened.search_vector([0.1, *([0.0] * 127)], k=2)] == [
        "origin",
        "right",
    ]


def test_collection_config_hnsw_reaches_native_manifest(tmp_path) -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")

    db_path = tmp_path / "db"
    docs = omendb_vector.create(db_path).collection(
        "docs",
        config=omendb_vector.CollectionConfig(
            dim=128,
            hnsw=omendb_vector.HNSWConfig(m=32, ef_construction=400, ef_search=512),
        ),
    )
    docs.set("origin", vector=[0.0] * 128)
    docs.flush()

    manifest = json.loads((db_path / "docs" / "manifest.json").read_text())

    assert manifest["M"] == 32
    assert manifest["ef_construction"] == 400


def test_multivector_add_search_get_and_reopen(tmp_path) -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")

    db_path = tmp_path / "db"
    docs = omendb_vector.create(db_path).collection(
        "code",
        config=omendb_vector.CollectionConfig(
            dim=2,
            vector_mode="multi",
            index="flat",
            metric="dot",
            text=True,
        ),
    )
    docs.set(
        "fn",
        vectors=[[1.0, 0.0], [0.0, 1.0]],
        text="async function store",
        metadata={"kind": "function"},
        source={"path": "src/fn.mojo", "line_start": 10, "line_end": 20},
    )
    docs.set(
        "trait",
        vectors=[[0.0, 1.0], [0.0, 0.8]],
        text="trait implementation",
    )

    semantic = docs.search_vectors([[1.0, 0.0], [0.0, 1.0]], k=2, explain=True)
    assert [row.id for row in semantic] == ["fn", "trait"]
    assert semantic[0].score == 2.0
    assert semantic[0].distance is None
    assert semantic[0].source == {
        "path": "src/fn.mojo",
        "line_start": 10,
        "line_end": 20,
    }
    semantic_explanation = semantic[0].explanation
    assert semantic_explanation is not None
    assert semantic_explanation == {"method": "maxsim", "maxsim": 2.0}
    assert semantic[0].evidence == omendb_vector.SearchEvidence(
        method="maxsim",
        candidate_source="maxsim",
        score_parts={"score": 2.0, "maxsim": 2.0},
        rerank="maxsim",
        source=semantic[0].source,
        raw=semantic_explanation,
    )

    hybrid = docs.search(
        vectors=[[1.0, 0.0]],
        text="function",
        mode="hybrid",
        k=2,
        explain=True,
    )
    assert [row.id for row in hybrid] == ["fn"]
    assert hybrid[0].explanation == {
        "method": "maxsim",
        "maxsim": 1.0,
        "candidate_source": "bm25",
    }
    assert hybrid[0].evidence is not None
    assert hybrid[0].evidence.method == "maxsim"
    assert hybrid[0].evidence.candidate_source == "bm25"
    assert hybrid[0].evidence.rerank == "maxsim"
    with pytest.raises(omendb_vector.EngineUnavailableError, match="dense RRF"):
        docs.search_hybrid(
            vectors=[[1.0, 0.0]],
            text="function",
            hybrid_alpha=0.9,
        )
    with pytest.raises(omendb_vector.EngineUnavailableError, match="dense HNSW"):
        docs.search_hybrid(vectors=[[1.0, 0.0]], text="function", ef=100)

    assert docs.get("fn", include_vectors=True, include_source=True, include_text=False) == {
        "id": "fn",
        "metadata": {"kind": "function"},
        "source": {"path": "src/fn.mojo", "line_start": 10, "line_end": 20},
        "vectors": [[1.0, 0.0], [0.0, 1.0]],
    }

    docs.flush()
    reopened = omendb_vector.open(db_path, create=False).collection("code", create=False)
    assert reopened.config.vector_mode == "multi"
    assert reopened.config.index == "flat"
    assert reopened.config.metric == "dot"
    assert [row.id for row in reopened.search_vectors([[1.0, 0.0]], k=1)] == ["fn"]


def test_muvera_multivector_config_and_reopen(tmp_path) -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")

    with pytest.raises(ValueError, match="index='hnsw'"):
        omendb_vector.CollectionConfig(
            dim=2,
            vector_mode="multi",
            index="flat",
            metric="dot",
            encoding="muvera",
        )
    with pytest.raises(ValueError, match="index='flat'"):
        omendb_vector.CollectionConfig(
            dim=2,
            vector_mode="multi",
            index="hnsw",
            metric="dot",
            encoding="none",
        )

    db_path = tmp_path / "db"
    docs = omendb_vector.create(db_path).collection(
        "code",
        config=omendb_vector.CollectionConfig(
            dim=2,
            vector_mode="multi",
            index="hnsw",
            metric="dot",
            encoding="muvera",
            text=True,
        ),
    )
    for i in range(55):
        if i == 0:
            docs.set(
                f"doc-{i}",
                vectors=[[1.0, 0.0], [0.0, 1.0]],
                text="target code path",
                source={"path": "src/target.mojo", "line_start": 1, "line_end": 8},
            )
        else:
            docs.set(
                f"doc-{i}",
                vectors=[[0.0, 1.0], [0.0, 0.8]],
                text="background code path",
            )

    before = docs.search_vectors([[1.0, 0.0], [0.0, 1.0]], k=1)
    assert [row.id for row in before] == ["doc-0"]
    assert before[0].source == {
        "path": "src/target.mojo",
        "line_start": 1,
        "line_end": 8,
    }
    docs.flush()

    manifest = json.loads((db_path / "code" / "manifest.json").read_text())
    assert manifest["encoding_mode"] == "muvera"
    assert manifest["candidate_index_mode"] == "fde_hnsw_rebuild_v0"
    assert manifest["candidate_k"] == 500

    reopened = omendb_vector.open(db_path, create=False).collection("code", create=False)
    assert reopened.config.vector_mode == "multi"
    assert reopened.config.index == "hnsw"
    assert reopened.config.metric == "dot"
    assert reopened.config.encoding == "muvera"
    reopened_results = reopened.search_vectors([[1.0, 0.0]], k=1)
    assert [row.id for row in reopened_results] == ["doc-0"]
    assert reopened_results[0].source == {
        "path": "src/target.mojo",
        "line_start": 1,
        "line_end": 8,
    }


def test_vacuum_compacts_muvera_multivector_collection(tmp_path) -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")

    db_path = tmp_path / "db"
    docs = omendb_vector.create(db_path).collection(
        "code",
        config=omendb_vector.CollectionConfig(
            dim=2,
            vector_mode="multi",
            index="hnsw",
            metric="dot",
            encoding="muvera",
            text=True,
        ),
    )
    docs.set(
        "keep",
        vectors=[[1.0, 0.0], [0.0, 1.0]],
        text="keep code path",
        source={"path": "src/keep.mojo", "line_start": 1, "line_end": 3},
    )
    docs.set(
        "stale",
        vectors=[[0.0, 1.0], [0.0, 0.5]],
        text="stale code path",
    )
    docs.flush()
    docs.delete("stale")
    docs.flush()

    stats = docs.maintenance_stats()
    assert stats.live_count == 1
    assert stats.record_count == 2
    assert stats.tombstone_count == 1
    assert stats.stale_candidate_count == 1
    assert stats.needs_rebuild

    vacuumed = docs.vacuum()
    assert vacuumed.live_count == 1
    assert vacuumed.record_count == 1
    assert vacuumed.tombstone_count == 0
    assert vacuumed.stale_candidate_count == 0
    assert not vacuumed.needs_rebuild
    assert docs.get(
        "keep",
        include_vectors=True,
        include_text=True,
        include_source=True,
    ) == {
        "id": "keep",
        "metadata": None,
        "source": {"path": "src/keep.mojo", "line_start": 1, "line_end": 3},
        "vectors": [[1.0, 0.0], [0.0, 1.0]],
        "text": "keep code path",
    }
    assert docs.get("stale") is None
    assert [row.id for row in docs.search_vectors([[1.0, 0.0]], k=1)] == ["keep"]

    manifest = json.loads((db_path / "code" / "manifest.json").read_text())
    assert manifest["encoding_mode"] == "muvera"
    assert manifest["candidate_index_mode"] == "fde_hnsw_rebuild_v0"
    assert manifest["candidate_k"] == 500
    assert manifest["record_count"] == 1
    assert manifest["tombstone_count"] == 0

    reopened = omendb_vector.open(db_path, create=False).collection("code", create=False)
    reopened_stats = reopened.maintenance_stats()
    assert reopened_stats.record_count == 1
    assert reopened_stats.tombstone_count == 0
    assert not reopened_stats.needs_rebuild
    assert [
        row.id
        for row in reopened.search_hybrid(
            vectors=[[0.0, 1.0]],
            text="stale",
            k=3,
        )
    ] == []


def test_multivector_dim128_exact_and_muvera_bindings(tmp_path) -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")

    def unit(dim_index: int) -> list[float]:
        values = [0.0] * 128
        values[dim_index] = 1.0
        return values

    db_path = tmp_path / "db"
    db = omendb_vector.create(db_path)
    exact = db.collection(
        "colbert_exact",
        config=omendb_vector.CollectionConfig(
            dim=128,
            vector_mode="multi",
            index="flat",
            metric="dot",
        ),
    )
    exact.set("target", vectors=[unit(3), unit(7)])
    exact.set("other", vectors=[unit(9), unit(11)])
    assert [row.id for row in exact.search_vectors([unit(3), unit(7)], k=1)] == [
        "target"
    ]

    muvera = db.collection(
        "colbert_muvera",
        config=omendb_vector.CollectionConfig(
            dim=128,
            vector_mode="multi",
            index="hnsw",
            metric="dot",
            encoding="muvera",
        ),
    )
    muvera.set(
        "target",
        vectors=[unit(3), unit(7)],
        source={"uri": "hf://datasets/BeIR/scifact/target", "doc_id": "target"},
    )
    muvera.set("other", vectors=[unit(9), unit(11)])
    assert [row.id for row in muvera.search_vectors([unit(3), unit(7)], k=1)] == [
        "target"
    ]
    muvera.flush()

    reopened = omendb_vector.open(db_path, create=False).collection(
        "colbert_muvera", create=False
    )
    rows = reopened.search_vectors([unit(3), unit(7)], k=1)
    assert [row.id for row in rows] == ["target"]
    assert rows[0].source == {
        "uri": "hf://datasets/BeIR/scifact/target",
        "doc_id": "target",
    }


def test_muvera_multivector_lifecycle_filters_stale_candidates(tmp_path) -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")

    db_path = tmp_path / "db"
    docs = omendb_vector.create(db_path).collection(
        "code",
        config=omendb_vector.CollectionConfig(
            dim=2,
            vector_mode="multi",
            index="hnsw",
            metric="dot",
            encoding="muvera",
            text=True,
        ),
    )
    for i in range(55):
        docs.set(
            f"fill-{i}",
            vectors=[[0.0, 1.0], [0.0, 0.8]],
            text="background path",
            metadata={"state": "fill"},
        )
    docs.set(
        "active",
        vectors=[[1.0, 0.0], [0.0, 1.0]],
        text="target live",
        metadata={"state": "live"},
    )
    docs.set(
        "stale",
        vectors=[[1.0, 0.0], [0.0, 1.0]],
        text="target stale",
        metadata={"state": "old"},
    )

    docs.update(
        "stale",
        vectors=[[0.0, 1.0], [0.0, 0.5]],
        text="archived stale",
        metadata={"state": "superseded"},
    )

    query = [[1.0, 0.0], [0.0, 1.0]]
    filtered_vector = docs.search_vectors(
        query, k=1, filter={"state": "live"}, explain=True
    )
    assert [row.id for row in filtered_vector] == ["active"]
    assert filtered_vector[0].explanation["method"] == "maxsim"
    assert filtered_vector[0].explanation["maxsim"] == 2.0
    assert filtered_vector[0].explanation["filter"]["strategy"] == "exact_allowlist"
    assert filtered_vector[0].explanation["filter"]["eligible_count"] == 1
    assert "selectivity" in filtered_vector[0].explanation["filter"]
    assert [
        row.id
        for row in docs.search_hybrid(
            vectors=query, text="target", k=5, filter={"state": "live"}
        )
    ] == ["active"]

    docs.delete("stale")
    docs.flush()
    reopened = omendb_vector.open(db_path, create=False).collection("code", create=False)

    unfiltered = [row.id for row in reopened.search_vectors(query, k=5)]
    assert "stale" not in unfiltered
    assert [
        row.id
        for row in reopened.search_hybrid(
            vectors=query, text="target", k=5, filter={"state": "live"}
        )
    ] == ["active"]
    assert (
        reopened.search_hybrid(
            vectors=query, text="target", k=5, filter={"state": "old"}
        )
        == []
    )


def test_multivector_requires_vectors_shape() -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")

    docs = omendb_vector.memory().collection(
        "code",
        config=omendb_vector.CollectionConfig(
            dim=2, vector_mode="multi", index="flat", metric="dot"
        ),
    )
    with pytest.raises(ValueError, match="require vectors"):
        docs.set("bad", vector=[1.0, 0.0])
    with pytest.raises(ValueError, match="dimension"):
        docs.set("bad", vectors=[[1.0]])
    with pytest.raises(ValueError, match="include_vectors"):
        docs.get("bad", include_vector=True)
    with pytest.raises(omendb_vector.EngineUnavailableError, match="relationships"):
        docs.set(
            "bad",
            vectors=[[1.0, 0.0]],
            relationships=[{"to": "other", "type": "calls"}],
        )
    with pytest.raises(ValueError, match="graph"):
        omendb_vector.CollectionConfig(
            dim=2, vector_mode="multi", index="flat", metric="dot", graph=True
        )


def test_multivector_search_rejects_dense_only_shapes() -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")

    docs = omendb_vector.memory().collection(
        "code",
        config=omendb_vector.CollectionConfig(
            dim=2,
            vector_mode="multi",
            index="flat",
            metric="dot",
            text=True,
        ),
    )
    docs.set("fn", vectors=[[1.0, 0.0]], text="function")

    with pytest.raises(ValueError, match="use vectors"):
        docs.search(vector=[1.0, 0.0], mode="vector")
    with pytest.raises(ValueError, match="use vectors"):
        docs.search_hybrid(vector=[1.0, 0.0], text="function")
    with pytest.raises(omendb_vector.EngineUnavailableError, match="text-only"):
        docs.search_text("function")


def test_multivector_filters_preserve_explanations() -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")

    docs = omendb_vector.memory().collection(
        "code",
        config=omendb_vector.CollectionConfig(
            dim=2,
            vector_mode="multi",
            index="flat",
            metric="dot",
            text=True,
        ),
    )
    docs.set(
        "fn",
        vectors=[[1.0, 0.0]],
        text="async function",
        metadata={"kind": "function"},
    )
    docs.set(
        "trait",
        vectors=[[1.0, 0.0]],
        text="async trait",
        metadata={"kind": "trait"},
    )

    vector_results = docs.search_vectors(
        [[1.0, 0.0]], k=2, filter={"kind": "function"}, explain=True
    )
    assert [row.id for row in vector_results] == ["fn"]
    assert vector_results[0].explanation["method"] == "maxsim"
    assert vector_results[0].explanation["maxsim"] == 1.0
    assert vector_results[0].explanation["filter"]["strategy"] == "exact_allowlist"
    assert vector_results[0].explanation["filter"]["eligible_count"] == 1
    assert "selectivity" in vector_results[0].explanation["filter"]

    hybrid_results = docs.search_hybrid(
        vectors=[[1.0, 0.0]],
        text="async",
        k=2,
        filter={"kind": "function"},
        explain=True,
    )
    assert [row.id for row in hybrid_results] == ["fn"]
    assert hybrid_results[0].explanation == {
        "method": "maxsim",
        "maxsim": 1.0,
        "candidate_source": "bm25",
        "candidate_k": 34,
        "filter": {"strategy": "all_live_bm25_candidates"},
    }


def test_single_vector_search_rejects_multivector_shapes() -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")

    docs = omendb_vector.memory().collection("docs", config=omendb_vector.CollectionConfig(dim=2))

    with pytest.raises(ValueError, match="use vector"):
        docs.search(vectors=[[1.0, 0.0]], mode="vector")
    with pytest.raises(ValueError, match="use vector"):
        docs.search_hybrid(vectors=[[1.0, 0.0]], text="query")


def test_native_duplicate_set_replaces_single_vector_record() -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")

    docs = omendb_vector.memory().collection("docs", config=omendb_vector.CollectionConfig(dim=2))
    native = docs._native_handle()
    native.set("dup", [0.0, 0.0], None, None)
    native.set("dup", [10.0, 0.0], None, None)

    record = docs.get("dup", include_vector=True)
    assert record is not None
    assert record["vector"] == [10.0, 0.0]

    results = docs.search_vector([10.0, 0.0], k=10)
    assert [r.id for r in results] == ["dup"]


def test_native_duplicate_set_replaces_multivector_record() -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")

    docs = omendb_vector.memory().collection(
        "docs",
        config=omendb_vector.CollectionConfig(
            dim=2, vector_mode="multi", index="flat", metric="dot"
        ),
    )
    native = docs._native_handle()
    native.set("dup", [[1.0, 0.0]], None, None)
    native.set("dup", [[0.0, 1.0]], None, None)

    record = docs.get("dup", include_vectors=True)
    assert record is not None
    assert record["vectors"] == [[0.0, 1.0]]

    results = docs.search_vectors([[0.0, 1.0]], k=10)
    assert [r.id for r in results] == ["dup"]


def test_multivector_update_preserves_omitted_text_and_metadata() -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")

    docs = omendb_vector.memory().collection(
        "code",
        config=omendb_vector.CollectionConfig(
            dim=2,
            vector_mode="multi",
            index="flat",
            metric="dot",
            text=True,
        ),
    )
    docs.set(
        "fn",
        vectors=[[1.0, 0.0]],
        text="async function",
        metadata={"kind": "function"},
    )

    docs.update("fn", vectors=[[0.0, 1.0]])

    assert docs.get("fn", include_vectors=True, include_text=True) == {
        "id": "fn",
        "metadata": {"kind": "function"},
        "vectors": [[0.0, 1.0]],
        "text": "async function",
        "source": None,
    }
    assert [
        row.id for row in docs.search_hybrid(vectors=[[0.0, 1.0]], text="async")
    ] == ["fn"]

    docs.update("fn", metadata={"kind": "updated"})
    assert docs.get("fn", include_vectors=True, include_text=True) == {
        "id": "fn",
        "metadata": {"kind": "updated"},
        "vectors": [[0.0, 1.0]],
        "text": "async function",
        "source": None,
    }


def test_multivector_update_can_clear_text_and_metadata() -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")

    docs = omendb_vector.memory().collection(
        "code",
        config=omendb_vector.CollectionConfig(
            dim=2,
            vector_mode="multi",
            index="flat",
            metric="dot",
            text=True,
        ),
    )
    docs.set(
        "fn",
        vectors=[[1.0, 0.0]],
        text="async function",
        metadata={"kind": "function"},
    )

    docs.update("fn", text=None, metadata=None)

    assert docs.get("fn", include_vectors=True, include_text=True) == {
        "id": "fn",
        "metadata": None,
        "vectors": [[1.0, 0.0]],
        "text": None,
        "source": None,
    }
    assert docs.search_hybrid(vectors=[[1.0, 0.0]], text="async") == []


def test_text_set_search_and_reopen(tmp_path) -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")

    db_path = tmp_path / "db"
    docs = omendb_vector.create(db_path).collection(
        "docs", config=omendb_vector.CollectionConfig(dim=2, text=True)
    )
    docs.set(
        "install",
        vector=[0.0, 0.0],
        text="Install OmenDB Vector with uv",
        metadata={"kind": "guide"},
    )
    docs.set("other", vector=[1.0, 0.0], text="Unrelated note")

    results = docs.search_text("install", k=1)
    assert results[0].id == "install"
    assert results[0].metadata == {"kind": "guide"}
    assert results[0].distance is None

    docs.flush()
    reopened = omendb_vector.open(db_path, create=False).collection("docs", create=False)
    assert reopened.config.text is True
    assert reopened.search_text("install", k=1)[0].id == "install"
    assert reopened.get("install", include_vector=True, include_text=False) == {
        "id": "install",
        "metadata": {"kind": "guide"},
        "vector": [0.0, 0.0],
        "source": None,
    }


def test_update_delete_text_and_reopen(tmp_path) -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")

    db_path = tmp_path / "db"
    docs = omendb_vector.create(db_path).collection(
        "docs", config=omendb_vector.CollectionConfig(dim=2, text=True)
    )
    docs.set("old", vector=[0.0, 0.0], text="install old", metadata={"version": 1})
    docs.update(
        "old",
        vector=[10.0, 0.0],
        text="install replacement",
        metadata={"version": 2},
    )
    docs.set("keep", vector=[1.0, 0.0], text="install keep")
    docs.delete("old")
    docs.flush()

    reopened = omendb_vector.open(db_path, create=False).collection("docs", create=False)

    assert reopened.get("old") is None
    assert [row.id for row in reopened.search_text("install", k=5)] == ["keep"]
    assert [row.id for row in reopened.search_vector([0.0, 0.0], k=5)] == ["keep"]


def test_single_vector_lifecycle_matrix_reopen_filters_hybrid_graph(
    tmp_path,
) -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")

    db_path = tmp_path / "db"
    docs = omendb_vector.create(db_path).collection(
        "docs", config=omendb_vector.CollectionConfig(dim=2, text=True, graph=True)
    )
    docs.set("neighbor", vector=[1.0, 0.0], text="neighbor")
    docs.set(
        "active",
        vector=[0.0, 0.0],
        text="target live",
        metadata={"state": "live"},
        relationships=[{"to": "neighbor", "type": "links"}],
    )
    docs.set(
        "stale",
        vector=[0.0, 0.0],
        text="target stale",
        metadata={"state": "old"},
    )
    docs.add_relationship("stale", "neighbor", type="links")

    docs.update(
        "stale",
        vector=[10.0, 0.0],
        text="archived stale",
        metadata={"state": "superseded"},
    )

    assert docs.neighbors("stale", type="links") == []
    assert not docs.has_path("stale", "neighbor", type="links")
    assert docs.neighbors("active", type="links") == ["neighbor"]
    assert [row.id for row in docs.search_vector([0.0, 0.0], k=1)] == ["active"]
    assert [
        row.id for row in docs.search_text("target", k=5, filter={"state": "live"})
    ] == ["active"]
    assert [
        row.id
        for row in docs.search_hybrid(
            vector=[0.0, 0.0],
            text="target",
            k=5,
            filter={"state": "live"},
        )
    ] == ["active"]
    assert docs.search_text("target", k=5, filter={"state": "old"}) == []

    docs.delete("stale")
    docs.flush()
    reopened = omendb_vector.open(db_path, create=False).collection("docs", create=False)

    assert reopened.get("stale") is None
    assert reopened.neighbors("stale", type="links") == []
    assert not reopened.has_path("stale", "neighbor", type="links")
    assert reopened.neighbors("active", type="links") == ["neighbor"]
    assert [row.id for row in reopened.search_vector([0.0, 0.0], k=1)] == ["active"]
    assert [
        row.id
        for row in reopened.search_hybrid(
            vector=[0.0, 0.0],
            text="target",
            k=5,
            filter={"state": "live"},
        )
    ] == ["active"]
    assert reopened.search_text("target", k=5, filter={"state": "old"}) == []


def test_update_preserves_omitted_text_and_metadata() -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")

    docs = omendb_vector.memory().collection(
        "docs", config=omendb_vector.CollectionConfig(dim=2, text=True)
    )
    docs.set("a", vector=[0.0, 0.0], text="install guide", metadata={"version": 1})

    docs.update("a", vector=[1.0, 0.0])

    assert docs.get("a", include_vector=True, include_text=True) == {
        "id": "a",
        "metadata": {"version": 1},
        "vector": [1.0, 0.0],
        "text": "install guide",
        "source": None,
    }
    assert [row.id for row in docs.search_text("install", k=1)] == ["a"]

    docs.update("a", metadata={"version": 2})
    assert docs.get("a", include_vector=True, include_text=True) == {
        "id": "a",
        "metadata": {"version": 2},
        "vector": [1.0, 0.0],
        "text": "install guide",
        "source": None,
    }


def test_metadata_filters_for_vector_and_text() -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")

    docs = omendb_vector.memory().collection(
        "docs", config=omendb_vector.CollectionConfig(dim=2, text=True)
    )
    docs.set(
        "project-task",
        vector=[0.0, 0.0],
        text="Install with uv",
        metadata={"scope": "project", "kind": "task"},
    )
    docs.set(
        "design-decision",
        vector=[0.1, 0.0],
        text="Install default is HNSW",
        metadata={"scope": "design", "kind": "decision"},
    )
    docs.set("plain", vector=[0.2, 0.0], text="Install note")

    vector_results = docs.search_vector(
        [0.1, 0.0], k=2, filter={"scope": {"$in": ["project", "design"]}}
    )
    assert set(result.id for result in vector_results) == {
        "design-decision",
        "project-task",
    }

    text_results = docs.search_text(
        "install", k=2, filter={"kind": "task", "scope": {"$exists": True}}
    )
    assert [result.id for result in text_results] == ["project-task"]


def test_filtered_text_uses_all_live_candidates_with_evidence() -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")

    docs = omendb_vector.memory().collection(
        "docs", config=omendb_vector.CollectionConfig(dim=2, text=True)
    )
    for i in range(80):
        docs.set(
            f"common-{i}",
            vector=[float(i), 0.0],
            text="install shared term",
            metadata={"kind": "common"},
        )
    docs.set(
        "rare",
        vector=[100.0, 0.0],
        text="install shared term",
        metadata={"kind": "rare"},
    )

    results = docs.search_text("install", k=1, filter={"kind": "rare"}, explain=True)

    assert [result.id for result in results] == ["rare"]
    assert results[0].explanation == {
        "method": "bm25",
        "candidate_k": 81,
        "filter": {"strategy": "all_live_candidates"},
    }


def test_include_exclude_id_constraints_for_search_modes() -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")

    docs = omendb_vector.memory().collection(
        "docs", config=omendb_vector.CollectionConfig(dim=2, text=True)
    )
    docs.set("allowed", vector=[0.0, 0.0], text="install guide")
    docs.set("blocked", vector=[0.0, 0.0], text="install guide")

    vector_results = docs.search_vector(
        [0.0, 0.0], include_ids=["allowed"], explain=True
    )
    assert [result.id for result in vector_results] == ["allowed"]
    # Only 1 eligible ID: HNSW underrun triggers exact fallback
    method = vector_results[0].explanation["method"]
    assert method in ("hnsw_filtered_l2", "exact_l2")
    f = vector_results[0].explanation["filter"]
    assert f["include_count"] == 1
    assert f["eligible_count"] == 1
    assert "selectivity" in f

    assert [
        result.id for result in docs.search_text("install", exclude_ids=["blocked"])
    ] == ["allowed"]

    hybrid_results = docs.search_hybrid(
        vector=[0.0, 0.0],
        text="install",
        include_ids=["allowed"],
        exclude_ids=["blocked"],
        explain=True,
    )
    assert [result.id for result in hybrid_results] == ["allowed"]
    assert hybrid_results[0].explanation is not None
    vf = hybrid_results[0].explanation["filter"]["vector"]
    assert vf["strategy"] == "engine_bitmap_filter"
    assert vf["include_count"] == 1
    assert vf["exclude_count"] == 1
    assert vf["eligible_count"] == 1
    assert "selectivity" in vf

    with pytest.raises(ValueError, match="include_ids"):
        docs.search_text("install", include_ids=[""])


def test_include_exclude_id_constraints_for_multivector() -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")

    docs = omendb_vector.memory().collection(
        "code",
        config=omendb_vector.CollectionConfig(
            dim=2,
            vector_mode="multi",
            index="flat",
            metric="dot",
            text=True,
        ),
    )
    docs.set("allowed", vectors=[[1.0, 0.0]], text="function allowed")
    docs.set("blocked", vectors=[[1.0, 0.0]], text="function blocked")

    vector_results = docs.search_vectors(
        [[1.0, 0.0]],
        include_ids=["allowed"],
        exclude_ids=["blocked"],
        explain=True,
    )
    assert [result.id for result in vector_results] == ["allowed"]
    assert vector_results[0].explanation["method"] == "maxsim"
    assert vector_results[0].explanation["maxsim"] == 1.0
    vf = vector_results[0].explanation["filter"]
    assert vf["strategy"] == "exact_allowlist"
    assert vf["include_count"] == 1
    assert vf["exclude_count"] == 1
    assert vf["eligible_count"] == 1
    assert "selectivity" in vf

    hybrid_results = docs.search_hybrid(
        vectors=[[1.0, 0.0]],
        text="function",
        include_ids=["allowed"],
        explain=True,
    )
    assert [result.id for result in hybrid_results] == ["allowed"]
    assert hybrid_results[0].explanation == {
        "method": "maxsim",
        "maxsim": 1.0,
        "candidate_source": "bm25",
        "candidate_k": 160,
        "filter": {"strategy": "all_live_bm25_candidates", "include_count": 1},
    }


def test_hybrid_search_uses_rrf_with_score_evidence() -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")

    docs = omendb_vector.memory().collection(
        "docs", config=omendb_vector.CollectionConfig(dim=2, text=True)
    )
    docs.set(
        "both",
        vector=[0.0, 0.0],
        text="install omendb_vector",
        metadata={"kind": "guide"},
    )
    docs.set("vector-only", vector=[0.1, 0.0], text="release notes")
    docs.set("text-only", vector=[10.0, 0.0], text="install guide")

    results = docs.search_hybrid(
        vector=[0.0, 0.0],
        text="install",
        k=3,
        explain=True,
        hybrid_alpha=0.5,
        rrf_k=60,
    )

    assert [result.id for result in results] == ["both", "text-only", "vector-only"]
    assert results[0].metadata == {"kind": "guide"}
    assert results[0].distance == 0.0
    first_explanation = results[0].explanation
    assert first_explanation is not None
    assert first_explanation["method"] == "rrf"
    assert first_explanation["rrf_k"] == 60
    assert first_explanation["hybrid_alpha"] == 0.5
    assert first_explanation["vector"] == {
        "rank": 1,
        "score": -0.0,
        "distance": 0.0,
    }
    assert first_explanation["text"]["rank"] == 1
    assert first_explanation["text"]["score"] > 0
    first_evidence = results[0].evidence
    assert first_evidence is not None
    assert first_evidence.method == "rrf"
    assert first_evidence.candidate_source == "rrf"
    assert first_evidence.rerank == "rrf"
    assert first_evidence.score_parts["vector"]["rank"] == 1
    assert first_evidence.score_parts["text"]["rank"] == 1

    text_explanation = results[1].explanation
    assert text_explanation is not None
    assert text_explanation["vector"]["rank"] == 3
    assert text_explanation["text"]["rank"] == 2

    vector_explanation = results[2].explanation
    assert vector_explanation is not None
    assert vector_explanation["vector"]["rank"] == 2
    assert vector_explanation["text"]["rank"] is None


def test_hybrid_search_requires_explicit_text_index_and_parameters() -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")

    docs = omendb_vector.memory().collection("docs", config=omendb_vector.CollectionConfig(dim=2))

    with pytest.raises(omendb_vector.EngineUnavailableError, match="config.text=True"):
        docs.search_hybrid(vector=[0.0, 0.0], text="install")
    with pytest.raises(ValueError, match="vector and text"):
        docs.search(vector=[0.0, 0.0], mode="hybrid")

    text_docs = omendb_vector.memory().collection(
        "docs", config=omendb_vector.CollectionConfig(dim=2, text=True)
    )
    text_docs.set("a", vector=[0.0, 0.0], text="install")

    with pytest.raises(ValueError, match="hybrid_alpha"):
        text_docs.search_hybrid(vector=[0.0, 0.0], text="install", hybrid_alpha=1.1)
    with pytest.raises(ValueError, match="rrf_k"):
        text_docs.search_hybrid(vector=[0.0, 0.0], text="install", rrf_k=0)


def test_hybrid_search_applies_metadata_filter() -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")

    docs = omendb_vector.memory().collection(
        "docs", config=omendb_vector.CollectionConfig(dim=2, text=True)
    )
    docs.set("task", vector=[0.0, 0.0], text="install", metadata={"kind": "task"})
    docs.set(
        "decision",
        vector=[0.1, 0.0],
        text="install",
        metadata={"kind": "decision"},
    )

    results = docs.search(
        vector=[0.0, 0.0],
        text="install",
        mode="hybrid",
        k=2,
        filter={"kind": "decision"},
        explain=True,
    )

    assert [result.id for result in results] == ["decision"]
    assert results[0].explanation is not None
    vf = results[0].explanation["filter"]["vector"]
    assert vf["strategy"] == "engine_bitmap_filter"
    assert vf["eligible_count"] == 1
    assert "selectivity" in vf
    assert results[0].explanation["filter"]["text"] == {
        "strategy": "all_live_bm25_candidates",
        "candidate_k": 34,
    }


def test_small_vector_filter_uses_exact_allowlist() -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")

    docs = omendb_vector.memory().collection("docs", config=omendb_vector.CollectionConfig(dim=2))
    for i in range(80):
        docs.set(
            f"common-{i}",
            vector=[float(i) + 100.0, 0.0],
            metadata={"kind": "common"},
        )
    docs.set("target", vector=[0.0, 0.0], metadata={"kind": "rare"})

    results = docs.search_vector([0.0, 0.0], k=1, filter={"kind": "rare"}, explain=True)

    assert [result.id for result in results] == ["target"]
    assert results[0].distance == 0.0
    assert results[0].explanation["method"] == "hnsw_filtered_bitmap"
    f = results[0].explanation["filter"]
    assert f["strategy"] == "engine_bitmap_filter"


def test_large_vector_filter_returns_nearest_matches() -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")

    docs = omendb_vector.memory().collection("docs", config=omendb_vector.CollectionConfig(dim=2))
    for i in range(2_100):
        docs.set(f"match-{i}", vector=[float(i), 0.0], metadata={"bucket": "match"})
    for i in range(2_100):
        docs.set(
            f"other-{i}",
            vector=[float(i) + 5_000.0, 0.0],
            metadata={"bucket": "other"},
        )

    results = docs.search_vector([0.0, 0.0], k=5, filter={"bucket": "match"}, ef=64)

    assert [result.id for result in results] == [
        "match-0",
        "match-1",
        "match-2",
        "match-3",
        "match-4",
    ]


def test_range_predicates() -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")

    docs = omendb_vector.memory().collection("docs", config=omendb_vector.CollectionConfig(dim=2))
    docs.set("low", vector=[0.0, 0.0], metadata={"score": 10})
    docs.set("mid", vector=[0.0, 0.0], metadata={"score": 50})
    docs.set("high", vector=[0.0, 0.0], metadata={"score": 90})
    docs.set("none", vector=[0.0, 0.0], metadata={"other": "no score field"})

    # $gt
    assert [r.id for r in docs.search_vector([0.0, 0.0], k=10, filter={"score": {"$gt": 40}})] == ["mid", "high"]

    # $gte
    assert [r.id for r in docs.search_vector([0.0, 0.0], k=10, filter={"score": {"$gte": 50}})] == ["mid", "high"]

    # $lt
    assert [r.id for r in docs.search_vector([0.0, 0.0], k=10, filter={"score": {"$lt": 60}})] == ["low", "mid"]

    # $lte
    assert [r.id for r in docs.search_vector([0.0, 0.0], k=10, filter={"score": {"$lte": 10}})] == ["low"]

    # Range combo: $gte + $lte
    assert [r.id for r in docs.search_vector([0.0, 0.0], k=10, filter={"score": {"$gte": 10, "$lte": 50}})] == ["low", "mid"]

    # Range with other filter
    assert [r.id for r in docs.search_vector([0.0, 0.0], k=10, filter={"score": {"$gt": 40, "$exists": True}})] == ["mid", "high"]

    # Missing field doesn't match range
    assert set(r.id for r in docs.search_vector([0.0, 0.0], k=10, filter={"score": {"$gt": 0}})) == {"low", "mid", "high"}

    # Unsupported operator still raises
    with pytest.raises(ValueError, match="unsupported filter operator"):
        docs.search_vector([0.0, 0.0], k=10, filter={"score": {"$regex": "x"}})


def test_snapshot_and_import(tmp_path) -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")

    db_path = tmp_path / "db"
    snapshot_path = tmp_path / "snapshot"
    imported_path = tmp_path / "imported"
    db = omendb_vector.create(db_path)
    docs = db.collection("docs", config=omendb_vector.CollectionConfig(dim=2, text=True))
    docs.set(
        "install",
        vector=[0.0, 0.0],
        text="Install",
        metadata={"kind": "guide"},
        source={"path": "docs/install.md", "line_start": 1, "line_end": 1},
    )

    db.snapshot(snapshot_path)
    imported = omendb_vector.create(imported_path)
    imported.import_snapshot(snapshot_path)

    reopened = omendb_vector.open(imported_path, create=False).collection("docs", create=False)
    assert reopened.get("install", include_vector=True, include_source=True, include_text=False) == {
        "id": "install",
        "metadata": {"kind": "guide"},
        "source": {"path": "docs/install.md", "line_start": 1, "line_end": 1},
        "vector": [0.0, 0.0],
    }
    assert reopened.search_text("install", k=1)[0].id == "install"


def test_graph_relationships_and_reopen(tmp_path) -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")

    db_path = tmp_path / "db"
    docs = omendb_vector.create(db_path).collection(
        "docs", config=omendb_vector.CollectionConfig(dim=2, graph=True)
    )
    docs.set("install", vector=[0.0, 0.0])
    docs.set("wheel", vector=[1.0, 0.0])
    docs.set("release", vector=[2.0, 0.0])

    edge_id = docs.add_relationship("install", "wheel", type="depends_on")
    assert edge_id == 0
    docs.add_relationship("wheel", "release", type="depends_on", weight=0.5)

    assert docs.neighbors("install", type="depends_on") == ["wheel"]
    assert docs.neighbors("wheel", direction="in", type="depends_on") == ["install"]
    assert docs.traverse("install", max_depth=2, type="depends_on") == [
        "install",
        "wheel",
        "release",
    ]
    assert docs.has_path("install", "release", max_depth=2, type="depends_on")
    assert docs.shortest_path("install", "release", max_depth=2) == [
        "install",
        "wheel",
        "release",
    ]

    docs.flush()
    reopened = omendb_vector.open(db_path, create=False).collection("docs", create=False)
    assert reopened.config.graph is True
    assert reopened.neighbors("install", type="depends_on") == ["wheel"]
    assert reopened.remove_relationship("wheel", "release", type="depends_on")
    assert not reopened.has_path("install", "release", max_depth=2)


def test_relationship_evidence_reports_bounded_path_and_reopen(tmp_path) -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")

    db_path = tmp_path / "db"
    docs = omendb_vector.create(db_path).collection(
        "docs", config=omendb_vector.CollectionConfig(dim=2, graph=True)
    )
    docs.set("install", vector=[0.0, 0.0])
    docs.set("wheel", vector=[1.0, 0.0])
    docs.set("release", vector=[2.0, 0.0])

    docs.add_relationship("install", "wheel", type="depends_on")
    docs.add_relationship("wheel", "release", type="depends_on", weight=0.5)

    evidence = docs.relationship_evidence(
        "install", "release", max_depth=2, type="depends_on"
    )

    assert evidence == omendb_vector.RelationshipEvidence(
        from_id="install",
        to_id="release",
        included=True,
        reason="shortest_path_within_constraints",
        direction="out",
        max_depth=2,
        type="depends_on",
        path=("install", "wheel", "release"),
        steps=(
            omendb_vector.RelationshipStep(
                edge_id=0,
                from_id="install",
                to_id="wheel",
                type="depends_on",
                depth=1,
                traversal_direction="out",
            ),
            omendb_vector.RelationshipStep(
                edge_id=1,
                from_id="wheel",
                to_id="release",
                type="depends_on",
                depth=2,
                traversal_direction="out",
                weight=0.5,
            ),
        ),
        depth=2,
    )

    missing = docs.relationship_evidence(
        "install", "release", max_depth=1, type="depends_on"
    )
    assert missing == omendb_vector.RelationshipEvidence(
        from_id="install",
        to_id="release",
        included=False,
        reason="no_path_within_constraints",
        direction="out",
        max_depth=1,
        type="depends_on",
    )

    docs.flush()
    reopened = omendb_vector.open(db_path, create=False).collection("docs", create=False)

    assert (
        reopened.relationship_evidence(
            "install", "release", max_depth=2, type="depends_on"
        )
        == evidence
    )


def test_delete_removes_record_from_graph_surface() -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")

    docs = omendb_vector.memory().collection(
        "docs", config=omendb_vector.CollectionConfig(dim=2, graph=True)
    )
    docs.set("install", vector=[0.0, 0.0])
    docs.set("wheel", vector=[1.0, 0.0])
    docs.add_relationship("install", "wheel", type="depends_on")

    docs.delete("wheel")

    assert docs.neighbors("install", type="depends_on") == []
    assert not docs.has_path("install", "wheel", type="depends_on")
    with pytest.raises(Exception, match="edge source or target not found"):
        docs.add_relationship("install", "wheel", type="depends_on")
