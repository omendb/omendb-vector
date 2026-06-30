from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path
from typing import Any

import omendb

DIM = 128


def _vec(axis: int, *, offset: int = 0, scale: float = 1.0) -> list[float]:
    vector = [0.0] * DIM
    vector[axis] = scale
    vector[(axis + 7 + offset) % DIM] = 0.05
    vector[(axis + 19 + offset) % DIM] = 0.02
    return vector


MEMORIES: list[dict[str, Any]] = [
    {
        "id": "task:package",
        "vector": _vec(0),
        "text": (
            "Build and publish the first macOS arm64 wheel with the private "
            "Mojo engine."
        ),
        "metadata": {"scope": "release", "kind": "task", "priority": "p2"},
    },
    {
        "id": "task:validation",
        "vector": _vec(16),
        "text": (
            "Run clean checkout release validation with SIFT-100K and Python "
            "package smoke."
        ),
        "metadata": {"scope": "release", "kind": "task", "priority": "p2"},
    },
    {
        "id": "task:workload",
        "vector": _vec(16, offset=2, scale=0.92),
        "text": (
            "Add a realistic agent memory workload using 128 dimensional embeddings."
        ),
        "metadata": {"scope": "release", "kind": "task", "priority": "p2"},
    },
    {
        "id": "decision:hnsw",
        "vector": _vec(32),
        "text": (
            "HNSW is the default local serving index for normal in-memory collections."
        ),
        "metadata": {"scope": "design", "kind": "decision"},
    },
    {
        "id": "decision:no-auto",
        "vector": _vec(32, offset=3, scale=0.95),
        "text": "Avoid smart automatic index-family routing in the first product.",
        "metadata": {"scope": "design", "kind": "decision"},
    },
    {
        "id": "note:filters",
        "vector": _vec(64),
        "text": (
            "Metadata filters prefer exact allow-list scoring before ACORN traversal."
        ),
        "metadata": {"scope": "roadmap", "kind": "note"},
    },
    {
        "id": "note:graph",
        "vector": _vec(48),
        "text": "Graph support is retrieval-oriented relationship context, not Cypher.",
        "metadata": {"scope": "roadmap", "kind": "note"},
    },
    {
        "id": "note:persistence",
        "vector": _vec(80),
        "text": "Snapshots carry checksums and publish from temporary directories.",
        "metadata": {"scope": "storage", "kind": "note"},
    },
]


def run(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    db_path = root / "agent-memory-release-db"
    snapshot_path = root / "agent-memory-release-snapshot"
    imported_path = root / "agent-memory-release-imported"

    db = omendb.create(db_path)
    memories = db.collection(
        "memories",
        config=omendb.CollectionConfig(dim=DIM, text=True, graph=True),
    )
    memories.set_many(MEMORIES)
    memories.set(
        "task:stale",
        vector=_vec(16, offset=11, scale=0.88),
        text="Remove stale release notes before publishing.",
        metadata={"scope": "release", "kind": "task", "priority": "p4"},
    )
    memories.update(
        "task:stale",
        vector=_vec(16, offset=12, scale=0.88),
        text="Superseded release note cleanup task.",
        metadata={"scope": "release", "kind": "task", "priority": "p4"},
    )
    memories.delete("task:stale")

    memories.add_relationship("task:package", "task:validation", type="blocks")
    memories.add_relationship("task:validation", "task:workload", type="requires")
    memories.add_relationship("task:workload", "decision:hnsw", type="references")
    memories.add_relationship("decision:hnsw", "decision:no-auto", type="related")

    dense_results = memories.search_vector(_vec(0), k=2)
    assert dense_results[0].id == "task:package"
    assert "task:stale" not in {hit.id for hit in dense_results}

    text_results = memories.search_text("automatic routing", k=1)
    assert text_results[0].id == "decision:no-auto"

    filtered_results = memories.search_vector(
        _vec(16),
        k=3,
        filter={"scope": "release", "priority": "p2"},
    )
    assert [hit.id for hit in filtered_results] == [
        "task:validation",
        "task:workload",
        "task:package",
    ]

    hybrid_results = memories.search_hybrid(
        vector=_vec(16),
        text="SIFT package validation",
        k=3,
        explain=True,
    )
    assert hybrid_results[0].id == "task:validation"
    assert hybrid_results[0].explanation is not None
    assert hybrid_results[0].explanation["method"] == "rrf"

    assert memories.neighbors("task:package", type="blocks") == ["task:validation"]
    assert memories.has_path("task:package", "decision:hnsw", max_depth=3, type=None)
    assert memories.shortest_path("task:package", "decision:hnsw", max_depth=3) == [
        "task:package",
        "task:validation",
        "task:workload",
        "decision:hnsw",
    ]

    db.close()
    reopened = omendb.open(db_path, create=False).collection("memories", create=False)
    assert reopened.get("task:stale") is None
    assert reopened.search_text("temporary directories", k=1)[0].id == (
        "note:persistence"
    )
    assert reopened.shortest_path("task:package", "decision:no-auto", max_depth=4) == [
        "task:package",
        "task:validation",
        "task:workload",
        "decision:hnsw",
        "decision:no-auto",
    ]

    omendb.open(db_path, create=False).snapshot(snapshot_path)
    imported = omendb.create(imported_path)
    imported.import_snapshot(snapshot_path)
    imported_memories = omendb.open(imported_path, create=False).collection(
        "memories", create=False
    )
    assert (
        imported_memories.search_hybrid(
            vector=_vec(16),
            text="release validation",
            k=1,
            explain=True,
        )[0].id
        == "task:validation"
    )
    assert imported_memories.neighbors("decision:hnsw", type="related") == [
        "decision:no-auto"
    ]

    print("agent_memory_release_smoke: ok")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()

    if args.root is None:
        with tempfile.TemporaryDirectory(prefix="omendb-agent-memory-release-") as tmp:
            run(Path(tmp))
        return

    run(args.root)


if __name__ == "__main__":
    main()
