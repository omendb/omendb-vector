from __future__ import annotations

import argparse
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path

import omendb_vector

MEMORIES = [
    {
        "id": "task:install",
        "vector": [0.02, 0.0],
        "text": "Install OmenDB Vector locally with uv and standard CPython.",
        "metadata": {"scope": "project", "kind": "task"},
    },
    {
        "id": "task:wheel",
        "vector": [0.08, 0.02],
        "text": "Build the macOS arm64 wheel with the private Mojo engine.",
        "metadata": {"scope": "project", "kind": "task"},
    },
    {
        "id": "decision:hnsw",
        "vector": [0.9, 0.05],
        "text": "HNSW is the default local serving index for OmenDB Vector.",
        "metadata": {"scope": "design", "kind": "decision"},
    },
    {
        "id": "decision:auto",
        "vector": [0.86, 0.12],
        "text": "Avoid smart automatic index-family routing in the first product.",
        "metadata": {"scope": "design", "kind": "decision"},
    },
    {
        "id": "note:filters",
        "vector": [0.15, 0.8],
        "text": "Metadata filters come after the core Python retrieval lifecycle.",
        "metadata": {"scope": "roadmap", "kind": "note"},
    },
    {
        "id": "note:graph",
        "vector": [0.2, 0.9],
        "text": "Graph support is retrieval-oriented, not a Cypher-first surface.",
        "metadata": {"scope": "roadmap", "kind": "note"},
    },
]


def run(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    db_path = root / "agent-memory-db"
    snapshot_path = root / "agent-memory-snapshot"
    imported_path = root / "agent-memory-imported"

    db = omendb_vector.create(db_path)
    memories = db.collection(
        "memories", config=omendb_vector.CollectionConfig(dim=2, text=True)
    )
    memories.set_many(MEMORIES)

    dense_results = memories.search_vector([0.05, 0.01], k=2)
    assert {hit.id for hit in dense_results} == {"task:install", "task:wheel"}

    text_results = memories.search_text("automatic routing", k=1)
    assert text_results[0].id == "decision:auto"
    assert text_results[0].metadata == {"scope": "design", "kind": "decision"}

    filtered_results = memories.search_vector(
        [0.05, 0.01], k=2, filter={"kind": "task", "scope": {"$exists": True}}
    )
    assert {hit.id for hit in filtered_results} == {"task:install", "task:wheel"}

    task = memories.get("task:install", include_vector=True)
    assert task is not None
    assert task["id"] == "task:install"
    assert task["metadata"] == {"scope": "project", "kind": "task"}
    assert _rounded_vector(task["vector"]) == [0.02, 0.0]

    db.close()
    reopened = omendb_vector.open(db_path, create=False).collection("memories", create=False)
    assert reopened.search_text("HNSW default", k=1)[0].id == "decision:hnsw"
    assert reopened.search_vector([0.18, 0.86], k=1)[0].id == "note:graph"

    omendb_vector.open(db_path, create=False).snapshot(snapshot_path)
    imported = omendb_vector.create(imported_path)
    imported.import_snapshot(snapshot_path)
    imported_memories = omendb_vector.open(imported_path, create=False).collection(
        "memories", create=False
    )
    assert imported_memories.get("decision:auto") == {
        "id": "decision:auto",
        "metadata": {"scope": "design", "kind": "decision"},
    }

    print("agent_memory_smoke: ok")


def _rounded_vector(value: Sequence[float]) -> list[float]:
    return [round(float(item), 2) for item in value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()

    if args.root is None:
        with tempfile.TemporaryDirectory(prefix="omendb_vector-agent-memory-") as tmp:
            run(Path(tmp))
        return

    run(args.root)


if __name__ == "__main__":
    main()
