"""Tests for graph operations: traverse, has_path, shortest_path, relationship_evidence.

Verifies graph traversal, path finding, and relationship evidence correctness.
"""

from __future__ import annotations

import sys

import pytest

import omendb_vector


def _skip_if_free_threaded() -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")


def _create_graph_collection(tmp_path, name="graph"):
    """Create a collection with graph enabled."""
    db = omendb_vector.create(str(tmp_path / "db"))
    col = db.collection(
        name,
        config=omendb_vector.CollectionConfig(
            dim=2,
            text=True,
            graph=True,
        ),
    )
    return col


def _build_simple_graph(col):
    """Build a simple graph: A -> B -> C -> D."""
    for id in ["A", "B", "C", "D"]:
        col.set(id, text=f"Node {id}", vector=[0.0, 0.0])

    col.add_relationship("A", "B", type="links")
    col.add_relationship("B", "C", type="links")
    col.add_relationship("C", "D", type="links")


def _build_branching_graph(col):
    """Build a branching graph: A -> B, A -> C, B -> D, C -> D."""
    for id in ["A", "B", "C", "D"]:
        col.set(id, text=f"Node {id}", vector=[0.0, 0.0])

    col.add_relationship("A", "B", type="links")
    col.add_relationship("A", "C", type="links")
    col.add_relationship("B", "D", type="links")
    col.add_relationship("C", "D", type="links")


def _build_cycle_graph(col):
    """Build a graph with a cycle: A -> B -> C -> A."""
    for id in ["A", "B", "C"]:
        col.set(id, text=f"Node {id}", vector=[0.0, 0.0])

    col.add_relationship("A", "B", type="links")
    col.add_relationship("B", "C", type="links")
    col.add_relationship("C", "A", type="links")


def _build_multi_type_graph(col):
    """Build a graph with multiple relationship types."""
    for id in ["A", "B", "C", "D"]:
        col.set(id, text=f"Node {id}", vector=[0.0, 0.0])

    col.add_relationship("A", "B", type="parent")
    col.add_relationship("A", "C", type="friend")
    col.add_relationship("B", "D", type="parent")
    col.add_relationship("C", "D", type="friend")


class TestTraverse:
    """Test traverse() operation."""

    def test_traverse_basic_forward(self, tmp_path) -> None:
        """Traverse forward from a node."""
        _skip_if_free_threaded()

        col = _create_graph_collection(tmp_path)
        _build_simple_graph(col)

        result = col.traverse("A", direction="out")
        # traverse includes the starting node
        assert "A" in result
        assert "B" in result
        assert "C" in result
        assert "D" in result

    def test_traverse_basic_backward(self, tmp_path) -> None:
        """Traverse backward from a node."""
        _skip_if_free_threaded()

        col = _create_graph_collection(tmp_path)
        _build_simple_graph(col)

        result = col.traverse("D", direction="in")
        # traverse includes the starting node
        assert "D" in result
        assert "C" in result
        assert "B" in result
        assert "A" in result

    def test_traverse_max_depth(self, tmp_path) -> None:
        """Traverse respects max_depth."""
        _skip_if_free_threaded()

        col = _create_graph_collection(tmp_path)
        _build_simple_graph(col)

        # Depth 1: only direct neighbors
        result = col.traverse("A", direction="out", max_depth=1)
        assert "B" in result
        assert "C" not in result
        assert "D" not in result

        # Depth 2: A -> B -> C
        result = col.traverse("A", direction="out", max_depth=2)
        assert "B" in result
        assert "C" in result
        assert "D" not in result

    def test_traverse_with_type_filter(self, tmp_path) -> None:
        """Traverse with relationship type filter."""
        _skip_if_free_threaded()

        col = _create_graph_collection(tmp_path)
        _build_multi_type_graph(col)

        # Only parent relationships
        result = col.traverse("A", direction="out", type="parent")
        assert "B" in result
        assert "C" not in result
        assert "D" in result  # A -> B -> D

    def test_traverse_isolated_node(self, tmp_path) -> None:
        """Traverse from isolated node returns just the node itself."""
        _skip_if_free_threaded()

        col = _create_graph_collection(tmp_path)
        col.set("isolated", text="Isolated", vector=[0.0, 0.0])

        result = col.traverse("isolated", direction="out")
        # traverse includes the starting node
        assert result == ["isolated"]

    def test_traverse_nonexistent_node(self, tmp_path) -> None:
        """Traverse from nonexistent node returns empty."""
        _skip_if_free_threaded()

        col = _create_graph_collection(tmp_path)
        _build_simple_graph(col)

        result = col.traverse("nonexistent", direction="out")
        assert result == []


class TestHasPath:
    """Test has_path() operation."""

    def test_has_path_direct(self, tmp_path) -> None:
        """Direct path exists."""
        _skip_if_free_threaded()

        col = _create_graph_collection(tmp_path)
        _build_simple_graph(col)

        assert col.has_path("A", "B") is True
        assert col.has_path("B", "C") is True
        assert col.has_path("C", "D") is True

    def test_has_path_indirect(self, tmp_path) -> None:
        """Indirect path exists."""
        _skip_if_free_threaded()

        col = _create_graph_collection(tmp_path)
        _build_simple_graph(col)

        assert col.has_path("A", "D") is True
        assert col.has_path("A", "C") is True

    def test_has_path_reverse(self, tmp_path) -> None:
        """Reverse path does not exist (directed graph)."""
        _skip_if_free_threaded()

        col = _create_graph_collection(tmp_path)
        _build_simple_graph(col)

        assert col.has_path("D", "A") is False
        assert col.has_path("B", "A") is False

    def test_has_path_with_max_depth(self, tmp_path) -> None:
        """has_path respects max_depth."""
        _skip_if_free_threaded()

        col = _create_graph_collection(tmp_path)
        _build_simple_graph(col)

        # A -> D requires depth 3
        assert col.has_path("A", "D", max_depth=3) is True
        assert col.has_path("A", "D", max_depth=2) is False

    def test_has_path_with_type_filter(self, tmp_path) -> None:
        """has_path with type filter."""
        _skip_if_free_threaded()

        col = _create_graph_collection(tmp_path)
        _build_multi_type_graph(col)

        # A -> D via parent (A -> B -> D)
        assert col.has_path("A", "D", type="parent") is True

        # A -> C via parent (no direct parent link)
        assert col.has_path("A", "C", type="parent") is False

    def test_has_path_same_node(self, tmp_path) -> None:
        """Path from node to itself."""
        _skip_if_free_threaded()

        col = _create_graph_collection(tmp_path)
        _build_simple_graph(col)

        # has_path to self should be true (trivial path)
        assert col.has_path("A", "A") is True

    def test_has_path_nonexistent_nodes(self, tmp_path) -> None:
        """Path with nonexistent nodes."""
        _skip_if_free_threaded()

        col = _create_graph_collection(tmp_path)
        _build_simple_graph(col)

        assert col.has_path("A", "nonexistent") is False
        assert col.has_path("nonexistent", "A") is False


class TestShortestPath:
    """Test shortest_path() operation."""

    def test_shortest_path_direct(self, tmp_path) -> None:
        """Shortest path for direct connection."""
        _skip_if_free_threaded()

        col = _create_graph_collection(tmp_path)
        _build_simple_graph(col)

        path = col.shortest_path("A", "B")
        assert path == ["A", "B"]

    def test_shortest_path_indirect(self, tmp_path) -> None:
        """Shortest path for indirect connection."""
        _skip_if_free_threaded()

        col = _create_graph_collection(tmp_path)
        _build_simple_graph(col)

        path = col.shortest_path("A", "D")
        assert path == ["A", "B", "C", "D"]

    def test_shortest_path_branching(self, tmp_path) -> None:
        """Shortest path in branching graph."""
        _skip_if_free_threaded()

        col = _create_graph_collection(tmp_path)
        _build_branching_graph(col)

        # Both paths have length 3, either is valid
        path = col.shortest_path("A", "D")
        assert len(path) == 3
        assert path[0] == "A"
        assert path[-1] == "D"

    def test_shortest_path_with_max_depth(self, tmp_path) -> None:
        """Shortest path respects max_depth."""
        _skip_if_free_threaded()

        col = _create_graph_collection(tmp_path)
        _build_simple_graph(col)

        # A -> D requires depth 3
        path = col.shortest_path("A", "D", max_depth=3)
        assert path == ["A", "B", "C", "D"]

        # With max_depth=2, no path exists
        path = col.shortest_path("A", "D", max_depth=2)
        assert path == []

    def test_shortest_path_with_type_filter(self, tmp_path) -> None:
        """Shortest path with type filter."""
        _skip_if_free_threaded()

        col = _create_graph_collection(tmp_path)
        _build_multi_type_graph(col)

        # A -> D via parent
        path = col.shortest_path("A", "D", type="parent")
        assert path == ["A", "B", "D"]

    def test_shortest_path_same_node(self, tmp_path) -> None:
        """Shortest path to self."""
        _skip_if_free_threaded()

        col = _create_graph_collection(tmp_path)
        _build_simple_graph(col)

        path = col.shortest_path("A", "A")
        assert path == ["A"]

    def test_shortest_path_nonexistent(self, tmp_path) -> None:
        """Shortest path with nonexistent nodes."""
        _skip_if_free_threaded()

        col = _create_graph_collection(tmp_path)
        _build_simple_graph(col)

        path = col.shortest_path("A", "nonexistent")
        assert path == []


class TestRelationshipEvidence:
    """Test relationship_evidence() operation."""

    def test_evidence_direct(self, tmp_path) -> None:
        """Evidence for direct relationship."""
        _skip_if_free_threaded()

        col = _create_graph_collection(tmp_path)
        _build_simple_graph(col)

        evidence = col.relationship_evidence("A", "B")
        assert evidence.included is True
        assert evidence.path == ("A", "B")
        assert evidence.depth == 1
        assert evidence.reason == "shortest_path_within_constraints"

    def test_evidence_indirect(self, tmp_path) -> None:
        """Evidence for indirect relationship."""
        _skip_if_free_threaded()

        col = _create_graph_collection(tmp_path)
        _build_simple_graph(col)

        evidence = col.relationship_evidence("A", "D")
        assert evidence.included is True
        assert evidence.path == ("A", "B", "C", "D")
        assert evidence.depth == 3

    def test_evidence_no_path(self, tmp_path) -> None:
        """Evidence when no path exists."""
        _skip_if_free_threaded()

        col = _create_graph_collection(tmp_path)
        _build_simple_graph(col)

        evidence = col.relationship_evidence("D", "A")
        assert evidence.included is False
        assert evidence.reason == "no_path_within_constraints"

    def test_evidence_same_node(self, tmp_path) -> None:
        """Evidence for same node."""
        _skip_if_free_threaded()

        col = _create_graph_collection(tmp_path)
        _build_simple_graph(col)

        evidence = col.relationship_evidence("A", "A")
        assert evidence.included is True
        assert evidence.reason == "same_record"
        assert evidence.depth == 0

    def test_evidence_with_type_filter(self, tmp_path) -> None:
        """Evidence with type filter."""
        _skip_if_free_threaded()

        col = _create_graph_collection(tmp_path)
        _build_multi_type_graph(col)

        evidence = col.relationship_evidence("A", "D", type="parent")
        assert evidence.included is True
        assert evidence.path == ("A", "B", "D")

    def test_evidence_with_max_depth(self, tmp_path) -> None:
        """Evidence respects max_depth."""
        _skip_if_free_threaded()

        col = _create_graph_collection(tmp_path)
        _build_simple_graph(col)

        # A -> D requires depth 3
        evidence = col.relationship_evidence("A", "D", max_depth=3)
        assert evidence.included is True

        evidence = col.relationship_evidence("A", "D", max_depth=2)
        assert evidence.included is False


class TestGraphPersistence:
    """Test graph operations survive flush/reopen."""

    def test_graph_survives_reopen(self, tmp_path) -> None:
        """Graph relationships persist across flush/reopen."""
        _skip_if_free_threaded()

        db = omendb_vector.create(str(tmp_path / "db"))
        col = db.collection(
            "graph",
            config=omendb_vector.CollectionConfig(dim=2, text=True, graph=True),
        )
        _build_simple_graph(col)
        db.flush()

        # Reopen
        db2 = omendb_vector.open(str(tmp_path / "db"))
        col2 = db2.collection("graph")

        assert col2.has_path("A", "D") is True
        path = col2.shortest_path("A", "D")
        assert path == ["A", "B", "C", "D"]

    def test_graph_types_survive_reopen(self, tmp_path) -> None:
        """Relationship types persist across flush/reopen."""
        _skip_if_free_threaded()

        db = omendb_vector.create(str(tmp_path / "db"))
        col = db.collection(
            "graph",
            config=omendb_vector.CollectionConfig(dim=2, text=True, graph=True),
        )
        _build_multi_type_graph(col)
        db.flush()

        # Reopen
        db2 = omendb_vector.open(str(tmp_path / "db"))
        col2 = db2.collection("graph")

        # Type filter should work after reopen
        assert col2.has_path("A", "D", type="parent") is True
        assert col2.has_path("A", "C", type="parent") is False


class TestGraphErrorHandling:
    """Test graph operation error handling."""

    def test_graph_requires_config(self, tmp_path) -> None:
        """Graph operations require graph=True in config."""
        _skip_if_free_threaded()

        db = omendb_vector.create(str(tmp_path / "db"))
        col = db.collection("nograph", config=omendb_vector.CollectionConfig(dim=2))
        col.set("A", vector=[0.0, 0.0])

        with pytest.raises(omendb_vector.EngineUnavailableError, match="relationships"):
            col.add_relationship("A", "B", type="links")

        with pytest.raises(omendb_vector.EngineUnavailableError, match="relationships"):
            col.traverse("A")

        with pytest.raises(omendb_vector.EngineUnavailableError, match="relationships"):
            col.has_path("A", "B")

        with pytest.raises(omendb_vector.EngineUnavailableError, match="relationships"):
            col.shortest_path("A", "B")

    def test_invalid_direction(self, tmp_path) -> None:
        """Invalid direction raises error."""
        _skip_if_free_threaded()

        col = _create_graph_collection(tmp_path)
        _build_simple_graph(col)

        with pytest.raises(ValueError, match="direction"):
            col.traverse("A", direction="invalid")

    def test_negative_max_depth(self, tmp_path) -> None:
        """Negative max_depth raises error."""
        _skip_if_free_threaded()

        col = _create_graph_collection(tmp_path)
        _build_simple_graph(col)

        with pytest.raises(ValueError, match="max_depth"):
            col.traverse("A", max_depth=-1)


class TestGraphOperationsWithVacuum:
    """Test graph operations survive vacuum."""

    def test_graph_survives_vacuum(self, tmp_path) -> None:
        """Graph relationships persist across vacuum."""
        _skip_if_free_threaded()

        col = _create_graph_collection(tmp_path)
        _build_simple_graph(col)

        # Delete a node not in the graph
        col.set("extra", text="Extra", vector=[0.0, 0.0])
        col.delete("extra")

        col.vacuum()

        # Graph should be intact
        assert col.has_path("A", "D") is True
        path = col.shortest_path("A", "D")
        assert path == ["A", "B", "C", "D"]
