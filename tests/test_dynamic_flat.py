"""Tests for dynamic flat collection (arbitrary dimensions)."""

import pytest

import omendb_vector


class TestDynamicFlatCollection:
    """Test dynamic flat collection for arbitrary dimensions."""

    def test_dynamic_flat_4097(self):
        """Test creating a dynamic flat collection with dim=4097."""
        db = omendb_vector.memory()
        # Use index="flat" with non-standard dimension
        config = omendb_vector.CollectionConfig(dim=4097, index="flat")
        collection = db.collection("test_4097", config=config)

        # Create a simple vector
        vector = [0.1] * 4097

        # Set and get
        collection.set("item1", vector=vector, metadata={"dim": 4097})
        result = collection.get("item1", include_vector=True)

        assert result is not None
        assert result["id"] == "item1"
        assert len(result["vector"]) == 4097
        assert result["metadata"] == {"dim": 4097}

    def test_dynamic_flat_100(self):
        """Test creating a dynamic flat collection with dim=100."""
        db = omendb_vector.memory()
        config = omendb_vector.CollectionConfig(dim=100, index="flat")
        collection = db.collection("test_100", config=config)

        # Add multiple items
        for i in range(5):
            vector = [float(i) / 100] * 100
            collection.set(f"item{i}", vector=vector, metadata={"index": i})

        # Search for similar vector
        query = [0.0] * 100
        results = collection.search_vector(query, k=3)

        assert len(results) == 3
        assert all(r.id.startswith("item") for r in results)

    def test_dynamic_flat_duplicate_set_replaces_record(self):
        """Native duplicate set should replace by tombstoning old record."""
        db = omendb_vector.memory()
        config = omendb_vector.CollectionConfig(dim=100, index="flat")
        collection = db.collection("test_duplicate", config=config)
        native = collection._native_handle()

        native.set("dup", [0.0] * 100, None, None)
        native.set("dup", [1.0] * 100, None, None)

        result = collection.get("dup", include_vector=True)
        assert result is not None
        assert result["vector"] == [1.0] * 100
        assert int(native.len()) == 1

    def test_dynamic_flat_vector_mismatch(self):
        """Test that vector dimension mismatch raises clear error."""
        db = omendb_vector.memory()
        config = omendb_vector.CollectionConfig(dim=4097, index="flat")
        collection = db.collection("test", config=config)

        # Wrong dimension vector
        wrong_vector = [0.1] * 256

        with pytest.raises(ValueError, match="expected dim=4097, got 256"):
            collection.set("item1", vector=wrong_vector)

    def test_unsupported_flat_dim(self):
        """Test that dims exceeding max are rejected."""
        with pytest.raises(ValueError, match="dim=8193"):
            omendb_vector.CollectionConfig(dim=8193, index="flat")

    def test_dynamic_flat_search_exact_ids(self):
        """Test exact ID search with dynamic flat."""
        db = omendb_vector.memory()
        config = omendb_vector.CollectionConfig(dim=100, index="flat")
        collection = db.collection("test", config=config)

        # Add items
        for i in range(5):
            vector = [float(i) / 100] * 100
            collection.set(f"item{i}", vector=vector)

        # Search with include_ids
        query = [0.0] * 100
        results = collection.search_vector(
            query, k=3, include_ids=["item0", "item1", "item2"]
        )

        assert len(results) == 3
        assert all(r.id in ["item0", "item1", "item2"] for r in results)

    def test_dynamic_flat_persistent_create_open(self, tmp_path):
        """Test persistent dynamic flat collection create/open/flush/reopen."""
        db_path = str(tmp_path / "test_dynamic_flat.db")
        config = omendb_vector.CollectionConfig(dim=4097, index="flat")

        # Create persistent collection
        db = omendb_vector.create(db_path)
        collection = db.collection("test_4097", config=config)

        # Add items
        for i in range(3):
            vector = [float(i) / 100] * 4097
            collection.set(f"item{i}", vector=vector, metadata={"index": i})

        # Flush and close
        collection.flush()
        del collection
        del db

        # Reopen and verify
        db2 = omendb_vector.open(db_path)
        collection2 = db2.collection("test_4097", config=config, create=False)

        # Verify search works
        query = [0.0] * 4097
        results = collection2.search_vector(query, k=2)
        assert len(results) == 2
        assert all(r.id.startswith("item") for r in results)

        # Verify get works
        item = collection2.get("item0", include_vector=True)
        assert item is not None
        assert item["id"] == "item0"
        assert len(item["vector"]) == 4097
        assert item["metadata"] == {"index": 0}

    def test_dynamic_flat_persistent_delete_reopen(self, tmp_path):
        """Test that deletes persist across reopen."""
        db_path = str(tmp_path / "test_delete.db")
        config = omendb_vector.CollectionConfig(dim=100, index="flat")

        # Create and add items
        db = omendb_vector.create(db_path)
        collection = db.collection("test", config=config)
        collection.set("item1", vector=[0.1] * 100)
        collection.set("item2", vector=[0.2] * 100)
        collection.set("item3", vector=[0.3] * 100)

        # Delete one item
        collection.delete("item2")

        # Flush and reopen
        collection.flush()
        del collection
        del db

        db2 = omendb_vector.open(db_path)
        collection2 = db2.collection("test", config=config, create=False)

        # Verify delete persisted
        assert collection2.get("item2") is None
        assert collection2.get("item1") is not None
        assert collection2.get("item3") is not None

        # Verify search returns only 2 results
        results = collection2.search_vector([0.0] * 100, k=10)
        assert len(results) == 2

    def test_dynamic_flat_persistent_source_span(self, tmp_path):
        """Test that source spans persist across reopen."""
        db_path = str(tmp_path / "test_source.db")
        config = omendb_vector.CollectionConfig(dim=50, index="flat")

        source = {"path": "file.py", "line_start": 10, "line_end": 20}

        # Create and add items with source spans
        db = omendb_vector.create(db_path)
        collection = db.collection("test", config=config)
        collection.set(
            "item1",
            vector=[0.1] * 50,
            source=source,
        )

        # Flush and reopen
        collection.flush()
        del collection
        del db

        db2 = omendb_vector.open(db_path)
        collection2 = db2.collection("test", config=config, create=False)

        # Verify source span persisted
        item = collection2.get("item1", include_source=True)
        assert item is not None
        assert item["source"] == source


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
