"""Smoke tests for new dense dimension bindings.

These tests verify that the new HNSW bindings for common dimensions
(256, 512, 768, 1024, 1536, 3072) work at runtime.
"""

import pytest

import omendb_vector

# Test dimensions to verify
TEST_DIMS = [256, 512, 768, 1024, 1536, 3072]


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary database for testing."""
    return omendb_vector.create(tmp_path / "test_db")


class TestDenseDimensionSmoke:
    """Smoke tests for new dense dimension bindings."""

    @pytest.mark.parametrize("dim", TEST_DIMS)
    def test_memory_collection(self, dim):
        """Test creating an in-memory collection with the dimension."""
        db = omendb_vector.memory()
        config = omendb_vector.CollectionConfig(dim=dim, index="hnsw")
        collection = db.collection(f"test_{dim}", config=config)

        # Create a simple vector
        vector = [0.1] * dim

        # Set and get
        collection.set("item1", vector=vector, metadata={"test": True})
        result = collection.get("item1", include_vector=True)

        assert result is not None
        assert result["id"] == "item1"
        assert len(result["vector"]) == dim
        assert result["metadata"] == {"test": True}

    @pytest.mark.parametrize("dim", TEST_DIMS)
    def test_persistent_collection(self, tmp_db, dim):
        """Test creating a persistent collection with the dimension."""
        config = omendb_vector.CollectionConfig(dim=dim, index="hnsw")
        collection = tmp_db.collection(f"test_{dim}", config=config)

        # Create a simple vector
        vector = [0.1] * dim

        # Set and flush
        collection.set("item1", vector=vector, metadata={"dim": dim})
        collection.flush()

        # Reopen and verify
        collection2 = tmp_db.collection(f"test_{dim}", create=False)
        result = collection2.get("item1", include_vector=True)

        assert result is not None
        assert result["id"] == "item1"
        assert len(result["vector"]) == dim
        assert result["metadata"] == {"dim": dim}

    @pytest.mark.parametrize("dim", TEST_DIMS)
    def test_search(self, tmp_db, dim):
        """Test vector search with the dimension."""
        config = omendb_vector.CollectionConfig(dim=dim, index="hnsw")
        collection = tmp_db.collection(f"test_{dim}", config=config)

        # Add multiple items
        for i in range(5):
            vector = [float(i) / dim] * dim
            collection.set(f"item{i}", vector=vector, metadata={"index": i})

        # Search for similar vector
        query = [0.0] * dim
        results = collection.search_vector(query, k=3)

        assert len(results) == 3
        assert all(r.id.startswith("item") for r in results)

    @pytest.mark.parametrize("dim", TEST_DIMS)
    def test_text_with_vectors(self, tmp_db, dim):
        """Test text indexing with vectors at the dimension."""
        config = omendb_vector.CollectionConfig(dim=dim, index="hnsw", text=True)
        collection = tmp_db.collection(f"test_{dim}", config=config)

        vector = [0.1] * dim
        collection.set(
            "item1",
            vector=vector,
            text="Hello world",
            metadata={"test": True},
        )

        # Text search
        results = collection.search_text("hello", k=1)
        assert len(results) == 1
        assert results[0].id == "item1"

    @pytest.mark.parametrize("dim", TEST_DIMS)
    def test_flat_index(self, tmp_db, dim):
        """Test flat (exact) index with the dimension."""
        config = omendb_vector.CollectionConfig(dim=dim, index="flat")
        collection = tmp_db.collection(f"test_flat_{dim}", config=config)

        vector = [0.1] * dim
        collection.set("item1", vector=vector)

        # Search
        results = collection.search_vector(vector, k=1)
        assert len(results) == 1
        assert results[0].id == "item1"

    def test_vector_mismatch_error(self):
        """Test that vector dimension mismatch raises clear error."""
        db = omendb_vector.memory()
        config = omendb_vector.CollectionConfig(dim=768, index="hnsw")
        collection = db.collection("test", config=config)

        # Wrong dimension vector
        wrong_vector = [0.1] * 256

        with pytest.raises(ValueError, match="expected dim=768, got 256"):
            collection.set("item1", vector=wrong_vector)

    def test_unsupported_hnsw_dim_error(self):
        """Test that unsupported HNSW dims raise clear error."""
        with pytest.raises(ValueError, match="dim=257"):
            omendb_vector.CollectionConfig(dim=257, index="hnsw")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
