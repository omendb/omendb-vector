"""Tests for OmenDB dimension support."""

import pytest

import omendb
from omendb._dimensions import (
    DENSE_DYNAMIC_FLAT_MAX_DIM,
    DENSE_OPTIMIZED_DIMS,
    MULTIVECTOR_EXACT_DIMS,
    MULTIVECTOR_MUVERA_DIMS,
    DimensionSupport,
    dense_dim_error,
    is_supported_dense_dim,
    is_supported_multivector_dim,
    multivector_dim_error,
    vector_mismatch_error,
)


class TestDimensionRegistry:
    """Test the dimension registry constants and functions."""

    def test_dense_optimized_dims(self):
        """Verify the optimized dense dimensions match the spec."""
        assert DENSE_OPTIMIZED_DIMS == (2, 128, 256, 384, 512, 768, 1024, 1536, 3072)

    def test_dense_dynamic_flat_max_dim(self):
        """Verify the max dynamic flat dimension."""
        assert DENSE_DYNAMIC_FLAT_MAX_DIM == 8192

    def test_multivector_dims(self):
        """Verify multivector dimensions."""
        assert MULTIVECTOR_EXACT_DIMS == (2, 48, 128)
        assert MULTIVECTOR_MUVERA_DIMS == (2, 48, 128)

    def test_is_supported_dense_dim_hnsw(self):
        """Test HNSW dimension support validation."""
        # Optimized dims should be supported
        for dim in DENSE_OPTIMIZED_DIMS:
            assert is_supported_dense_dim(dim, "hnsw") is True

        # Non-optimized dims should not be supported for HNSW
        assert is_supported_dense_dim(257, "hnsw") is False
        assert is_supported_dense_dim(4096, "hnsw") is False
        assert is_supported_dense_dim(8192, "hnsw") is False

        # Invalid dims
        assert is_supported_dense_dim(0, "hnsw") is False
        assert is_supported_dense_dim(-1, "hnsw") is False

    def test_is_supported_dense_dim_flat(self):
        """Test flat dimension support validation."""
        # Optimized dims should be supported
        for dim in DENSE_OPTIMIZED_DIMS:
            assert is_supported_dense_dim(dim, "flat") is True

        # Arbitrary dims up to max should be supported for flat
        assert is_supported_dense_dim(257, "flat") is True
        assert is_supported_dense_dim(4096, "flat") is True
        assert is_supported_dense_dim(8192, "flat") is True

        # Dims exceeding max should not be supported
        assert is_supported_dense_dim(8193, "flat") is False

        # Invalid dims
        assert is_supported_dense_dim(0, "flat") is False
        assert is_supported_dense_dim(-1, "flat") is False

    def test_is_supported_multivector_dim(self):
        """Test multivector dimension support validation."""
        # Exact multivector
        for dim in MULTIVECTOR_EXACT_DIMS:
            assert is_supported_multivector_dim(dim, "none") is True
        assert is_supported_multivector_dim(256, "none") is False

        # MuVERA multivector
        for dim in MULTIVECTOR_MUVERA_DIMS:
            assert is_supported_multivector_dim(dim, "muvera") is True
        assert is_supported_multivector_dim(256, "muvera") is False

    def test_dense_dim_error_messages(self):
        """Test error message generation for dense dimensions."""
        # HNSW error
        error = dense_dim_error(257, "hnsw")
        assert "dim=257" in error
        assert "hnsw" in error
        assert "flat" in error  # Should suggest flat as alternative

        # Flat error for out-of-range
        error = dense_dim_error(8193, "flat")
        assert "dim=8193" in error
        assert "8192" in error

        # Invalid dim
        error = dense_dim_error(0, "hnsw")
        assert "positive" in error

    def test_multivector_dim_error_messages(self):
        """Test error message generation for multivector dimensions."""
        error = multivector_dim_error(256, "none")
        assert "dim=256" in error

        error = multivector_dim_error(256, "muvera")
        assert "dim=256" in error
        assert "muvera" in error

    def test_vector_mismatch_error(self):
        """Test vector mismatch error message."""
        error = vector_mismatch_error(1536, 768)
        assert "expected dim=1536" in error
        assert "got 768" in error


class TestDimensionSupportAPI:
    """Test the public dimension_support() and supported_dimensions() API."""

    def test_dimension_support_single_hnsw(self):
        """Test dimension support query for single-vector HNSW."""
        support = omendb.dimension_support(vector_mode="single", index="hnsw")
        assert isinstance(support, DimensionSupport)
        assert support.optimized == DENSE_OPTIMIZED_DIMS
        assert support.dynamic is False
        assert support.max_dynamic_dim is None

    def test_dimension_support_single_flat(self):
        """Test dimension support query for single-vector flat."""
        support = omendb.dimension_support(vector_mode="single", index="flat")
        assert isinstance(support, DimensionSupport)
        assert support.optimized == DENSE_OPTIMIZED_DIMS
        assert support.dynamic is True
        assert support.max_dynamic_dim == DENSE_DYNAMIC_FLAT_MAX_DIM

    def test_dimension_support_multi_exact(self):
        """Test dimension support query for multi-vector exact."""
        support = omendb.dimension_support(vector_mode="multi", encoding="none")
        assert isinstance(support, DimensionSupport)
        assert support.optimized == MULTIVECTOR_EXACT_DIMS
        assert support.dynamic is False

    def test_dimension_support_multi_muvera(self):
        """Test dimension support query for multi-vector MuVERA."""
        support = omendb.dimension_support(vector_mode="multi", encoding="muvera")
        assert isinstance(support, DimensionSupport)
        assert support.optimized == MULTIVECTOR_MUVERA_DIMS
        assert support.dynamic is False

    def test_supported_dimensions_convenience(self):
        """Test the supported_dimensions() convenience function."""
        dims = omendb.supported_dimensions(vector_mode="single", index="hnsw")
        assert dims == DENSE_OPTIMIZED_DIMS

        dims = omendb.supported_dimensions(vector_mode="multi", encoding="none")
        assert dims == MULTIVECTOR_EXACT_DIMS

    def test_dimension_support_in_all(self):
        """Verify dimension_support is exported in __all__."""
        assert "dimension_support" in omendb.__all__
        assert "supported_dimensions" in omendb.__all__
        assert "DimensionSupport" in omendb.__all__


class TestCollectionConfigValidation:
    """Test CollectionConfig dimension validation."""

    def test_valid_dense_hnsw_dims(self):
        """Test that optimized dims are accepted for HNSW."""
        for dim in DENSE_OPTIMIZED_DIMS:
            config = omendb.CollectionConfig(dim=dim, index="hnsw")
            assert config.dim == dim

    def test_invalid_dense_hnsw_dim(self):
        """Test that non-optimized dims are rejected for HNSW."""
        with pytest.raises(ValueError, match="dim=257"):
            omendb.CollectionConfig(dim=257, index="hnsw")

    def test_valid_dense_flat_dims(self):
        """Test that arbitrary dims up to max are accepted for flat."""
        for dim in [257, 4096, 8192]:
            config = omendb.CollectionConfig(dim=dim, index="flat")
            assert config.dim == dim

    def test_invalid_dense_flat_dim(self):
        """Test that dims exceeding max are rejected for flat."""
        with pytest.raises(ValueError, match="dim=8193"):
            omendb.CollectionConfig(dim=8193, index="flat")

    def test_valid_multivector_dims(self):
        """Test that valid multivector dims are accepted."""
        for dim in MULTIVECTOR_EXACT_DIMS:
            config = omendb.CollectionConfig(
                dim=dim, vector_mode="multi", metric="dot", index="flat"
            )
            assert config.dim == dim

    def test_invalid_multivector_dim(self):
        """Test that invalid multivector dims are rejected."""
        with pytest.raises(ValueError, match="dim=256"):
            omendb.CollectionConfig(
                dim=256, vector_mode="multi", metric="dot", index="flat"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
