"""Central dimension registry for OmenDB vector collections.

This module is the single source of truth for supported dimensions,
used by _native.py, validation, error messages, public capability APIs,
and tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Dense single-vector optimized dimensions (HNSW bindings compiled for these).
# 2 is for tests, 128 for SIFT/ColBERT, rest cover common embedding models.
DENSE_OPTIMIZED_DIMS: tuple[int, ...] = (2, 128, 256, 384, 512, 768, 1024, 1536, 3072)

# Maximum dimension for dynamic exact flat search.
DENSE_DYNAMIC_FLAT_MAX_DIM: int = 8192

# Multivector exact MaxSim dimensions.
MULTIVECTOR_EXACT_DIMS: tuple[int, ...] = (2, 48, 128)

# Multivector MuVERA FDE dimensions.
MULTIVECTOR_MUVERA_DIMS: tuple[int, ...] = (2, 48, 128)


type VectorMode = Literal["single", "multi"]
type IndexMode = Literal["hnsw", "flat"]
type EncodingMode = Literal["none", "muvera"]


@dataclass(frozen=True, slots=True)
class DimensionSupport:
    """Result of dimension_support() queries."""

    optimized: tuple[int, ...]
    dynamic: bool
    max_dynamic_dim: int | None = None


def dimension_support(
    vector_mode: VectorMode = "single",
    index: IndexMode = "hnsw",
    encoding: EncodingMode = "none",
) -> DimensionSupport:
    """Query dimension support for a given vector mode/index/encoding combination.

    Args:
        vector_mode: 'single' for dense single-vector, 'multi' for multivector.
        index: 'hnsw' for ANN, 'flat' for exact search.
        encoding: 'none' for standard, 'muvera' for MuVERA FDE (multi-vector only).

    Returns:
        DimensionSupport with optimized dims, dynamic support flag, and max dim.
    """
    if vector_mode == "multi":
        if encoding == "muvera":
            return DimensionSupport(
                optimized=MULTIVECTOR_MUVERA_DIMS,
                dynamic=False,
            )
        return DimensionSupport(
            optimized=MULTIVECTOR_EXACT_DIMS,
            dynamic=False,
        )

    # Single-vector (dense)
    if index == "flat":
        return DimensionSupport(
            optimized=DENSE_OPTIMIZED_DIMS,
            dynamic=True,
            max_dynamic_dim=DENSE_DYNAMIC_FLAT_MAX_DIM,
        )

    # HNSW: only optimized dims, no dynamic
    return DimensionSupport(
        optimized=DENSE_OPTIMIZED_DIMS,
        dynamic=False,
    )


def supported_dimensions(
    vector_mode: VectorMode = "single",
    index: IndexMode = "hnsw",
    encoding: EncodingMode = "none",
) -> tuple[int, ...]:
    """Convenience function returning just the optimized dimensions tuple."""
    return dimension_support(vector_mode, index, encoding).optimized


def is_supported_dense_dim(dim: int, index: IndexMode = "hnsw") -> bool:
    """Check if a dense dimension is supported for the given index mode."""
    if dim <= 0:
        return False
    if index == "flat":
        return dim <= DENSE_DYNAMIC_FLAT_MAX_DIM
    return dim in DENSE_OPTIMIZED_DIMS


def is_supported_multivector_dim(dim: int, encoding: EncodingMode = "none") -> bool:
    """Check if a multivector dimension is supported."""
    if encoding == "muvera":
        return dim in MULTIVECTOR_MUVERA_DIMS
    return dim in MULTIVECTOR_EXACT_DIMS


def dense_dim_error(dim: int, index: IndexMode = "hnsw") -> str:
    """Generate an error message for unsupported dense dimensions."""
    if dim <= 0:
        return f"dense dimension must be positive; got dim={dim}"
    if index == "hnsw":
        return (
            f"OmenDB dense hnsw/l2 supports dim in {DENSE_OPTIMIZED_DIMS}; "
            f"got dim={dim}. Use index='flat' for exact dynamic search, or "
            f"choose a supported optimized HNSW dimension."
        )
    return (
        f"OmenDB dense flat supports dim in 1..{DENSE_DYNAMIC_FLAT_MAX_DIM}; "
        f"got dim={dim}"
    )


def multivector_dim_error(dim: int, encoding: EncodingMode = "none") -> str:
    """Generate an error message for unsupported multivector dimensions."""
    if encoding == "muvera":
        return (
            f"OmenDB multivector muvera supports dim in {MULTIVECTOR_MUVERA_DIMS}; "
            f"got dim={dim}"
        )
    return (
        f"OmenDB multivector exact supports dim in {MULTIVECTOR_EXACT_DIMS}; "
        f"got dim={dim}"
    )


def vector_mismatch_error(expected: int, got: int) -> str:
    """Generate an error message for vector dimension mismatch."""
    return f"vector dimension mismatch: expected dim={expected}, got {got}"
