from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from ._dimensions import (
    DENSE_OPTIMIZED_DIMS,
    EncodingMode,
    IndexMode,
    dense_dim_error,
    is_supported_dense_dim,
    is_supported_multivector_dim,
    multivector_dim_error,
)


class EngineUnavailableError(RuntimeError):
    """Raised when the packaged Mojo engine cannot be loaded."""


def _load_engine():
    if "t" in sys.abiflags:
        raise EngineUnavailableError(
            "OmenDB Vector Mojo engine extension is built for the standard CPython ABI; "
            "free-threaded Python is not supported yet"
        )
    try:
        return import_module("omendb_vector._omendb_vector_engine")
    except ModuleNotFoundError as exc:
        if exc.name == "omendb_vector._omendb_vector_engine":
            raise EngineUnavailableError(
                "OmenDB Vector Mojo engine extension is not built; run "
                "`pixi run mojo build src/python_engine.mojo "
                "--emit shared-lib -o src/omendb_vector/_omendb_vector_engine.so`"
            ) from exc
        raise


def engine_version() -> str:
    engine = _load_engine()
    return str(engine.engine_version())


def dense_collection2(
    path: str | Path | None = None, *, create: bool = True, hnsw: Any | None = None
) -> Any:
    return dense_collection(2, path, create=create, hnsw=hnsw)


def dense_collection(
    dim: int,
    path: str | Path | None = None,
    *,
    create: bool = True,
    hnsw: Any | None = None,
    index: str = "hnsw",
) -> Any:
    engine = _load_engine()
    if not is_supported_dense_dim(dim, cast(IndexMode, index)):
        raise EngineUnavailableError(dense_dim_error(dim, cast(IndexMode, index)))
    # For HNSW, we need optimized bindings
    if index == "hnsw" and dim not in DENSE_OPTIMIZED_DIMS:
        raise EngineUnavailableError(dense_dim_error(dim, cast(IndexMode, index)))
    # For flat with non-optimized dims, use dynamic flat
    if index == "flat" and dim not in DENSE_OPTIMIZED_DIMS:
        return _dynamic_flat_collection(dim, path, create=create)
    suffix = str(dim)
    params = _hnsw_params(hnsw)
    if path is None:
        return getattr(engine, f"_dense_collection{suffix}_memory")(*params)
    if create:
        return getattr(engine, f"_dense_collection{suffix}_create")(str(path), *params)
    return getattr(engine, f"_dense_collection{suffix}_open")(str(path))


def _dynamic_flat_collection(
    dim: int,
    path: str | Path | None = None,
    *,
    create: bool = True,
) -> Any:
    """Create or open a dynamic exact flat collection for arbitrary dimensions."""
    engine = _load_engine()
    if path is None:
        return engine._dynamic_flat_collection_memory(dim)
    if create:
        return engine._dynamic_flat_collection_create(str(path), dim)
    return engine._dynamic_flat_collection_open(str(path))


def multivector_collection(
    dim: int,
    path: str | Path | None = None,
    *,
    create: bool = True,
    encoding: str = "none",
) -> Any:
    engine = _load_engine()
    if not is_supported_multivector_dim(dim, cast(EncodingMode, encoding)):
        raise EngineUnavailableError(
            multivector_dim_error(dim, cast(EncodingMode, encoding))
        )
    if encoding not in ("none", "muvera"):
        raise EngineUnavailableError(
            "OmenDB Vector Mojo engine currently supports multi-vector "
            "encoding='none' or encoding='muvera'"
        )
    suffix = str(dim)
    encoding_suffix = "" if encoding == "none" else "_muvera"
    if path is None:
        return getattr(
            engine, f"_multivector_collection{suffix}{encoding_suffix}_memory"
        )()
    if create:
        return getattr(
            engine, f"_multivector_collection{suffix}{encoding_suffix}_create"
        )(str(path))
    return getattr(engine, f"_multivector_collection{suffix}{encoding_suffix}_open")(
        str(path)
    )


def _hnsw_params(hnsw: Any | None) -> tuple[int, int, int, float, str]:
    if hnsw is None:
        return (16, 100, 100, 1.0, "l2")
    return (
        int(hnsw.m),
        int(hnsw.ef_construction),
        int(hnsw.ef_search),
        float(hnsw.alpha),
        str(getattr(hnsw, 'metric', 'l2')),
    )
