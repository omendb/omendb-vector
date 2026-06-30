"""Type stubs for the compiled OmenDB Mojo engine extension."""

from __future__ import annotations

from typing import Any

def engine_version() -> str: ...

# ---- Collection constructors ----

def dense_collection(
    dim: int,
    path: str | None = None,
    *,
    create: bool = True,
    hnsw: Any | None = None,
) -> Any: ...

def dense_collection2(
    path: str | None = None,
    *,
    create: bool = True,
    hnsw: Any | None = None,
) -> Any: ...

def dense_collection128(
    path: str | None = None,
    *,
    create: bool = True,
    hnsw: Any | None = None,
) -> Any: ...

def dense_collection256(
    path: str | None = None,
    *,
    create: bool = True,
    hnsw: Any | None = None,
) -> Any: ...

def dense_collection384(
    path: str | None = None,
    *,
    create: bool = True,
    hnsw: Any | None = None,
) -> Any: ...

def dense_collection512(
    path: str | None = None,
    *,
    create: bool = True,
    hnsw: Any | None = None,
) -> Any: ...

def dense_collection768(
    path: str | None = None,
    *,
    create: bool = True,
    hnsw: Any | None = None,
) -> Any: ...

def dense_collection1024(
    path: str | None = None,
    *,
    create: bool = True,
    hnsw: Any | None = None,
) -> Any: ...

def dense_collection1536(
    path: str | None = None,
    *,
    create: bool = True,
    hnsw: Any | None = None,
) -> Any: ...

def dense_collection3072(
    path: str | None = None,
    *,
    create: bool = True,
    hnsw: Any | None = None,
) -> Any: ...
