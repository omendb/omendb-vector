"""Local Python API for OmenDB Vector."""

from ._api import (
    CheckIssue,
    CheckResult,
    Collection,
    CollectionConfig,
    Database,
    EngineUnavailableError,
    HNSWConfig,
    MaintenanceStats,
    RelationshipEvidence,
    RelationshipStep,
    SearchEvidence,
    SearchResult,
    StoreBusyError,
    check,
    create,
    memory,
    open,
)
from ._dimensions import (
    DimensionSupport,
    dimension_support,
    supported_dimensions,
)
from ._native import engine_version
from .ingest import ingest_directory, ingest_file, ingest_text

__all__ = [
    "CheckIssue",
    "CheckResult",
    "Collection",
    "CollectionConfig",
    "Database",
    "DimensionSupport",
    "EngineUnavailableError",
    "HNSWConfig",
    "MaintenanceStats",
    "RelationshipEvidence",
    "RelationshipStep",
    "SearchEvidence",
    "SearchResult",
    "StoreBusyError",
    "check",
    "create",
    "dimension_support",
    "engine_version",
    "ingest_directory",
    "ingest_file",
    "ingest_text",
    "memory",
    "open",
    "supported_dimensions",
]
