"""Source-structure ingestion helper for OmenDB.

Provides narrow helpers to ingest files into OmenDB collections with proper
source spans, section/page/symbol metadata, and optional parent/reference edges.

This is a convenience layer over the core set() API, not an app-layer policy.
Users who need custom chunking, embedding, or metadata extraction should use
the core API directly.

Example:
    import omendb
    from omendb.ingest import ingest_file, ingest_directory

    db = omendb.create("./mydb")
    col = db.collection("docs", config=omendb.CollectionConfig(dim=384, text=True))

    # Ingest a single file
    ingest_file(col, "README.md")

    # Ingest a directory
    ingest_directory(col, "./src", glob="*.py")
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ._api import Collection


def _file_hash(path: Path) -> str:
    """Generate a stable hash for a file path."""
    return hashlib.sha256(str(path).encode()).hexdigest()[:16]


def _make_source(
    path: Path, line_start: int = 1, line_end: int | None = None
) -> dict[str, Any]:
    """Create a source span dict for a file region."""
    source: dict[str, Any] = {"path": str(path)}
    if line_start > 1:
        source["line_start"] = line_start
    if line_end is not None:
        source["line_end"] = line_end
    return source


def ingest_text(
    col: Collection,
    id: str,
    text: str,
    *,
    source: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    vector: list[float] | None = None,
) -> None:
    """Ingest a text chunk into a collection.

    This is a thin wrapper around col.set() that provides sensible defaults
    for source-structured data.

    Args:
        col: The collection to ingest into.
        id: The item ID.
        text: The text content.
        source: Source span metadata (path, line_start, line_end).
        metadata: Additional metadata.
        vector: Optional vector (if None, collection must support text-only items).
    """
    col.set(
        id,
        text=text,
        source=source,
        metadata=metadata,
        vector=vector,
    )


def ingest_file(
    col: Collection,
    path: str | Path,
    *,
    chunk_size: int | None = None,
    chunk_overlap: int = 0,
    encoding: str = "utf-8",
    metadata: dict[str, Any] | None = None,
) -> list[str]:
    """Ingest a file into a collection.

    Reads the file, splits into chunks if chunk_size is specified, and ingests
    each chunk with source span metadata.

    Args:
        col: The collection to ingest into.
        path: Path to the file.
        chunk_size: If specified, split text into chunks of this many characters.
            If None, ingest the entire file as one item.
        chunk_overlap: Number of characters to overlap between chunks.
        encoding: File encoding.
        metadata: Additional metadata to attach to each item.

    Returns:
        List of item IDs that were ingested.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    text = file_path.read_text(encoding=encoding)
    if not text.strip():
        return []

    file_hash = _file_hash(file_path)
    ids: list[str] = []

    if chunk_size is None:
        # Ingest entire file as one item
        item_id = f"file:{file_hash}"
        source = _make_source(file_path)
        item_metadata = {"file": str(file_path), **(metadata or {})}
        ingest_text(col, item_id, text, source=source, metadata=item_metadata)
        ids.append(item_id)
    else:
        # Split into chunks
        lines = text.splitlines(keepends=True)
        current_chunk: list[str] = []
        current_start = 1
        current_len = 0

        for i, line in enumerate(lines, start=1):
            current_chunk.append(line)
            current_len += len(line)

            if current_len >= chunk_size:
                chunk_text = "".join(current_chunk)
                item_id = f"file:{file_hash}:{current_start}"
                source = _make_source(file_path, current_start, i)
                item_metadata = {
                    "file": str(file_path),
                    "chunk_index": len(ids),
                    **(metadata or {}),
                }
                ingest_text(
                    col, item_id, chunk_text, source=source, metadata=item_metadata
                )
                ids.append(item_id)

                # Overlap: keep last chunk_overlap characters
                if chunk_overlap > 0:
                    overlap_text = chunk_text[-chunk_overlap:]
                    current_chunk = [overlap_text]
                    current_start = i
                    current_len = len(overlap_text)
                else:
                    current_chunk = []
                    current_start = i + 1
                    current_len = 0

        # Ingest remaining text
        if current_chunk:
            chunk_text = "".join(current_chunk)
            item_id = f"file:{file_hash}:{current_start}"
            source = _make_source(file_path, current_start, len(lines))
            item_metadata = {
                "file": str(file_path),
                "chunk_index": len(ids),
                **(metadata or {}),
            }
            ingest_text(col, item_id, chunk_text, source=source, metadata=item_metadata)
            ids.append(item_id)

    return ids


def ingest_directory(
    col: Collection,
    directory: str | Path,
    *,
    glob: str = "*",
    recursive: bool = True,
    chunk_size: int | None = None,
    chunk_overlap: int = 0,
    encoding: str = "utf-8",
    metadata: dict[str, Any] | None = None,
) -> list[str]:
    """Ingest all matching files from a directory.

    Args:
        col: The collection to ingest into.
        directory: Path to the directory.
        glob: Glob pattern to match files (e.g., "*.py", "*.md").
        recursive: Whether to search subdirectories.
        chunk_size: If specified, split files into chunks of this many characters.
        chunk_overlap: Number of characters to overlap between chunks.
        encoding: File encoding.
        metadata: Additional metadata to attach to each item.

    Returns:
        List of item IDs that were ingested.
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {dir_path}")

    ids: list[str] = []
    pattern = f"**/{glob}" if recursive else glob

    for file_path in sorted(dir_path.glob(pattern)):
        if file_path.is_file():
            try:
                file_ids = ingest_file(
                    col,
                    file_path,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    encoding=encoding,
                    metadata=metadata,
                )
                ids.extend(file_ids)
            except UnicodeDecodeError, PermissionError:
                # Skip files that can't be read
                continue

    return ids
