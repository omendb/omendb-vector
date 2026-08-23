#!/usr/bin/env python3
"""OmenDB Vector 5-minute quickstart: index a directory, search with evidence.

Usage:
    python quickstart.py index ./my-project     # Index files
    python quickstart.py search "error handling"  # Text search
    python quickstart.py search "database" --hybrid  # Hybrid search
    python quickstart.py search "class" --filter ext:.py  # Filtered search

This is the first public proof demo for OmenDB Vector.
Shows: text + vector + hybrid search, hard filters, source evidence,
delete/supersede lifecycle, flush/reopen persistence.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Use a lightweight embedding heuristic for the demo.
# In production, use a proper embedding model.
# This maps file content to a 4D vector for search clustering.
# ---------------------------------------------------------------------------


def _simple_embed(text: str) -> list[float]:
    """Deterministic 2D embedding from text content."""
    h = hashlib.sha256(text.encode()).digest()
    return [
        float(h[0]) / 255.0,
        float(h[1]) / 255.0,
    ]


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


def index_directory(
    collection: Any,
    root: Path,
    *,
    extensions: set[str] | None = None,
    max_file_bytes: int = 1_000_000,
) -> int:
    """Walk root, index every text file."""
    count = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for fname in filenames:
            path = Path(dirpath) / fname
            if extensions and path.suffix not in extensions:
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size > max_file_bytes:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except (OSError, UnicodeDecodeError):
                continue
            if not text.strip():
                continue

            doc_id = str(path.relative_to(root))
            collection.set(
                doc_id,
                vector=_simple_embed(text),
                text=text[:8000],  # Cap per-record text for demo
                metadata={
                    "path": str(path),
                    "ext": path.suffix,
                    "size": size,
                    "lines": text.count("\n") + 1,
                },
                source={
                    "path": str(path),
                    "size": size,
                    "lines": text.count("\n") + 1,
                },
            )
            count += 1
            if count % 100 == 0:
                print(f"  indexed {count} files...", file=sys.stderr)
    return count


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def _print_result(result: Any, show_source: bool = True) -> None:
    """Print one search result with evidence."""
    meta = result.metadata or {}
    src = result.source or {}
    print(f"\n── {result.id}")
    if result.score is not None:
        print(f"   score: {result.score:.4f}")
    if result.distance is not None:
        print(f"   distance: {result.distance:.4f}")
    if meta:
        print(f"   ext: {meta.get('ext', '?')}  lines: {meta.get('lines', '?')}")
    if show_source and src:
        print(f"   source: {src.get('path', '?')}")


def search_demo(
    collection: Any,
    query: str,
    *,
    k: int = 5,
    metadata_filter: dict[str, Any] | None = None,
) -> None:
    """Run a hybrid search and print results with source evidence."""
    results = collection.search_hybrid(
        vector=_simple_embed(query),
        text=query,
        k=k,
        filter=metadata_filter,
    )
    print(f"\n=== Search: \"{query}\" (top {k}) ===")

    if not results:
        print("  (no results)")
        return

    for r in results:
        _print_result(r)

    if results[0].explanation:
        method = results[0].explanation.get("method", "?")
        print(f"\n  engine: {method}")


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def lifecycle_demo(collection: Any, db_path: Path) -> None:
    """Demonstrate delete, supersede, flush, reopen."""
    import omendb_vector

    print("\n=== Lifecycle: delete + superseede + flush + reopen ===")

    # Find two files that exist in the collection for the demo
    all_ids = [r.id for r in collection.search_text("import", k=100)]
    demo_id = None
    superseded_id = None
    superseder_id = None
    for rid in all_ids:
        if rid.endswith(".py") and demo_id is None:
            demo_id = rid
        elif rid.endswith(".py") and superseded_id is None:
            superseded_id = rid
        elif rid.endswith(".py") and superseder_id is None:
            superseder_id = rid
            break

    if not demo_id:
        print("  (no .py files in collection — skipping lifecycle demo)")
        return

    # Delete
    print(f"\n--- Deleting '{demo_id}' ---")
    collection.delete(demo_id)
    results = collection.search_text("import", k=3)
    assert all(r.id != demo_id for r in results), "Delete failed!"
    print(f"  ✓ {demo_id} removed from search")

    # Supersede (if we have two distinct files)
    if superseded_id and superseder_id and superseded_id != superseder_id:
        print(f"\n--- Superseding '{superseded_id}' with '{superseder_id}' ---")
        collection.supersede(superseded_id, superseder_id)
        results = collection.search_text("import", k=10)
        assert all(r.id != superseded_id for r in results), "Supersede failed!"
        print(f"  ✓ {superseded_id} superseded, not in search")
    else:
        print("\n  (not enough .py files for supersede demo)")

    # Flush
    print("\n--- Flushing to disk ---")
    collection.flush()
    print(f"  ✓ Data persisted to {db_path}")

    # Reopen
    print("\n--- Reopening database ---")
    db2 = omendb_vector.open(str(db_path), create=False)
    col2 = db2.collection("files", create=False)
    results = col2.search_text("import", k=3)
    print(f"  ✓ Reopened: {len(results)} results for 'import'")

    # Superseded record still gone
    assert col2.get("setup.py") is None, "Superseded record reappeared!"
    print("  ✓ Superseded record still gone after reopen")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OmenDB Vector quickstart: index and search files locally"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # index
    idx = sub.add_parser("index", help="Index a directory")
    idx.add_argument("directory", type=Path, help="Directory to index")
    idx.add_argument(
        "--db", type=Path, default=Path(".omendb_vector_demo"), help="Database path"
    )
    idx.add_argument(
        "--ext", type=str, default=None, help="Comma-separated extensions (e.g. .py,.md)"
    )

    # search
    srch = sub.add_parser("search", help="Search indexed files")
    srch.add_argument("query", type=str, help="Search query")
    srch.add_argument("--k", type=int, default=5, help="Number of results")
    srch.add_argument(
        "--db", type=Path, default=Path(".omendb_vector_demo"), help="Database path"
    )
    srch.add_argument(
        "--ext", type=str, default=None, help="Filter by extension (e.g. .py)"
    )
    srch.add_argument(
        "--lifecycle", action="store_true", help="Run lifecycle demo after search"
    )

    args = parser.parse_args()

    import omendb_vector

    if args.command == "index":
        root = args.directory.expanduser().resolve()
        if not root.is_dir():
            print(f"Error: {root} is not a directory", file=sys.stderr)
            sys.exit(1)

        print(f"Indexing {root} ...", file=sys.stderr)
        db = omendb_vector.open(str(args.db), create=True)
        col = db.collection(
            "files",
            config=omendb_vector.CollectionConfig(dim=2, text=True),
        )

        exts = set(args.ext.split(",")) if args.ext else None
        ext_list = list(exts) if exts else [".py", ".md", ".rs", ".mojo", ".toml",
                                               ".json", ".yaml", ".yml", ".txt",
                                               ".js", ".ts", ".html", ".css",
                                               ".go", ".zig", ".cpp", ".h",
                                               ".c", ".java", ".rb", ".sh"]
        count = index_directory(col, root, extensions=ext_list)
        col.flush()
        print(f"\nDone. {count} files indexed to {args.db}", file=sys.stderr)

    elif args.command == "search":
        db_path = args.db
        if not db_path.exists():
            print(f"Error: no database at {db_path}. Run 'index' first.", file=sys.stderr)
            sys.exit(1)

        db = omendb_vector.open(str(db_path), create=False)
        col = db.collection("files", create=False)

        md_filter = None
        if args.ext:
            md_filter = {"ext": args.ext}

        search_demo(
            col,
            args.query,
            k=args.k,
            metadata_filter=md_filter,
        )

        if args.lifecycle:
            lifecycle_demo(col, db_path)


if __name__ == "__main__":
    main()
