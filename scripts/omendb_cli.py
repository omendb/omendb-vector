#!/usr/bin/env python3
"""OmenDB CLI for backup, export, import, and maintenance operations.

Usage:
    python omendb_cli.py snapshot <db_path> <snapshot_path>
    python omendb_cli.py import <db_path> <snapshot_path> [--replace]
    python omendb_cli.py check <db_path>
    python omendb_cli.py vacuum <db_path> <collection_name>
    python omendb_cli.py info <db_path>
"""

from __future__ import annotations

import argparse
import sys

import omendb


def cmd_snapshot(args: argparse.Namespace) -> int:
    """Create a snapshot of the database."""
    try:
        db = omendb.open(args.db_path, create=False)
        db.snapshot(args.snapshot_path)
        print(f"Snapshot created: {args.snapshot_path}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_import(args: argparse.Namespace) -> int:
    """Import a snapshot into the database."""
    try:
        db = omendb.open(args.db_path, create=True)
        db.import_snapshot(args.snapshot_path, replace=args.replace)
        print(f"Snapshot imported: {args.snapshot_path}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_check(args: argparse.Namespace) -> int:
    """Check database integrity."""
    try:
        db = omendb.open(args.db_path, create=False)
        result = db.check()
        if result.ok:
            print("Database integrity check passed")
            return 0
        else:
            print("Database integrity check failed:")
            for issue in result.issues:
                print(f"  - {issue}")
            return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_vacuum(args: argparse.Namespace) -> int:
    """Vacuum a collection to reclaim space."""
    try:
        db = omendb.open(args.db_path, create=False)
        col = db.collection(args.collection_name, create=False)
        col.vacuum()
        print(f"Vacuumed collection: {args.collection_name}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_info(args: argparse.Namespace) -> int:
    """Show database information."""
    try:
        db = omendb.open(args.db_path, create=False)
        print(f"Database: {args.db_path}")
        # Note: Collection listing not yet exposed; this is a placeholder
        print("Use Python API for detailed collection inspection")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="OmenDB CLI for backup, export, import, and maintenance"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Snapshot command
    snapshot_parser = subparsers.add_parser(
        "snapshot", help="Create a database snapshot"
    )
    snapshot_parser.add_argument("db_path", help="Path to the database")
    snapshot_parser.add_argument("snapshot_path", help="Path for the snapshot")

    # Import command
    import_parser = subparsers.add_parser("import", help="Import a snapshot")
    import_parser.add_argument("db_path", help="Path to the database")
    import_parser.add_argument("snapshot_path", help="Path to the snapshot")
    import_parser.add_argument(
        "--replace", action="store_true", help="Replace existing database"
    )

    # Check command
    check_parser = subparsers.add_parser("check", help="Check database integrity")
    check_parser.add_argument("db_path", help="Path to the database")

    # Vacuum command
    vacuum_parser = subparsers.add_parser("vacuum", help="Vacuum a collection")
    vacuum_parser.add_argument("db_path", help="Path to the database")
    vacuum_parser.add_argument("collection_name", help="Name of the collection")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show database information")
    info_parser.add_argument("db_path", help="Path to the database")

    args = parser.parse_args()

    if args.command == "snapshot":
        return cmd_snapshot(args)
    elif args.command == "import":
        return cmd_import(args)
    elif args.command == "check":
        return cmd_check(args)
    elif args.command == "vacuum":
        return cmd_vacuum(args)
    elif args.command == "info":
        return cmd_info(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
