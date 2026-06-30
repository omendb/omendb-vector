from __future__ import annotations

import argparse

from ._api import check
from ._native import engine_version


def engine_check() -> None:
    print(engine_version())


def store_check(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="omendb-check",
        description="Verify a persistent OmenDB store.",
    )
    parser.add_argument("path", help="database root path")
    args = parser.parse_args(argv)

    result = check(args.path)
    if result.ok:
        print(f"ok: checked {result.collections_checked} collections")
        return

    print(
        f"failed: checked {result.collections_checked} collections; "
        f"{len(result.issues)} issues"
    )
    for issue in result.issues:
        location = issue.collection or "<database>"
        path = f" ({issue.path})" if issue.path else ""
        print(f"{location}: {issue.code}: {issue.message}{path}")
    raise SystemExit(1)
