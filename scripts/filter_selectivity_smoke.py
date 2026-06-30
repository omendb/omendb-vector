from __future__ import annotations

import argparse
import random
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import omendb


@dataclass(frozen=True, slots=True)
class Case:
    name: str
    selectivity: float
    correlated: bool


CASES = [
    Case("correlated_50pct", 0.50, True),
    Case("correlated_5pct", 0.05, True),
    Case("correlated_1pct", 0.01, True),
    Case("uncorrelated_50pct", 0.50, False),
    Case("uncorrelated_5pct", 0.05, False),
    Case("uncorrelated_1pct", 0.01, False),
]


def run(root: Path, n: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(7)

    print("case,n,selectivity,correlated,matches,planner,top_id,elapsed_ms")
    for case in CASES:
        db = omendb.create(root / case.name, exist_ok=True)
        docs = db.collection("docs", config=omendb.CollectionConfig(dim=2))
        matches = max(1, int(n * case.selectivity))
        match_ids = set(_match_ids(n, matches, case.correlated, rng))
        for i in range(n):
            metadata = {"bucket": "match"} if i in match_ids else {"bucket": "other"}
            docs.set(f"doc-{i}", vector=[float(i), 0.0], metadata=metadata)

        started = time.perf_counter()
        results = docs.search_vector(
            [0.0, 0.0], k=10, filter={"bucket": "match"}, ef=64
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        planner = "exact"
        top_id = results[0].id if results else ""
        assert results
        assert all(result.metadata == {"bucket": "match"} for result in results)
        print(
            f"{case.name},{n},{case.selectivity},{case.correlated},"
            f"{matches},{planner},{top_id},{elapsed_ms:.3f}"
        )


def _match_ids(n: int, matches: int, correlated: bool, rng: random.Random) -> list[int]:
    if correlated:
        return list(range(matches))
    return rng.sample(range(n), matches)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--n", type=int, default=1000)
    args = parser.parse_args()

    if args.root is not None:
        run(args.root, args.n)
        return

    with tempfile.TemporaryDirectory(prefix="omendb-filter-selectivity-") as tmp:
        run(Path(tmp), args.n)


if __name__ == "__main__":
    main()
