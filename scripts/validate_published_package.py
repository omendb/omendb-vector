from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

SMOKE = """
from pathlib import Path
import tempfile

import omendb

with tempfile.TemporaryDirectory(prefix="omendb-published-smoke-") as tmp:
    db = omendb.create(Path(tmp) / "db")
    docs = db.collection(
        "docs", config=omendb.CollectionConfig(dim=128, text=True, graph=True)
    )
    origin = [0.0] * 128
    right = [0.0] * 128
    right[0] = 1.0
    docs.set("origin", vector=origin, text="install package")
    docs.set("right", vector=right, text="release validation")
    docs.add_relationship("origin", "right", type="related")
    docs.flush()

    reopened = omendb.open(Path(tmp) / "db", create=False).collection(
        "docs", create=False
    )
    assert reopened.search_vector([0.1] + ([0.0] * 127), k=2)[0].id == "origin"
    assert reopened.search_text("validation", k=1)[0].id == "right"
    assert reopened.neighbors("origin", type="related") == ["right"]

print("published_package_smoke: ok")
"""


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--package", default="omendb==0.1.0")
    parser.add_argument("--index-url", default=None)
    parser.add_argument("--extra-index-url", default=None)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="omendb-published-package-") as tmp:
        root = Path(tmp)
        venv = root / "venv"
        run(["uv", "venv", "--python", args.python, str(venv)])
        python = venv / "bin" / "python"
        engine_check = venv / "bin" / "omendb-engine-check"

        install = ["uv", "pip", "install", "--python", str(python)]
        if args.index_url is not None:
            install.extend(["--index-url", args.index_url])
        if args.extra_index_url is not None:
            install.extend(["--extra-index-url", args.extra_index_url])
        install.append(args.package)
        run(install)

        run([str(engine_check)])
        run([str(python), "-c", SMOKE])


if __name__ == "__main__":
    main()
