from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE_TY_TARGETS = [
    "src/omendb",
    "tests/test_python_api.py",
    "scripts/agent_memory_smoke.py",
    "scripts/agent_memory_release_smoke.py",
    "scripts/build_python_engine.py",
    "scripts/validate_p2_storage.py",
    "scripts/validate_release_gate.py",
]
GRAPH_SMOKE = """
import tempfile
from pathlib import Path

import omendb

with tempfile.TemporaryDirectory(prefix="omendb-graph-wheel-") as tmp:
    docs = omendb.create(Path(tmp) / "db").collection(
        "docs", config=omendb.CollectionConfig(dim=2, graph=True)
    )
    docs.set("a", vector=[0.0, 0.0])
    docs.set("b", vector=[1.0, 0.0])
    docs.add_relationship("a", "b", type="references")
    assert docs.neighbors("a", type="references") == ["b"]
    docs.flush()
    reopened = omendb.open(Path(tmp) / "db", create=False).collection(
        "docs", create=False
    )
    assert reopened.shortest_path("a", "b") == ["a", "b"]

    wide = omendb.memory().collection("wide", config=omendb.CollectionConfig(dim=128))
    origin = [0.0] * 128
    right = [0.0] * 128
    right[0] = 1.0
    wide.set("origin", vector=origin)
    wide.set("right", vector=right)
    assert [row.id for row in wide.search_vector([0.1] + ([0.0] * 127), k=2)] == [
        "origin",
        "right",
    ]
"""


def run(command: Sequence[str], *, cwd: Path = ROOT) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def standard_python() -> str:
    configured = os.environ.get("OMENDB_STANDARD_PYTHON")
    if configured:
        return configured

    if "t" not in sys.abiflags:
        return sys.executable

    candidates = [
        "/opt/homebrew/bin/python3.14",
        str(Path.home() / ".local/share/mise/installs/python/latest/bin/python3.14"),
        shutil.which("python3.14"),
        shutil.which("python3"),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        resolved = str(Path(candidate))
        try:
            abiflags = subprocess.check_output(
                [resolved, "-c", "import sys; print(sys.abiflags)"],
                text=True,
            ).strip()
        except OSError, subprocess.CalledProcessError:
            continue
        if "t" not in abiflags:
            print(f"standard CPython: {resolved}", flush=True)
            return resolved

    raise RuntimeError(
        "native engine validation requires standard CPython; set "
        "OMENDB_STANDARD_PYTHON=/path/to/python"
    )


def mojo_tests() -> None:
    run(["pixi", "run", "mojo", "format", "src/"])
    for path in sorted((ROOT / "src").glob("*_test.mojo")):
        run(["pixi", "run", "mojo", "run", str(path.relative_to(ROOT))])


def python_checks() -> None:
    python = standard_python()
    run(["uv", "run", "ruff", "check", "."])
    run(["uv", "run", "ruff", "format", "--check", "."])
    run(["uv", "run", "ty", "check", *CORE_TY_TARGETS])
    run(["uv", "run", "--python", python, "python", "scripts/build_python_engine.py"])
    run(["uv", "run", "--python", python, "pytest"])
    run(["uv", "run", "--python", python, "python", "scripts/agent_memory_smoke.py"])


def wheel_smoke() -> None:
    standard = standard_python()
    run(["uv", "build", "--wheel"])
    wheels = sorted(
        (ROOT / "dist").glob("omendb-*.whl"), key=lambda p: p.stat().st_mtime
    )
    if not wheels:
        raise RuntimeError("uv build did not produce an omendb wheel")

    wheel = wheels[-1]
    with tempfile.TemporaryDirectory(prefix="omendb-p2-wheel-") as tmp:
        venv = Path(tmp) / "venv"
        run(["uv", "venv", "--python", standard, str(venv)])
        python = venv / "bin" / "python"
        run(["uv", "pip", "install", "--python", str(python), str(wheel)])
        run([str(python), "scripts/agent_memory_smoke.py"])
        run([str(python), "-c", GRAPH_SMOKE], cwd=Path(tmp))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-wheel",
        action="store_true",
        help="skip wheel build/install smoke for faster local iteration",
    )
    args = parser.parse_args()

    mojo_tests()
    python_checks()
    if not args.skip_wheel:
        wheel_smoke()


if __name__ == "__main__":
    main()
