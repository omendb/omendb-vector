from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE_SOURCE = ROOT / "src" / "python_engine.mojo"
ENGINE_OUTPUT = ROOT / "src" / "omendb" / "_omendb_engine.so"


def main() -> None:
    subprocess.run(
        [
            "pixi",
            "run",
            "mojo",
            "build",
            str(ENGINE_SOURCE),
            "--emit",
            "shared-lib",
            "-o",
            str(ENGINE_OUTPUT),
        ],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
