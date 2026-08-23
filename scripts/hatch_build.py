from __future__ import annotations

import subprocess
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict[str, object]) -> None:
        build_data["pure_python"] = False
        build_data["tag"] = "cp314-cp314-macosx_11_0_arm64"
        if version != "standard":
            return

        root = Path(__file__).resolve().parents[1]
        subprocess.run(
            [
                "pixi",
                "run",
                "mojo",
                "build",
                str(root / "src" / "python_engine.mojo"),
                "--emit",
                "shared-lib",
                "-o",
                str(root / "src" / "omendb_vector" / "_omendb_vector_engine.so"),
            ],
            cwd=root,
            check=True,
        )
