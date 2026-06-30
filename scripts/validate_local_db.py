from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = "/opt/homebrew/bin/python3.14"
type CommandRow = dict[str, object]
RUFF_TARGETS = [
    "src/omendb",
    "tests/test_python_api.py",
    "benchmarks/real_data",
    "benchmarks/local_competitors",
    "scripts",
]
TY_TARGETS = [
    "benchmarks/real_data",
    "src/omendb",
    "tests/test_python_api.py",
    "scripts",
]


def default_artifact_dir() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "ai" / "tmp" / "local-db-validation" / timestamp


def controlled_env() -> dict[str, str]:
    env = os.environ.copy()
    env["UV_NO_CONFIG"] = "1"
    return env


def standard_python(configured: str | None) -> str:
    candidates = []
    if configured is not None:
        candidates.append(configured)
    candidates.extend(
        [
            os.environ.get("OMENDB_STANDARD_PYTHON"),
            DEFAULT_PYTHON,
            shutil.which("python3.14"),
            shutil.which("python3"),
        ]
    )
    for candidate in candidates:
        if candidate is None:
            continue
        resolved = str(Path(candidate))
        try:
            abiflags = subprocess.check_output(
                [resolved, "-c", "import sys; print(sys.abiflags)"],
                cwd=ROOT,
                text=True,
            ).strip()
        except OSError, subprocess.CalledProcessError:
            continue
        if "t" not in abiflags:
            return resolved
    raise RuntimeError("standard CPython 3.14 is required for native engine smoke")


def run(
    name: str,
    command: Sequence[str],
    *,
    artifact_dir: Path,
    env: dict[str, str] | None = None,
) -> dict[str, object]:
    print("+", " ".join(command), flush=True)
    started = datetime.now(UTC)
    log_path = artifact_dir / f"{name}.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(command),
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        returncode = process.wait()
    ended = datetime.now(UTC)
    row: dict[str, object] = {
        "name": name,
        "command": list(command),
        "returncode": returncode,
        "log": str(log_path),
        "started_at": started.isoformat(),
        "ended_at": ended.isoformat(),
        "duration_seconds": (ended - started).total_seconds(),
    }
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, list(command))
    return row


def run_mojo_tests(artifact_dir: Path, commands: list[CommandRow]) -> None:
    commands.append(
        run(
            "mojo_version",
            ["pixi", "run", "mojo", "--version"],
            artifact_dir=artifact_dir,
        )
    )
    for path in sorted((ROOT / "src").glob("*_test.mojo")):
        name = f"mojo_{path.stem}"
        commands.append(
            run(
                name,
                ["pixi", "run", "mojo", "run", str(path.relative_to(ROOT))],
                artifact_dir=artifact_dir,
            )
        )


def latest_wheel() -> Path:
    wheels = sorted(
        (ROOT / "dist").glob("omendb-*.whl"), key=lambda path: path.stat().st_mtime
    )
    if not wheels:
        raise RuntimeError("uv build did not produce an omendb wheel")
    return wheels[-1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the local embedded OmenDB DB surface."
    )
    parser.add_argument("--python", default=None, help="standard CPython 3.14 path")
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--skip-mojo", action="store_true")
    parser.add_argument("--skip-wheel", action="store_true")
    args = parser.parse_args()

    artifact_dir = args.artifact_dir or default_artifact_dir()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    python = standard_python(args.python)
    env = controlled_env()
    lock_path = ROOT / "uv.lock"
    original_lock = lock_path.read_bytes() if lock_path.exists() else None
    lock_restored = False
    commands: list[CommandRow] = []
    manifest: dict[str, object] = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "artifact_dir": str(artifact_dir),
        "python": python,
        "uv_no_config": env.get("UV_NO_CONFIG"),
        "commands": commands,
    }

    try:
        commands.append(
            run(
                "ruff_check",
                ["uv", "run", "--python", python, "ruff", "check", *RUFF_TARGETS],
                artifact_dir=artifact_dir,
                env=env,
            )
        )
        commands.append(
            run(
                "ty_check",
                ["uv", "run", "--python", python, "ty", "check", *TY_TARGETS],
                artifact_dir=artifact_dir,
                env=env,
            )
        )
        commands.append(
            run(
                "python_api_tests",
                [
                    "uv",
                    "run",
                    "--python",
                    python,
                    "pytest",
                    "tests/test_python_api.py",
                ],
                artifact_dir=artifact_dir,
                env=env,
            )
        )
        if not args.skip_mojo:
            run_mojo_tests(artifact_dir, commands)
        if not args.skip_wheel:
            commands.append(
                run(
                    "wheel_build",
                    ["uv", "build", "--wheel", "--clear"],
                    artifact_dir=artifact_dir,
                    env=env,
                )
            )
            wheel = latest_wheel()
            manifest["wheel"] = str(wheel)
            commands.append(
                run(
                    "wheel_install_smoke",
                    [
                        python,
                        "scripts/validate_published_package.py",
                        "--python",
                        python,
                        "--package",
                        str(wheel),
                    ],
                    artifact_dir=artifact_dir,
                    env=env,
                )
            )
    finally:
        if original_lock is not None and lock_path.exists():
            current_lock = lock_path.read_bytes()
            if current_lock != original_lock:
                lock_path.write_bytes(original_lock)
                lock_restored = True
        manifest["uv_lock_restored"] = lock_restored
        manifest["completed_at"] = datetime.now(UTC).isoformat()
        (artifact_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(f"local_db_validation_artifact={artifact_dir}")


if __name__ == "__main__":
    main()
