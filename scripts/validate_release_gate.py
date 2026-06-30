from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIFT_RECALL_FLOORS = {100: 0.994}


def run(command: list[str], *, log_path: Path | None = None) -> str:
    print("+", " ".join(command), flush=True)
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output: list[str] = []
    assert process.stdout is not None
    if log_path is None:
        for line in process.stdout:
            print(line, end="", flush=True)
            output.append(line)
    else:
        with log_path.open("w") as log:
            for line in process.stdout:
                print(line, end="", flush=True)
                output.append(line)
                log.write(line)
    returncode = process.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)
    return "".join(output)


def parse_sift_recall(output: str) -> dict[int, float]:
    recalls: dict[int, float] = {}
    # Parse the actual sift_benchmark.mojo output format
    for line in output.splitlines():
        if line.startswith("Recall@10:"):
            recall = float(line.split(":")[1].strip())
            # Use ef_search from the ef_s line
            for ef_line in output.splitlines():
                if ef_line.startswith("ef_s:"):
                    ef_search = int(ef_line.split(":")[1].strip())
                    recalls[ef_search] = recall
                    break
    return recalls


def validate_sift_recall(recalls: dict[int, float]) -> None:
    for ef_search, floor in SIFT_RECALL_FLOORS.items():
        recall = recalls.get(ef_search)
        if recall is None:
            raise RuntimeError(f"missing SIFT-100K ef={ef_search} recall row")
        if recall < floor:
            raise RuntimeError(
                f"SIFT-100K ef={ef_search} recall {recall} below floor {floor}"
            )


def default_artifact_dir() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "ai" / "tmp" / "release-validation" / timestamp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--skip-p2", action="store_true")
    parser.add_argument("--skip-wheel", action="store_true")
    parser.add_argument("--skip-sift100k", action="store_true")
    args = parser.parse_args()

    artifact_dir = args.artifact_dir or default_artifact_dir()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "created_at": datetime.now(UTC).isoformat(),
        "artifact_dir": str(artifact_dir),
        "commands": [],
    }

    if not args.skip_p2:
        command = [sys.executable, "scripts/validate_p2_storage.py"]
        if args.skip_wheel:
            command.append("--skip-wheel")
        manifest["commands"].append(command)
        run(command, log_path=artifact_dir / "validate_p2_storage.log")

    # Run release smoke test in a venv with omendb installed
    wheels = sorted(
        (ROOT / "dist").glob("omendb-*.whl"),
        key=lambda p: p.stat().st_mtime,
    )
    if wheels:
        with tempfile.TemporaryDirectory(prefix="omendb-release-smoke-") as tmp:
            venv = Path(tmp) / "venv"
            run(["uv", "venv", "--python", sys.executable, str(venv)])
            python = venv / "bin" / "python"
            run(
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    str(python),
                    str(wheels[-1]),
                ]
            )
            release_smoke = [
                str(python),
                "scripts/agent_memory_release_smoke.py",
            ]
            manifest["commands"].append(release_smoke)
            run(
                release_smoke,
                log_path=artifact_dir / "agent_memory_release_smoke.log",
            )
    else:
        print("WARNING: No wheel found, skipping release smoke test")

    if not args.skip_sift100k:
        sift_command = ["pixi", "run", "mojo", "run", "src/sift_benchmark.mojo", "100k"]
        manifest["commands"].append(sift_command)
        sift_output = run(sift_command, log_path=artifact_dir / "sift100k.log")
        recalls = parse_sift_recall(sift_output)
        validate_sift_recall(recalls)
        manifest["sift100k_recall_at_10"] = recalls
        manifest["sift100k_recall_floors"] = SIFT_RECALL_FLOORS

    (artifact_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"release_validation_artifact={artifact_dir}")


if __name__ == "__main__":
    main()
