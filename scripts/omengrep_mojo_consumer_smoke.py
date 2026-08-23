from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OMENGREP_REPO = REPO_ROOT.parent.parent / "nijaru" / "omengrep"
DEFAULT_TARGET_DIR = Path(tempfile.gettempdir()) / "omengrep-mojo-consumer-target"


def run(args: argparse.Namespace) -> None:
    omengrep_repo = args.omengrep_repo.expanduser().resolve()
    if not (omengrep_repo / "Cargo.toml").exists():
        raise FileNotFoundError(f"{omengrep_repo} is not an omengrep checkout")

    mojo_rust_crate = (args.omendb_vector_mojo_root / "rust" / "omendb_vector").resolve()
    if not (mojo_rust_crate / "Cargo.toml").exists():
        raise FileNotFoundError(f"{mojo_rust_crate} is not the Rust OmenDB Vector shim")

    work_parent = args.workdir or Path(tempfile.mkdtemp(prefix="omengrep-mojo-smoke-"))
    work_parent = work_parent.expanduser().resolve()
    work_parent.mkdir(parents=True, exist_ok=True)
    workdir = work_parent / "omengrep"
    if workdir.exists():
        shutil.rmtree(workdir)

    shutil.copytree(
        omengrep_repo,
        workdir,
        ignore=shutil.ignore_patterns(".git", ".og", ".ruff_cache", "target"),
    )
    _patch_omendb_vector_dependency(workdir / "Cargo.toml", mojo_rust_crate)

    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(args.target_dir.expanduser().resolve())
    command = [
        "cargo",
        "check",
        "--manifest-path",
        str(workdir / "Cargo.toml"),
        "--all-targets",
    ]
    print("running:", " ".join(command))
    print("patched omendb_vector path:", mojo_rust_crate)
    subprocess.run(command, cwd=workdir, env=env, check=True)

    if not args.check_only:
        _run_cli_smoke(
            omendb_vector_mojo_root=args.omendb_vector_mojo_root.resolve(),
            omengrep_workdir=workdir,
            omengrep_repo=omengrep_repo,
            work_parent=work_parent,
            env=env,
        )

    if not args.keep_workdir:
        shutil.rmtree(work_parent, ignore_errors=True)


def _patch_omendb_vector_dependency(cargo_toml: Path, mojo_rust_crate: Path) -> None:
    text = cargo_toml.read_text(encoding="utf-8")
    replacement = f'omendb_vector = {{ path = "{mojo_rust_crate.as_posix()}" }}'
    patched, count = re.subn(
        r"^omendb_vector\s*=.*$",
        replacement,
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise ValueError(f"could not find omendb_vector dependency in {cargo_toml}")
    cargo_toml.write_text(patched, encoding="utf-8")


def _run_cli_smoke(
    *,
    omendb_vector_mojo_root: Path,
    omengrep_workdir: Path,
    omengrep_repo: Path,
    work_parent: Path,
    env: dict[str, str],
) -> None:
    dylib_path = work_parent / "libomendb_vector_multivector_c_abi.dylib"
    mojo_command = [
        "pixi",
        "run",
        "mojo",
        "build",
        "src/multivector_c_abi.mojo",
        "--emit",
        "shared-lib",
        "-o",
        str(dylib_path),
    ]
    print("running:", " ".join(mojo_command))
    subprocess.run(mojo_command, cwd=omendb_vector_mojo_root, env=env, check=True)

    fixture_dir = work_parent / "fixture"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    for source in (omengrep_repo / "tests" / "golden").iterdir():
        if source.is_file():
            shutil.copy2(source, fixture_dir / source.name)

    runtime_env = env | {"OMENDB_VECTOR_MOJO_DYLIB": str(dylib_path)}
    build_command = [
        "cargo",
        "run",
        "--manifest-path",
        str(omengrep_workdir / "Cargo.toml"),
        "--quiet",
        "--",
        "build",
        str(fixture_dir),
    ]
    print("running:", " ".join(build_command))
    subprocess.run(build_command, cwd=omengrep_workdir, env=runtime_env, check=True)

    search_command = [
        "cargo",
        "run",
        "--manifest-path",
        str(omengrep_workdir / "Cargo.toml"),
        "--quiet",
        "--",
        "--json",
        "error",
        str(fixture_dir),
        "-n",
        "1",
    ]
    print("running:", " ".join(search_command))
    search = subprocess.run(
        search_command,
        cwd=omengrep_workdir,
        env=runtime_env,
        check=True,
        capture_output=True,
        text=True,
    )
    results = json.loads(search.stdout)
    if not isinstance(results, list) or not results:
        raise AssertionError("omengrep search returned no JSON results")
    print("top result:", results[0].get("file", "<missing file>"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compile the real omengrep checkout against the local Mojo C ABI Rust "
            "shim without modifying the omengrep worktree."
        )
    )
    parser.add_argument("--omengrep-repo", type=Path, default=DEFAULT_OMENGREP_REPO)
    parser.add_argument("--omendb-vector-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET_DIR)
    parser.add_argument(
        "--workdir",
        type=Path,
        help="Optional parent directory for the temporary patched omengrep copy.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only compile omengrep against the local shim; skip CLI runtime smoke.",
    )
    parser.add_argument("--keep-workdir", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
