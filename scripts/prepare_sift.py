#!/usr/bin/env python3
"""Downloads SIFT datasets and converts them to raw binary files for Mojo."""

import argparse
import ast
import os
import shutil
import struct
import urllib.request
import zipfile
from pathlib import Path

HF_BASE = "https://huggingface.co/datasets/hhy3/ann-datasets/resolve/main"

DATASETS = {
    "siftsmall": {
        "url": f"{HF_BASE}/siftsmall-128-euclidean.hdf5",
        "filename": "siftsmall-128-euclidean.hdf5",
    },
    "sift1m": {
        "url": f"{HF_BASE}/sift-128-euclidean.hdf5",
        "filename": "sift-128-euclidean.hdf5",
    },
    "sift100k": {
        "format": "local",
        "filename": "sift100k_manifest.txt",
    },
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def convert_vecs(src: Path, dst: Path, rows: int, dim: int, kind: str) -> None:
    """Convert fvecs/ivecs with per-row dim headers to raw little-endian rows."""
    if dst.exists():
        print(f"Found {dst}")
        return

    print(f"Converting {src} -> {dst}")
    elem_size = 4
    with src.open("rb") as f, dst.open("wb") as out:
        for row in range(rows):
            header = f.read(4)
            if len(header) != 4:
                raise EOFError(f"{src}: missing row header at row {row}")
            (row_dim,) = struct.unpack("<i", header)
            if row_dim != dim:
                raise ValueError(f"{src}: row {row} dim {row_dim}, expected {dim}")
            payload = f.read(dim * elem_size)
            if len(payload) != dim * elem_size:
                raise EOFError(f"{src}: truncated row {row}")
            if kind == "f32" or kind == "u32":
                out.write(payload)
            else:
                raise ValueError(f"unsupported kind: {kind}")

        extra = f.read(1)
        if extra:
            print(f"Warning: {src} has extra bytes after {rows} rows")


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        print(f"Found {dst}")
        return
    try:
        os.symlink(src, dst)
        print(f"Symlinked {dst} -> {src}")
    except OSError:
        shutil.copyfile(src, dst)
        print(f"Copied {src} -> {dst}")


def extract_npy_from_npz(
    src: Path,
    member: str,
    dst: Path,
    expected_descr: str,
    expected_shape: tuple[int, ...],
) -> None:
    if dst.exists():
        print(f"Found {dst}")
        return

    print(f"Extracting {member} from {src} -> {dst}")
    with zipfile.ZipFile(src) as zf:
        with zf.open(member) as f:
            magic = f.read(6)
            if magic != b"\x93NUMPY":
                raise ValueError(f"{member}: not a .npy payload")
            major = f.read(1)[0]
            minor = f.read(1)[0]
            if major == 1:
                header_len = struct.unpack("<H", f.read(2))[0]
            elif major in (2, 3):
                header_len = struct.unpack("<I", f.read(4))[0]
            else:
                raise ValueError(f"{member}: unsupported npy version {major}.{minor}")
            header = ast.literal_eval(f.read(header_len).decode("latin1"))
            if header["descr"] != expected_descr:
                raise ValueError(
                    f"{member}: dtype {header['descr']}, expected {expected_descr}"
                )
            if header["fortran_order"]:
                raise ValueError(f"{member}: fortran-order arrays are unsupported")
            if tuple(header["shape"]) != expected_shape:
                raise ValueError(
                    f"{member}: shape {header['shape']}, expected {expected_shape}"
                )
            dst.write_bytes(f.read())


def prepare_local_sift100k(outdir: Path) -> None:
    root = repo_root()
    vector_repo = root.parent / "omendb-vector"
    base_src = vector_repo / "benchmarks" / "data" / "sift-100k.f32bin"
    npz_src = vector_repo / "benchmarks" / "data" / "sift-100k.npz"

    for src in [base_src, npz_src]:
        if not src.exists():
            raise FileNotFoundError(src)

    base_dst = outdir / "sift100k_base.f32bin"
    query_dst = outdir / "sift100k_query.f32bin"
    gt_dst = outdir / "sift100k_groundtruth.u32bin"
    manifest = outdir / "sift100k_manifest.txt"

    link_or_copy(base_src, base_dst)
    extract_npy_from_npz(npz_src, "queries.npy", query_dst, "<f4", (10000, 128))
    extract_npy_from_npz(npz_src, "ground_truth.npy", gt_dst, "<i4", (10000, 10))
    manifest.write_text(
        "\n".join(
            [
                "dataset=sift100k",
                "base_vectors=100000",
                "query_vectors=10000",
                "dim=128",
                "groundtruth_neighbors=10",
                f"base_source={base_src}",
                f"npz_source={npz_src}",
            ]
        )
        + "\n"
    )
    print(f"Wrote {manifest}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["siftsmall", "sift100k", "sift1m"])
    parser.add_argument("--outdir", default="data")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "sift100k":
        prepare_local_sift100k(outdir)
        print("Done.")
        return

    import h5py  # ty: ignore[unresolved-import]
    import numpy as np  # ty: ignore[unresolved-import]

    info = DATASETS[args.dataset]
    hdf5_path = outdir / info["filename"]

    if not hdf5_path.exists():
        print(f"Downloading {info['url']} to {hdf5_path}...")
        urllib.request.urlretrieve(info["url"], hdf5_path)
    else:
        print(f"Found {hdf5_path}")

    base_path = outdir / f"{args.dataset}_base.f32bin"
    query_path = outdir / f"{args.dataset}_query.f32bin"
    groundtruth_path = outdir / f"{args.dataset}_groundtruth.u32bin"

    print("Extracting to raw binaries...")
    with h5py.File(hdf5_path, "r") as f:
        # Train (base vectors)
        train = np.array(f["train"])
        print(f"Base vectors: {train.shape}, {train.dtype}")
        train.astype(np.float32).tofile(base_path)

        # Test (query vectors)
        test = np.array(f["test"])
        print(f"Query vectors: {test.shape}, {test.dtype}")
        test.astype(np.float32).tofile(query_path)

        # Neighbors (groundtruth)
        neighbors = np.array(f["neighbors"])
        print(f"Groundtruth: {neighbors.shape}, {neighbors.dtype}")
        neighbors.astype(np.uint32).tofile(groundtruth_path)

    print("Done.")


if __name__ == "__main__":
    main()
