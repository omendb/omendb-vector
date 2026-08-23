# OmenDB Vector

> **Developer preview.** OmenDB Vector is a work-in-progress embedded local
> database for hybrid vector and text retrieval. APIs, persistence formats,
> supported platforms, and performance are subject to change.

The repository contains the Mojo engine and its Python API. It is source-only
for now; package publication and release compatibility are not yet supported.

## Build and test

Requires [Pixi](https://pixi.sh):

```bash
git clone https://github.com/omendb/omendb-vector.git
cd omendb-vector
pixi install
pixi run python -m pip install -e .
```

Build the native extension and run the checks:

```bash
pixi run mojo build src/python_engine.mojo --emit shared-lib \
  -o src/omendb_vector/_omendb_vector_engine.so
pixi run ruff check --ignore E501 src/omendb_vector scripts examples tests/test_python_api.py
pixi run python -m pytest tests -q
```

## Python example

```python
import omendb_vector

db = omendb_vector.create("./my-db")
docs = db.collection(
    "docs", config=omendb_vector.CollectionConfig(dim=128, text=True)
)
docs.set(
    "example",
    vector=[0.0] * 128,
    text="hybrid vector and text retrieval",
    metadata={"kind": "guide"},
)
print(docs.search_hybrid(vector=[0.0] * 128, text="retrieval", k=1))
```

The separate [`omendb-rs`](https://github.com/omendb/omendb-rs) repository
contains the earlier Rust vector implementation and its existing package
history. This repository does not publish crates or Python packages yet.

## License

AGPL-3.0-only.
