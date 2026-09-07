# API v2 Draft — The Dev-Facing Surface

Status: draft for Nick's approval. Decides the v2 surface across Rust + Python
(+ future bindings). Companion evidence: `docs/api-survey.md`. Nothing here is
implemented; sprint 8 starts after approval.

Design constraints (settled, from handoff + this session):
- Audience: sqlite/PostgreSQL/DuckDB-experienced devs. No Mongo conventions.
- Uniform-but-idiomatic across languages; one name per concept everywhere.
- Minimal verb set; batch-shaped everywhere; single = batch of 1.
- API is the product: no algorithm, quantization, or storage tier leaks.
- Rust surface may change (ADR-0003 applies to our own API too).

## 1. Decided surface

Six names, two data verbs:

```python
import omendb

db = omendb.connect("./mydata", metric="cosine", safety="strict")

db.add(ids, vectors, text=texts, metadata=metadatas)   # the one write verb
db.delete(ids)                                          # batch-shaped always
db.get(ids)                                             # batch-in, batch-out
hits = db.search(query_vector=v, k=10, where="year >= 2024 AND lang != 'xx'")
for hit in hits:
    print(hit.id, hit.score, hit.record)                # .record is lazy
```

Rust mirror: `Store::connect`, `add`, `delete`, `get`, `search`, `transaction`.

## 2. Decisions, each with rationale

### 2.1 `connect(path, *, metric, safety, dim=None)` — not `open`

Endpoint-neutral from day one: architecture §1 runs one engine core local or
server, and `lancedb.connect` / `libsql` prove the verb absorbs the server
mode without breaking. Python-side unanimity (sqlite3, duckdb, lancedb,
libsql). Module-level function in Python; `Store::connect` in Rust.

- `metric="cosine" | "l2" | "dot"` — fixed at connect (unanimous category
  policy; HNSW is single-metric by construction). Per-call metric is a footgun.
- `safety="strict" | "normal"` — strict (default): fsync per commit, never
  lose an acked write. normal: fsync at checkpoint, process-crash-safe with a
  power-crash window — identical meaning to sqlite `PRAGMA
  synchronous=NORMAL` in WAL mode, which the audience already knows. No third
  "off" tier (fail-closed posture).
- `dim` optional at connect; first write locks it if unspecified (existing
  dim-lock rule unchanged).

### 2.2 `add(ids, vectors, *, text=None, metadata=None)` — the one write verb

Columnar, n≥1, single = batch of 1 (two brackets, one unambiguous shape).
Keyed last-wins — `INSERT OR REPLACE`, not append. Contract lines required in
every binding's docstring:
1. "add replaces any existing record with the same id — INSERT OR REPLACE,
   not append."
2. "add writes whole records; omitted fields are not preserved."

`add` was chosen over `upsert` (single-verb discipline removes the sibling
ambiguity that burned Chroma/LanceDB; name stays neutral under a future
`conflict="error"` knob) and over `set` (SQL audience reads SET as UPDATE
patching — a destructive misread under whole-record replace; `add`'s worst
misread, append assumption, is non-destructive). Full evaluation in survey §2.

### 2.3 `delete(ids)` — batch-shaped always

Ids in, count out. Accepts the same id kinds `add` accepts. Filters on delete
(where=) deferred until engine support exists — not in v2's surface.

### 2.4 `get(ids) -> list[Record]`

Batch-in, batch-out, order-preserving, None-missing. Lazy `.record` on `Hit`
uses the same fetch machinery — `get` is the accessor and the backing store
for lazy hit payloads.

### 2.5 `search(query_vector=None, query_text=None, k=10, where=None, ...)`

One verb, modal inputs; both query forms optional, at least one required:

- vector-only → dense path (HNSW/exact; tier invisible)
- text-only → BM25 path
- both → hybrid RRF (weakest-link guard specified in architecture §6)

`where` accepts either shape:
- **Equality dict**: `where={"lang": "en"}` — AND of equalities, the 90% case.
  Cannot express OR by construction (conjunctive dict shape).
- **SQL-subset string**: `where="year >= 2024 OR lang != 'xx'"` — comparisons,
  IN, IS NULL/IS NOT NULL, AND/OR/NOT, parentheses. ~150-line dependency-free
  parser compiling to the existing `Predicate` AST extended with a boolean
  layer. OR ships now: exact per-record evaluation is the oracle semantics
  (OR costs `any()` there); disjunction-aware *planning* is the later
  performance tier — OR-heavy filters route to exact-fallback more often.
  Slower sometimes, never wrong.

Prefix-dim vector queries stay first-class (Matryoshka). `ef_search` and
`per_segment_k` leave the public API — they are algorithm knobs, not product
surface; the recall floor is engine policy (bench-gated).

`Hit`: `id`, `score`, lazy `.record`. No `include=` flags (Chroma include /
Qdrant with_payload flag matrices rejected; lazy access is zero-config for the
common case: score + id).

### 2.6 `transaction()` — guard with rollback

Autocommit default (sqlite/rusqlite precedent, now adopted Rust-side too).
Explicit transactions: context manager (Python) / RAII guard (Rust) with
rollback on drop — truncate-to-last-commit (drop L0 entries above barrier +
truncate WAL to barrier offset; the recovery machinery already does exactly
this, so in-process rollback reuses it). Engine-level `commit()` stays as
the mechanism-level escape hatch for tests/bench.

### 2.7 Ids: `ExternalId::Int(u64) | Str(String)` — first-class, kind-locked

Format v2 bump (window open now, closes at 1.0 — no-compat-shims rule for
0.x). Id-kind locked at first write, same rule as dim-locking. Hash-mapping
rejected (silent collisions). Mixed kinds per record rejected (two id spaces,
no demand). Python: int or str ids, kind inferred, locked after first write.

### 2.8 No `optimize()` in the public API

Checkpoint cadence is engine policy, invisible to the caller (turbopuffer's
invisible WAL + async-index + exact-tail lifecycle is the validated product
shape for the direction we're going). Rust `checkpoint()` stays as the
mechanism-level verb for tests/bench.

### 2.9 No collection layer

One directory = one collection (sqlite model). Server mode later introduces
namespaces turbopuffer-style; the embedded API neither needs nor gets a
layer for it.

## 3. What v2 deliberately does NOT have

- No `upsert`/`add_many`/scalar-dispatch verb zoo (one batch verb).
- No `$`-operator filter dicts (dead Mongo convention, audience-wrong).
- No include/with_payload flag matrices (lazy access).
- No builder chains (scalar kwargs beat method-chaining at this surface size).
- No per-call metric, no ef_search/per_segment_k in the public API.
- No delete-by-filter, no upsert-return-values, no bulk_get variants —
  each earns its place via demand (ADR-0003 applies to API surface too).

## 4. Binding notes (per language)

- **Rust**: `Store::connect/add/delete/get/search`, `Transaction` guard,
  `ExternalId`, `Predicate` AST + string parser. Everything is
  engine-native; no binding translation layer.
- **Python** (`omendb_vector`): thin PyO3 over the same names. Module-level
  `connect`; keyword-idiomatic signatures; `Hit.record` as a lazy property;
  errors surface as one `OmenDBError` hierarchy. PyO3 0.29 gotchas already in
  mem (spring 7).
- **Future bindings** (node, C ABI): the same six names, the same contract
  lines. The surface is small enough to spec once and port.

## 5. Open (not blocking approval)

- Exact `search()` kwarg names in Python (`query_vector` vs `vector` vs
  `q_vector`) — decide at binding implementation, docstring-driven.
- Hit score normalization across metric kinds (raw distance vs similarity:
  higher-better everywhere, or metric-faithful?) — decide with binding tests;
  oracle-equality tests constrain it.
- Whether `transaction()` returns a token usable as a with-target in the
  Rust API or only the guard type — Rust-side idiom decision at impl time.

## 6. Sprint 8 sequence (after approval, not before)

1. Record format v2: id kind + metric persisted (kind-locked/dim-locked
   state in the store header).
2. Filter string parser: SQL subset → boolean `Predicate` AST; oracle
   equality tests against per-record exact evaluation (the evaluator IS the
   oracle).
3. Python binding v2: the six-name surface, both `where` shapes,
   transactions, lazy `Hit.record`.
4. Rust API backport where it wins: `connect` naming, transaction guard,
   `ExternalId` in public signatures.
5. Dogfood: migrate tests + bench harness callers onto the v2 surface.
6. turbopuffer server-reference bench lane (deferred per Nick; separate
   task).

Do not publish bindings anywhere (no crates.io/PyPI) without explicit OK.
