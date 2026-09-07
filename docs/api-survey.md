# API Survey — Conventions Behind the v2 Design

Status: evidence register for `docs/api-v2-draft.md`. Grades every convention claim
by evidence level. Survey date: October 2026.

Evidence grades:
- **[P]** primary source read this session (official docs or spec, current)
- **[S]** stable knowledge (APIs unchanged for years, low churn risk, not re-verified)
- **[R]** repo-internal evidence (our engine's code and tests)
- **[X]** secondary/quoted (not re-verified; never load-bearing alone)

Audience definition (ADR-0002, mem constraint): developers with sqlite/PostgreSQL/
DuckDB experience. Mongo conventions explicitly rejected. Uniform-but-idiomatic
across Rust + Python (+future bindings) is a product requirement.

---

## 1. Constructor

| System | Verb | Shape |
|---|---|---|
| Python sqlite3 | `connect` | `sqlite3.connect(path, timeout, isolation_level, ...)` **[P]** |
| Python duckdb | `connect` | `duckdb.connect(database, read_only, config)` **[S]** |
| rusqlite | `open` | `Connection::open(path)` **[S]** |
| Python lancedb | `connect` | `lancedb.connect(uri_or_path)` — accepts local path *or* remote URI **[P]** |
| libsql | `connect` | path or remote URL **[S]** |
| Qdrant | `QdrantClient(url=...)` | endpoint, always server **[S]** |

Decisions:
- **`connect`** over `open`. Decisive: architecture §1 runs the same engine local
  or server; `lancedb`/`libsql` are existence proofs that one endpoint-neutral
  verb absorbs the server mode without breaking [P/S]. Python-side winner is
  unanimous (sqlite3, duckdb, lancedb, libsql); Rust adopts `connect` for
  cross-binding uniformity over rusqlite's `open` precedent.
- Module-level function in Python (`omendb.connect(...)`), matching
  sqlite3/duckdb/lancedb shape. Rust: `Store::connect(dir, options)`.

## 2. Write verb (single, batch-shaped)

Our semantics: keyed last-wins, i.e. `INSERT OR REPLACE`. Decided verb: **`add`**,
with a loud contract line ("add replaces any existing record with the same id —
INSERT OR REPLACE, not append"). Evolution: `conflict="error"` knob if demand
earns it; `add` stays neutral where `upsert` would self-contradict under such a
knob.

| System | Verb(s) | Duplicate-id behavior |
|---|---|---|
| Chroma | `add`, plus `upsert` added later | `add` = upsert-like replace on same id **[X]** (Chroma docs, quoted; the confusion pattern is real but was sibling-ambiguity across verbs, not `add` alone) |
| LanceDB | `add`, `merge_insert` | `add` = append-only, duplicates survive **[X]**; `merge_insert` is the keyed upsert |
| Weaviate | `add` (client), `replace` REST | `add` on keyed objects **[S]** |
| Qdrant | `upsert` | replace by point id **[S]** |
| sqlite | `INSERT` / `INSERT OR REPLACE` | explicit; PG adds `ON CONFLICT` **[P]** |
| DuckDB | `INSERT` (append) / `INSERT OR REPLACE` | explicit **[S]** |

Analysis recorded for the decision (not load-bearing on any [X] claim):
- `upsert` is the semantics-exact name for relational audiences, and it is the
  cross-product standard for keyed-replace among server products (Qdrant). It
  was rejected because (a) the historical `add` pain in Chroma/LanceDB was
  sibling-ambiguity — with a single write verb, the failure mode cannot
  replicate; (b) a name that hardcodes replace-policy blocks the
  `conflict="error"` evolution; (c) embedded-peer warmth for our actual
  audience.
- `set` was evaluated in full before rejection. Its real strengths: exact
  keyed-replace meaning to every programmer (zero contract-line dependency);
  `set`/`get` symmetry; best-in-class extensibility precedent (Redis `SET NX/XX`
  — policy knobs on a verb named set). Rejected on misread asymmetry for the
  declared audience: SQL reads `SET` as UPDATE field-assignment (patch), so
  `set(id, vector, metadata={"c": 1})` invites a partial-update expectation,
  and whole-record replace then silently drops unpassed fields — a destructive
  misread (data the caller meant to keep exists only in the store). `add`'s
  available misread (LanceDB append semantics: expect two records under one
  id, find one) is non-destructive — nothing written is lost, and duplicate-id
  versions in a keyed store are an exotic want. Tie otherwise, so take the
  name whose worst misreading is survivable. Docs must add under either verb:
  "add writes whole records; omitted fields are not preserved."
- `insert` rejected: implies conflict-free semantics to this audience.

## 3. Batch shape

Columnar, one verb, n≥1. Single = batch of 1 (two brackets, one unambiguous shape).

| System | Batch shape |
|---|---|
| Chroma | `add(ids, embeddings, metadatas, documents)` — columnar kwargs **[S]** |
| LanceDB | `table.add(dict_of_lists)` — columnar dict **[S]** |
| Qdrant | `upsert(points=[...])` — row-shaped struct list **[S]** |
| DuckDB appender | columnar bulk load **[S]** |
| sqlite3 | `executemany` row-shaped; batchability via prepared stmt **[P]** |

Decision: **columnar** (Chroma/LanceDB shape). Reasons: matches engine record
layout, avoids per-row struct construction in Python, aligns with the
columnar segment format (records encode columnar; row-shaped input would
imply a transpose for no caller benefit).

## 4. Filters

| System | Filter surface |
|---|---|
| turbopuffer | SQL-ish `filters: "rating >= 3 AND tags IN ('a','b')"` **[X]** (turbopuffer docs, quoted) |
| LanceDB | SQL-subset string `where="..."` (comparisons, IN, boolean ops, IS NULL) **[S]** |
| Qdrant | nested JSON `must`/`should`/`must_not` — boolean clause groups **[S]** |
| Chroma | `where={"field": {"$eq": v}}` — `$`-operator dicts **[S]** |
| sqlite/PG/DuckDB | SQL WHERE, full boolean algebra **[P]** |

Decisions:
- **Equality dict** for the 90% case: `where={"lang": "en"}`. AND of equality
  predicates; cannot express OR by construction — intentional (dicts of
  conditions are inherently conjunctive).
- **SQL-subset string** for expressiveness: comparisons, IN, IS NULL/IS NOT
  NULL, AND/OR, parentheses. Compiles to the existing `Predicate` AST extended
  with a boolean layer (`Or`/`Not` nodes). Parser is ~150 lines, no
  dependencies.
- `$`-operators rejected: dead Mongo convention, explicitly out per audience
  definition (mem constraint).
- OR is included despite engine AND-only routing today: per-record exact
  evaluation is the oracle semantics and OR costs nothing there (predicate
  `any()`); the deferred cost is disjunction-aware *planning*, a performance
  tier (route to exact-fallback more often under OR-heavy filters). Slower
  sometimes, never wrong. SQL-literate audiences expect OR in a "SQL subset";
  shipping it while the format/AST window is open (pre-1.0) is cheaper than
  adding it post-1.0.

## 5. Durability surface

| System | Knob |
|---|---|
| sqlite | `PRAGMA synchronous = FULL / NORMAL / OFF` — NORMAL in WAL mode = fsync only at checkpoints, process-crash-safe with power-loss window **[P]** |
| DuckDB | no user-facing durability ladder (checkpoint config only) **[S]** |
| Qdrant | `wait=true` per write **[S]** |

Decision: `safety="strict" | "normal"` at connect. strict (default) = fsync
per commit ("never lose acked writes" = product spine); normal = fsync at
checkpoint, process-crash-safe, power-crash window — identical meaning to
sqlite `synchronous=NORMAL` in WAL mode, which the audience already knows
[P][R: our WAL/commit machinery matches this shape]. No third "off" tier
(disallowed by fail-closed durability posture).

## 6. Transactions

| System | Model |
|---|---|
| sqlite/PG | autocommit default, explicit `BEGIN`/`COMMIT`/`ROLLBACK` **[P]** |
| rusqlite | autocommit default; explicit transactions via guard **[S]** |
| DuckDB | autocommit; explicit BEGIN/COMMIT **[S]** |
| Chroma/LanceDB/Qdrant | no real transactions **[S]** |

Decision: autocommit default; explicit transactions via a `transaction()`
context-manager guard (Python) / RAII guard (Rust) with rollback on drop
(truncate-to-last-commit: drop L0 entries above barrier + truncate WAL to
barrier offset — machinery exists in recovery paths [R]). rusqlite precedent
adopted on the Rust side too; engine-level `commit()` remains the
mechanism-level escape hatch for tests/bench.

## 7. Metric

| System | Policy |
|---|---|
| Qdrant | metric fixed at collection create **[S]** |
| LanceDB | metric per column, fixed at create **[S]** |
| Chroma | metric fixed at collection create **[S]** |

Decision: metric fixed at connect (collection state), one of
`"cosine" | "l2" | "dot"`. Unanimous category policy [S]; HNSW is
single-metric by construction [R: sprint-2 dot-graph rejection], per-call
metric is a footgun.

## 8. IDs

| System | ID surface |
|---|---|
| sqlite | INTEGER PRIMARY KEY rowid + arbitrary text PKs **[P]** |
| Chroma/LanceDB/Qdrant/Weaviate | int or string ids first-class **[S]** |

Decision: `ExternalId::Int(u64) | Str(String)` first-class in the record
format (format v2 bump — no-compat-shims rule for 0.x, window open now,
closes at 1.0). Id-kind locked at first write, same rule as dim-locking [R].
Hash-mapping strings to u64 rejected: silent collision risk violates the
durability-honesty posture. Mixed kinds per record rejected: two id spaces
complicate segments/WAL/bindings without demand.

## 9. Maintenance / lifecycle surface

| System | Maintenance verb |
|---|---|
| sqlite | `VACUUM` / `PRAGMA optimize` — public but rarely called programmatically **[S]** |
| LanceDB | `table.optimize()` — public compaction **[S]** |
| turbopuffer | invisible lifecycle (async indexing, no user checkpoint verb) **[X]** |

Decision: **no public `optimize()`**. Checkpoint cadence is engine policy;
invisible lifecycle is the turbopuffer-validated product shape. Rust
`checkpoint()` stays as the mechanism-level verb for tests/bench [R]. The
object-storage regime comparison (register §11) is the strategic model: WAL +
async-index + exact-tail lifecycle, invisible to callers.

## 10. Record payload on hits

| System | Hit → payload |
|---|---|
| Chroma | `include=[...]` opt-in flag **[S]** |
| LanceDB | separate `.select()` / `.to_list()` stages **[S]** |
| Qdrant | `with_payload`, `with_vector` flags **[S]** |

Decision: **no include flags** — lazy `.record` accessor on `Hit` (property
fetch on demand). One fewer knob; no flag-boolean sprawl; the fetch is a
single keyed lookup [R: `get` machinery]. Rejected `include=` on
flag-sprawl grounds and because lazy access makes the common case
(score + id + lazy payload) zero-config.

## 11. Search surface

| System | Query surface |
|---|---|
| turbopuffer | `search(vector=..., distance=..., filters=..., top_k=...)` — one verb, modal inputs **[X]** |
| Chroma | `query(query_embeddings, where, include)` **[S]** |
| LanceDB | `search(vector).where(...).limit(k)` — builder chain **[S]** |
| Qdrant | `search` / `query` points API with prefetch-rescore **[S]** |

Decision: **one `search` verb, modal inputs** — `search(query_vector=...,
query_text=..., k=..., where=...)`; both query inputs optional but at least
one required; both present = hybrid RRF [R: sprint-3/4 RRF exists, weakest-
link guard specified]. Vector-only = dense HNSW/exact path; text-only = BM25
path. No builder chain (LanceDB's builder is a bigger surface than six names
permits; scalar kwargs beat method-chaining for a minimal surface). Prefix-dim
queries allowed (Matryoshka) [R]. `where=` is shared with delete filters
where engine-supported.

## 12. Read/pattern summary

The decided surface is six names — `connect`, `add`, `delete`, `get`,
`search`, `transaction` — two data verbs (`add`, `search`), uniform across
Rust and Python. Smaller than the sqlite C API, smaller than any vector peer
surveyed. Every deviation from peer convention is a deliberate, documented
one: `add` (single-verb discipline + intent fit over `upsert`'s exactness),
`connect` (endpoint neutrality), no `optimize` (invisible lifecycle), no
include flags (lazy access), flat dict + string filter duality (SQL literacy
without `$`-operators).

Peers rely on flag matrices (Chroma include, Qdrant with_payload) and verb
pairs (add/upsert, insert/merge_insert) to cover the same space; we compress
to modal inputs + lazy accessors. The bet: discoverability comes from docs,
not surface area, and durability defaults (strict) do the trust-building
that peers spend verbs on.
