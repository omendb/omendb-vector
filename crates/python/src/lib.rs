//! OmenDB Vector Python bindings (PyO3).
//!
//! `Store` wraps the engine's single-writer store; see
//! docs/architecture.md for the semantics (commit = fsync barrier,
//! checkpoint = seal+publish, WAL-first durability).
//!
//! Metric mapping: `"dot" | "cosine" | "l2"`. Backend config:
//! `"exact"` (default) or `"hnsw"`. Metadata values map int/float/
//! bool/str/bytes; filters are dicts with eq/in/range/present specs.

use omendb_vector_engine::filter::{Filter, Num, Predicate};
use omendb_vector_engine::index::{HnswConfig, Metric};
use omendb_vector_engine::records::{MetaValue, Record};
use omendb_vector_engine::store::{IndexBackend, Store};
use omendb_vector_engine::EngineError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::create_exception;

create_exception!(omendb_vector, VectorEngineError, PyValueError);

fn engine_err(e: EngineError) -> PyErr {
    match e {
        EngineError::Schema(msg) => PyValueError::new_err(msg),
        other => VectorEngineError::new_err(other.to_string()),
    }
}

type EnginePyResult<T> = Result<T, PyErr>;

fn parse_metric(s: &str) -> PyResult<Metric> {
    match s {
        "dot" => Ok(Metric::Dot),
        "cosine" => Ok(Metric::Cosine),
        "l2" => Ok(Metric::L2),
        other => Err(PyValueError::new_err(format!(
            "unknown metric {other:?} (use dot/cosine/l2)"
        ))),
    }
}

fn parse_backend(s: &str, m: usize, m0: usize, ef_construction: usize, ef_search: usize) -> PyResult<IndexBackend> {
    match s {
        "exact" => Ok(IndexBackend::Exact),
        "hnsw" => Ok(IndexBackend::Hnsw(HnswConfig {
            m,
            m0,
            ef_construction,
            metric: Metric::L2, // overwritten by metric arg at open? No: graph metric fixed at config; see open_with docs
            seed: 0x5EED_5EED,
            ef_search_default: ef_search,
        })),
        other => Err(PyValueError::new_err(format!(
            "unknown backend {other:?} (use exact/hnsw)"
        ))),
    }
}

/// Convert a Python value to a typed metadata value.
fn to_meta_value(v: Bound<'_, PyAny>) -> PyResult<MetaValue> {
    if let Ok(i) = v.extract::<i64>() {
        Ok(MetaValue::Int(i))
    } else if let Ok(f) = v.extract::<f64>() {
        Ok(MetaValue::Float(f))
    } else if let Ok(b) = v.extract::<bool>() {
        Ok(MetaValue::Bool(b))
    } else if let Ok(s) = v.extract::<String>() {
        Ok(MetaValue::Str(s))
    } else if let Ok(b) = v.extract::<Vec<u8>>() {
        Ok(MetaValue::Bytes(b))
    } else {
        Err(PyValueError::new_err(format!(
            "metadata value must be int/float/bool/str/bytes, got {}",
            v.get_type().name()?
        )))
    }
}

fn from_meta_value(v: &MetaValue) -> PyResult<Py<PyAny>> {
    Python::attach(|py| {
        let any: Bound<'_, PyAny> = match v {
            MetaValue::Int(i) => i.into_pyobject(py)?.into_any(),
            MetaValue::Float(f) => f.into_pyobject(py)?.into_any(),
            MetaValue::Bool(b) => {
                // PyBool::new borrows; to_owned() gives a Bound we
                // can move into Py<PyAny>.
                pyo3::types::PyBool::new(py, *b)
                    .to_owned()
                    .into_any()
            }
            MetaValue::Str(s) => s.into_pyobject(py)?.into_any(),
            MetaValue::Bytes(b) => b.into_pyobject(py)?.into_any(),
        };
        Ok(any.unbind())
    })
}

/// Build a `Filter` from a list of predicate dicts:
/// `{"eq": {"field": ..., "value": ...}}`,
/// `{"in": {"field": ..., "values": [...]}}`,
/// `{"range": {"field": ..., "lo": int|float, "hi": int|float}}`,
/// `{"present": "field"}`.
fn build_filter(specs: Vec<Bound<'_, PyAny>>) -> PyResult<Filter> {
    let mut filter = Filter::new();
    for spec in specs {
        let dict = spec.cast::<pyo3::types::PyDict>()?;
        if let Some(eq) = dict.get_item("eq")? {
            let d = eq.cast::<pyo3::types::PyDict>()?;
            let field: String = d.get_item("field")?.unwrap().extract()?;
            let value = to_meta_value(d.get_item("value")?.unwrap())?;
            filter = filter.and(Predicate::Eq { field, value });
        } else if let Some(ins) = dict.get_item("in")? {
            let d = ins.cast::<pyo3::types::PyDict>()?;
            let field: String = d.get_item("field")?.unwrap().extract()?;
            let values = d
                .get_item("values")?
                .unwrap()
                .try_iter()?
                .map(|v| to_meta_value(v.map_err(PyErr::from)?))
                .collect::<PyResult<Vec<_>>>()?;
            filter = filter.and(Predicate::In { field, values });
        } else if let Some(r) = dict.get_item("range")? {
            let d = r.cast::<pyo3::types::PyDict>()?;
            let field: String = d.get_item("field")?.unwrap().extract()?;
            let lo = parse_num(&d.get_item("lo")?.unwrap())?;
            let hi = parse_num(&d.get_item("hi")?.unwrap())?;
            filter = filter.and(Predicate::Range { field, lo, hi });
        } else if let Some(p) = dict.get_item("present")? {
            let field: String = p.extract()?;
            filter = filter.and(Predicate::Present { field });
        } else {
            return Err(PyValueError::new_err(
                "filter spec must contain one of: eq, in, range, present",
            ));
        }
    }
    Ok(filter)
}

fn parse_num(v: &Bound<'_, PyAny>) -> PyResult<Num> {
    if let Ok(i) = v.extract::<i64>() {
        Ok(Num::Int(i))
    } else if let Ok(f) = v.extract::<f64>() {
        Ok(Num::Float(f))
    } else {
        Err(PyValueError::new_err("range bound must be int or float"))
    }
}

/// One search hit: (external_id, score, seq).
#[pyclass(get_all)]
#[derive(Clone)]
struct PyHit {
    external_id: u64,
    score: f32,
    seq: u64,
}

#[pymethods]
impl PyHit {
    fn __repr__(&self) -> String {
        format!("Hit(id={}, score={:.4}, seq={})", self.external_id, self.score, self.seq)
    }
}

/// What `Store.open` recovered.
#[pyclass(get_all)]
struct PyRecovery {
    committed_seq: u64,
    rebuilt_from_wal: bool,
    generation: u64,
}

/// The OmenDB Vector store (single writer, WAL-first durability).
#[pyclass]
struct PyStore {
    inner: Store,
}

#[pymethods]
impl PyStore {
    /// Open (or create) a store. Returns (store, recovery).
    #[staticmethod]
    #[pyo3(signature = (path, backend = "exact", metric = "l2", hnsw_m = 16, hnsw_m0 = 32, hnsw_ef_construction = 192, hnsw_ef_search = 200))]
    fn open(
        path: String,
        backend: &str,
        metric: &str,
        hnsw_m: usize,
        hnsw_m0: usize,
        hnsw_ef_construction: usize,
        hnsw_ef_search: usize,
    ) -> PyResult<(Self, PyRecovery)> {
        let m = parse_metric(metric)?;
        let b = parse_backend(backend, hnsw_m, hnsw_m0, hnsw_ef_construction, hnsw_ef_search)?;
        // HNSW graphs are single-metric: align the config metric.
        let b = match b {
            IndexBackend::Hnsw(mut cfg) => {
                cfg.metric = m;
                IndexBackend::Hnsw(cfg)
            }
            other => other,
        };
        let (store, rec) = Store::open_with(&path, b).map_err(engine_err)?;
        Ok((
            PyStore { inner: store },
            PyRecovery {
                committed_seq: rec.wal.committed_seq,
                rebuilt_from_wal: rec.rebuilt_from_wal,
                generation: rec.manifest_generation.unwrap_or(0),
            },
        ))
    }

    /// Upsert a record. Returns the assigned WAL seq. Not durable
    /// until `commit()`.
    #[pyo3(signature = (id, vector, text = None, meta = None))]
    fn upsert(
        &mut self,
        id: u64,
        vector: Vec<f32>,
        text: Option<String>,
        meta: Option<Bound<'_, PyAny>>,
    ) -> PyResult<u64> {
        let mut r = Record::new(id, vector);
        if let Some(t) = text {
            r = r.with_text(t);
        }
        if let Some(m) = meta {
            let dict = m.cast::<pyo3::types::PyDict>()?;
            for (k, v) in dict.iter() {
                let key: String = k.extract()?;
                let value = to_meta_value(v)?;
                r = r.with_meta(key, value);
            }
        }
        self.inner.upsert(r).map_err(engine_err)
    }

    /// Delete (tombstone) an id. Unknown/dead ids raise ValueError.
    fn delete(&mut self, id: u64) -> PyResult<u64> {
        self.inner.delete(id).map_err(engine_err)
    }

    /// Commit barrier: fsync. Everything appended is now durable.
    fn commit(&mut self) -> PyResult<u64> {
        self.inner.commit().map_err(engine_err)
    }

    /// Checkpoint: seal live set into a segment, publish manifest,
    /// reset the RAM tier.
    fn checkpoint(&mut self) -> PyResult<u64> {
        self.inner.checkpoint().map_err(engine_err)
    }

    /// Collection dim (0 until the first record defines it).
    #[getter]
    fn dim(&self) -> u32 {
        self.inner.dim()
    }

    /// Number of live records.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Fetch a live record as a dict, or None.
    fn get(&self, id: u64) -> PyResult<Option<Py<PyAny>>> {
        let Some(r) = self.inner.get(id) else {
            return Ok(None);
        };
        let obj = Python::attach(|py| -> PyResult<Py<PyAny>> {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("external_id", r.external_id)?;
            dict.set_item("vector", r.vector.clone())?;
            dict.set_item("text", r.text.clone())?;
            dict.set_item("norm", r.norm)?;
            let meta = pyo3::types::PyDict::new(py);
            for (k, v) in &r.meta {
                meta.set_item(k, from_meta_value(v)?)?;
            }
            dict.set_item("meta", meta)?;
            Ok(dict.into_any().unbind())
        })?;
        Ok(Some(obj))
    }

    /// Exact flat scan (the correctness oracle).
    #[pyo3(signature = (metric, query, k))]
    fn exact_search(&self, metric: &str, query: Vec<f32>, k: usize) -> PyResult<Vec<PyHit>> {
        let m = parse_metric(metric)?;
        let hits = self.inner.exact_top_k(m, &query, k).map_err(engine_err)?;
        Ok(hits
            .into_iter()
            .map(|h| PyHit {
                external_id: h.external_id,
                score: h.score,
                seq: h.seq,
            })
            .collect())
    }

    /// Backend search over segments + L0 (HNSW when configured).
    #[pyo3(signature = (metric, query, k, window, filters = None))]
    fn search(
        &self,
        metric: &str,
        query: Vec<f32>,
        k: usize,
        window: usize,
        filters: Option<Vec<Bound<'_, PyAny>>>,
    ) -> PyResult<Vec<PyHit>> {
        let m = parse_metric(metric)?;
        let hits = match filters {
            Some(specs) => {
                let f = build_filter(specs)?;
                self.inner.filtered_exact_top_k(m, &query, k, &f).map_err(engine_err)?
            }
            None => self.inner.exact_top_k(m, &query, k).map_err(engine_err)?,
        };
        Ok(hits
            .into_iter()
            .map(|h| PyHit {
                external_id: h.external_id,
                score: h.score,
                seq: h.seq,
            })
            .collect())
    }

    /// BM25 text search.
    fn text_search(&self, query: &str, k: usize) -> Vec<PyHit> {
        self.inner
            .text_search(query, k)
            .into_iter()
            .map(|h| PyHit {
                external_id: h.external_id,
                score: h.score,
                seq: h.seq,
            })
            .collect()
    }

    /// Hybrid RRF search: vector and/or text, both optional but at
    /// least one required.
    #[pyo3(signature = (k, window, vector_query = None, vector_metric = "l2", text_query = None))]
    fn hybrid_search(
        &self,
        k: usize,
        window: usize,
        vector_query: Option<Vec<f32>>,
        vector_metric: &str,
        text_query: Option<String>,
    ) -> PyResult<Vec<PyHit>> {
        let metric = parse_metric(vector_metric)?;
        let m = vector_query.as_deref().map(|q| (q, metric));
        let hits = self.inner.hybrid_search_rrf(m, text_query.as_deref(), k, window).map_err(engine_err)?;
        Ok(hits
            .into_iter()
            .map(|h| PyHit {
                external_id: h.external_id,
                score: h.score,
                seq: h.seq,
            })
            .collect())
    }
}

#[pymodule(name = "_omendb_vector")]
fn omendb_vector(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyStore>()?;
    m.add_class::<PyHit>()?;
    m.add_class::<PyRecovery>()?;
    m.add("VectorEngineError", m.py().get_type::<VectorEngineError>())?;
    Ok(())
}
