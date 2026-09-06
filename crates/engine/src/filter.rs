//! Typed metadata predicates (v0: AND composition, exact evaluation).
//!
//! A filter is a set of field predicates evaluated exactly against
//! each record's metadata. This is the **oracle semantics** for
//! filtered search; per-field indexes (equality inverted, range
//! sorted, presence bitmaps — architecture §6) are the acceleration
//! that lands with the benchmark harness, and they must reproduce
//! this module's behavior exactly.
//!
//! v0 predicate set: `Eq`, `In`, `Range` (Int/Float), `Presence`.
//! `Range` on integers compares exactly when both bounds are `Int`;
//! mixed Int/Float bounds promote to f64 — **exact only below 2^53**
//! (documented precision boundary: i64 values beyond 2^53 lose exact
//! f64 representation). NaN bounds are rejected loud, never
//! silently-matching. OR composition is out of v0 (needed: ranked
//! union merge order — deferred until a real query demands it).

use crate::error::{EngineError, EngineResult};
use crate::records::{MetaValue, Record};

/// Numeric bound for `Range`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Num {
    Int(i64),
    Float(f64),
}

impl Num {
    fn as_f64(self) -> f64 {
        match self {
            Num::Int(v) => v as f64,
            Num::Float(v) => v,
        }
    }
}

/// One field predicate.
#[derive(Debug, Clone, PartialEq)]
pub enum Predicate {
    /// Field equals this value (type-sensitive: Int(1) != Float(1.0)).
    Eq { field: String, value: MetaValue },
    /// Field equals any of these values (type-sensitive like Eq).
    In {
        field: String,
        values: Vec<MetaValue>,
    },
    /// Numeric field within [lo, hi] (both inclusive). Int/Float
    /// records both match; Int/Int bounds compare exactly.
    Range { field: String, lo: Num, hi: Num },
    /// Field present (any value, including Bool(false)).
    Present { field: String },
}

impl Predicate {
    /// Does `record` satisfy this predicate?
    pub fn matches(&self, record: &Record) -> EngineResult<bool> {
        match self {
            Predicate::Eq { field, value } => Ok(record
                .meta
                .iter()
                .any(|(k, v)| k == field && meta_eq(v, value))),
            Predicate::In { field, values } => Ok(record
                .meta
                .iter()
                .any(|(k, v)| k == field && values.iter().any(|w| meta_eq(v, w)))),
            Predicate::Range { field, lo, hi } => {
                if lo.as_f64() > hi.as_f64() {
                    return Ok(false); // empty range matches nothing
                }
                // Reject NaN bounds loud: they never match anything
                // meaningful and usually mean a caller bug.
                if matches!(lo, Num::Float(f) if f.is_nan())
                    || matches!(hi, Num::Float(f) if f.is_nan())
                {
                    return Err(EngineError::Schema("NaN range bound".into()));
                }
                // Exact i64 path when both bounds are Int.
                let (Some(lo_i), Some(hi_i)) = (lo.int(), hi.int()) else {
                    let (lo_f, hi_f) = (lo.as_f64(), hi.as_f64());
                    return Ok(record.meta.iter().any(|(k, v)| {
                        k == field
                            && match v {
                                MetaValue::Int(x) => {
                                    // Beyond 2^53 the f64 cast is lossy;
                                    // documented boundary.
                                    (*x as f64) >= lo_f && (*x as f64) <= hi_f
                                }
                                MetaValue::Float(x) => *x >= lo_f && *x <= hi_f,
                                _ => false,
                            }
                    }));
                };
                Ok(record.meta.iter().any(|(k, v)| {
                    k == field
                        && match v {
                            MetaValue::Int(x) => *x >= lo_i && *x <= hi_i,
                            MetaValue::Float(x) => {
                                // A float in an int range: exact when it
                                // integral, else no match on the exact path.
                                let Some(xi) = float_to_exact_i64(*x) else {
                                    return false;
                                };
                                xi >= lo_i && xi <= hi_i
                            }
                            _ => false,
                        }
                }))
            }
            Predicate::Present { field } => Ok(record.meta.iter().any(|(k, _)| k == field)),
        }
    }
}

impl Num {
    fn int(self) -> Option<i64> {
        match self {
            Num::Int(v) => Some(v),
            Num::Float(f) => float_to_exact_i64(f),
        }
    }
}

/// f64 -> i64 only when exactly integral and in range.
fn float_to_exact_i64(f: f64) -> Option<i64> {
    if f.is_finite() && f.fract() == 0.0 && f.abs() <= 9.007_199_254_740_992e15 {
        Some(f as i64)
    } else {
        None
    }
}

/// Type-sensitive equality: Int(1) != Float(1.0); strings/bytes/bools
/// compare structurally.
fn meta_eq(a: &MetaValue, b: &MetaValue) -> bool {
    match (a, b) {
        (MetaValue::Int(x), MetaValue::Int(y)) => x == y,
        (MetaValue::Float(x), MetaValue::Float(y)) => x == y,
        (MetaValue::Bool(x), MetaValue::Bool(y)) => x == y,
        (MetaValue::Str(x), MetaValue::Str(y)) => x == y,
        (MetaValue::Bytes(x), MetaValue::Bytes(y)) => x == y,
        _ => false,
    }
}

/// A filter: AND of predicates (empty filter = matches everything).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Filter {
    pub predicates: Vec<Predicate>,
}

impl Filter {
    pub fn new() -> Self {
        Filter::default()
    }

    pub fn and(mut self, predicate: Predicate) -> Self {
        self.predicates.push(predicate);
        self
    }

    /// Does `record` satisfy every predicate?
    pub fn matches(&self, record: &Record) -> EngineResult<bool> {
        for p in &self.predicates {
            if !p.matches(record)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec_with(meta: Vec<(&str, MetaValue)>) -> Record {
        let mut r = Record::new(1, vec![0.1]);
        for (k, v) in meta {
            r = r.with_meta(k, v);
        }
        r
    }

    #[test]
    fn eq_is_type_sensitive() {
        let r = rec_with(vec![("n", MetaValue::Int(1))]);
        assert!(Predicate::Eq {
            field: "n".into(),
            value: MetaValue::Int(1),
        }
        .matches(&r)
        .unwrap());
        assert!(!Predicate::Eq {
            field: "n".into(),
            value: MetaValue::Float(1.0),
        }
        .matches(&r)
        .unwrap());
    }

    #[test]
    fn eq_across_all_kinds() {
        let cases = vec![
            MetaValue::Int(-7),
            MetaValue::Float(2.5),
            MetaValue::Bool(false),
            MetaValue::Str("x".into()),
            MetaValue::Bytes(vec![1, 2]),
        ];
        for v in cases {
            let r = rec_with(vec![("f", v.clone())]);
            assert!(Predicate::Eq {
                field: "f".into(),
                value: v
            }
            .matches(&r)
            .unwrap());
        }
    }

    #[test]
    fn in_matches_any() {
        let r = rec_with(vec![("c", MetaValue::Str("red".into()))]);
        let p = Predicate::In {
            field: "c".into(),
            values: vec![MetaValue::Str("blue".into()), MetaValue::Str("red".into())],
        };
        assert!(p.matches(&r).unwrap());
        let p2 = Predicate::In {
            field: "c".into(),
            values: vec![MetaValue::Str("blue".into())],
        };
        assert!(!p2.matches(&r).unwrap());
    }

    #[test]
    fn range_int_exact() {
        let r = rec_with(vec![("age", MetaValue::Int(30))]);
        let in_range = |lo: Num, hi: Num| {
            Predicate::Range {
                field: "age".into(),
                lo,
                hi,
            }
            .matches(&r)
            .unwrap()
        };
        assert!(in_range(Num::Int(20), Num::Int(40)));
        assert!(!in_range(Num::Int(31), Num::Int(40)));
        // i64 bounds beyond f64-exact range still compare exactly on
        // the int path.
        assert!(in_range(Num::Int(i64::MIN + 1), Num::Int(i64::MAX - 1)));
    }

    #[test]
    fn range_float_bounds_match_int_records() {
        let r = rec_with(vec![("age", MetaValue::Int(30))]);
        let p = Predicate::Range {
            field: "age".into(),
            lo: Num::Float(29.5),
            hi: Num::Float(30.5),
        };
        assert!(p.matches(&r).unwrap());
        let p2 = Predicate::Range {
            field: "age".into(),
            lo: Num::Float(30.5),
            hi: Num::Float(31.0),
        };
        assert!(!p2.matches(&r).unwrap());
    }

    #[test]
    fn range_float_records_match_int_bounds_via_exactness() {
        let r = rec_with(vec![("x", MetaValue::Float(30.0))]);
        let p = Predicate::Range {
            field: "x".into(),
            lo: Num::Int(30),
            hi: Num::Int(30),
        };
        // 30.0 is exactly integral: matches the inclusive int bound.
        assert!(p.matches(&r).unwrap());
        let r2 = rec_with(vec![("x", MetaValue::Float(30.5))]);
        assert!(!p.matches(&r2).unwrap());
    }

    #[test]
    fn range_empty_and_inverted() {
        let r = rec_with(vec![("x", MetaValue::Int(5))]);
        let inverted = Predicate::Range {
            field: "x".into(),
            lo: Num::Int(10),
            hi: Num::Int(1),
        };
        assert!(!inverted.matches(&r).unwrap());
        let nan_lo = Predicate::Range {
            field: "x".into(),
            lo: Num::Float(f64::NAN),
            hi: Num::Float(1.0),
        };
        assert!(nan_lo.matches(&r).is_err());
    }

    #[test]
    fn presence_includes_false_and_zero() {
        let r = rec_with(vec![
            ("b", MetaValue::Bool(false)),
            ("z", MetaValue::Int(0)),
        ]);
        assert!(Predicate::Present { field: "b".into() }
            .matches(&r)
            .unwrap());
        assert!(Predicate::Present { field: "z".into() }
            .matches(&r)
            .unwrap());
        assert!(!Predicate::Present {
            field: "nope".into()
        }
        .matches(&r)
        .unwrap());
    }

    #[test]
    fn and_composition_and_empty() {
        let r = rec_with(vec![
            ("color", MetaValue::Str("red".into())),
            ("age", MetaValue::Int(30)),
        ]);
        let f = Filter::new()
            .and(Predicate::Eq {
                field: "color".into(),
                value: MetaValue::Str("red".into()),
            })
            .and(Predicate::Range {
                field: "age".into(),
                lo: Num::Int(18),
                hi: Num::Int(65),
            });
        assert!(f.matches(&r).unwrap());
        let f2 = f.clone().and(Predicate::Present {
            field: "missing".into(),
        });
        assert!(!f2.matches(&r).unwrap());
        assert!(Filter::new().matches(&r).unwrap());
    }

    #[test]
    fn missing_field_never_matches_predicates() {
        let r = rec_with(vec![]);
        assert!(!Predicate::Eq {
            field: "any".into(),
            value: MetaValue::Int(1),
        }
        .matches(&r)
        .unwrap());
        assert!(!Predicate::Range {
            field: "any".into(),
            lo: Num::Int(0),
            hi: Num::Int(9),
        }
        .matches(&r)
        .unwrap());
    }

    #[test]
    fn duplicate_meta_keys_first_match_wins() {
        // Records allow duplicate keys (Vec-based meta); predicates
        // match if ANY entry matches.
        let mut r = Record::new(1, vec![0.1]);
        r = r.with_meta("tag", MetaValue::Str("a".into()));
        r = r.with_meta("tag", MetaValue::Str("b".into()));
        assert!(Predicate::Eq {
            field: "tag".into(),
            value: MetaValue::Str("b".into()),
        }
        .matches(&r)
        .unwrap());
    }
}
