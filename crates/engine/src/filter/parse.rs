//! SQL-subset filter strings -> `Filter` (boolean tree).
//!
//! Grammar (recursive descent, precedence LOW to HIGH):
//!
//! ```text
//! or_expr   := and_expr (OR and_expr)*
//! and_expr  := not_expr (AND not_expr)*
//! not_expr  := NOT not_expr | primary
//! primary   := '(' or_expr ')' | comparison
//! comparison := field op literal
//!            | field IN '(' literal (',' literal)* ')'
//!            | field IS [NOT] NULL
//! op        := = | != | <> | >= | <= | > | <
//! literal   := int | float | true | false | 'string' | "string"
//! ```
//!
//! Conventions (audience: SQL users — sqlite/PG/DuckDB hands):
//! - `field != v` means `NOT field = v` including when the field is
//!   absent (missing != value is TRUE, matching SQL three-valued
//!   logic's practical reading: an absent field is never equal to a
//!   value, so it differs).
//! - Comparisons on strings support only = / != (ranges and
//!   ordering are numeric-only; string collation is out of scope).
//! - `>`/`>=`/`<`/`<=` on numbers only.
//! - Type sensitivity follows `Predicate::Eq`: Int(1) != Float(1.0).
//! - Keywords are case-insensitive (SQL convention); field names are
//!   case-sensitive bare identifiers `[A-Za-z_][A-Za-z0-9_]*`.
//!
//! Errors are loud: unknown syntax is a parse error, never a
//! match-everything filter.

use super::{BoolNode, Filter, Num, Predicate};
use crate::error::{EngineError, EngineResult};
use crate::records::MetaValue;

/// Parse a where-string into a Filter. Empty/whitespace-only input
/// matches everything (same as no filter).
pub fn parse_where(input: &str) -> EngineResult<Filter> {
    let mut p = Parser::new(input);
    p.skip_ws();
    if p.at_end() {
        return Ok(Filter::new());
    }
    let node = p.or_expr()?;
    p.skip_ws();
    if !p.at_end() {
        return Err(p.err("trailing input after expression"));
    }
    Ok(Filter { root: Some(node) })
}

struct Parser<'a> {
    src: &'a [u8],
    pos: usize,
    /// Human-readable span origin for errors.
    src_len: usize,
}

impl<'a> Parser<'a> {
    fn new(s: &'a str) -> Self {
        Parser {
            src: s.as_bytes(),
            pos: 0,
            src_len: s.len(),
        }
    }

    fn err(&self, msg: &str) -> EngineError {
        EngineError::Schema(format!(
            "where-string parse error at byte {}/{}: {msg}",
            self.pos, self.src_len
        ))
    }

    fn at_end(&self) -> bool {
        self.pos >= self.src.len()
    }

    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    /// Consume a keyword only when followed by a non-identifier byte
    /// (so `not_field` is a field name, not NOT + `field`).
    fn eat_keyword(&mut self, kw: &str) -> bool {
        let bytes = kw.as_bytes();
        if self.src.len() >= self.pos + bytes.len()
            && self.src[self.pos..self.pos + bytes.len()].eq_ignore_ascii_case(bytes)
        {
            let after = self.src.get(self.pos + bytes.len()).copied();
            let ident_continue =
                matches!(after, Some(b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_'));
            if !ident_continue {
                self.pos += bytes.len();
                return true;
            }
        }
        false
    }

    fn expect_keyword(&mut self, kw: &str) -> EngineResult<()> {
        if self.eat_keyword(kw) {
            Ok(())
        } else {
            Err(self.err(&format!("expected {kw}")))
        }
    }

    /// Bare identifier: field name.
    fn ident(&mut self) -> EngineResult<String> {
        self.skip_ws();
        let start = self.pos;
        match self.peek() {
            Some(b'a'..=b'z' | b'A'..=b'Z' | b'_') => {}
            _ => return Err(self.err("expected field name")),
        }
        while matches!(
            self.peek(),
            Some(b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_')
        ) {
            self.pos += 1;
        }
        Ok(String::from_utf8_lossy(&self.src[start..self.pos]).into_owned())
    }

    fn eat_char(&mut self, c: u8) -> bool {
        if self.peek() == Some(c) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn expect_char(&mut self, c: u8, what: &str) -> EngineResult<()> {
        if self.eat_char(c) {
            Ok(())
        } else {
            Err(self.err(&format!("expected {what}")))
        }
    }

    // --- grammar ---

    fn or_expr(&mut self) -> EngineResult<BoolNode> {
        let mut left = self.and_expr()?;
        loop {
            self.skip_ws();
            if self.eat_keyword("OR") {
                let right = self.and_expr()?;
                left = BoolNode::Or(Box::new(left), Box::new(right));
            } else {
                return Ok(left);
            }
        }
    }

    fn and_expr(&mut self) -> EngineResult<BoolNode> {
        let mut left = self.not_expr()?;
        loop {
            self.skip_ws();
            if self.eat_keyword("AND") {
                let right = self.not_expr()?;
                left = BoolNode::And(Box::new(left), Box::new(right));
            } else {
                return Ok(left);
            }
        }
    }

    fn not_expr(&mut self) -> EngineResult<BoolNode> {
        self.skip_ws();
        if self.eat_keyword("NOT") {
            let inner = self.not_expr()?;
            return Ok(BoolNode::Not(Box::new(inner)));
        }
        self.primary()
    }

    fn primary(&mut self) -> EngineResult<BoolNode> {
        self.skip_ws();
        if self.eat_char(b'(') {
            let node = self.or_expr()?;
            self.skip_ws();
            self.expect_char(b')', "')' to close group")?;
            return Ok(node);
        }
        Ok(BoolNode::Leaf(self.comparison()?))
    }

    fn comparison(&mut self) -> EngineResult<Predicate> {
        let field = self.ident()?;
        self.skip_ws();

        // IS [NOT] NULL
        if self.eat_keyword("IS") {
            self.skip_ws();
            let negate = self.eat_keyword("NOT");
            self.skip_ws();
            self.expect_keyword("NULL")?;
            return Ok(if negate {
                Predicate::Present { field } // IS NOT NULL: field present
            } else {
                Predicate::NotPresent { field } // IS NULL: field absent
            });
        }

        // IN (...)
        if self.eat_keyword("IN") {
            self.skip_ws();
            self.expect_char(b'(', "'(' after IN")?;
            let mut values = Vec::new();
            loop {
                self.skip_ws();
                values.push(self.literal()?);
                self.skip_ws();
                if self.eat_char(b',') {
                    continue;
                }
                self.expect_char(b')', "')' or ',' in IN list")?;
                break;
            }
            return Ok(Predicate::In { field, values });
        }

        // comparison operators
        if self.eat_char(b'=') {
            let value = self.literal()?;
            return Ok(Predicate::Eq { field, value });
        }
        if self.eat_char(b'!') {
            self.expect_char(b'=', "'!=' operator")?;
            let value = self.literal()?;
            return Ok(Predicate::Neq { field, value });
        }
        if self.eat_char(b'<') {
            if self.eat_char(b'=') {
                let hi = self.number_literal()?;
                return Ok(Predicate::Range {
                    field,
                    lo: Num::Float(f64::NEG_INFINITY),
                    hi,
                });
            }
            if self.eat_char(b'>') {
                let value = self.literal()?;
                return Ok(Predicate::Neq { field, value });
            }
            // field < X: (-inf, X) — exclusive hi widens down
            let hi = self.number_literal()?;
            return Ok(Predicate::Range {
                field,
                lo: Num::Float(f64::NEG_INFINITY),
                hi: next_down(hi),
            });
        }
        if self.eat_char(b'>') {
            if self.eat_char(b'=') {
                let lo = self.number_literal()?;
                return Ok(Predicate::Range {
                    field,
                    lo,
                    hi: Num::Float(f64::INFINITY),
                });
            }
            // field > X: (X, +inf) — exclusive lo widens up
            let lo = self.number_literal()?;
            return Ok(Predicate::Range {
                field,
                lo: next_up(lo),
                hi: Num::Float(f64::INFINITY),
            });
        }
        Err(self.err(&format!(
            "expected comparison operator after field {field:?}"
        )))
    }

    fn number_literal(&mut self) -> EngineResult<Num> {
        self.skip_ws();
        let start = self.pos;
        if matches!(self.peek(), Some(b'-' | b'+')) {
            self.pos += 1;
        }
        let mut saw_digit = false;
        let mut saw_dot = false;
        while let Some(c) = self.peek() {
            match c {
                b'0'..=b'9' => {
                    saw_digit = true;
                    self.pos += 1;
                }
                b'.' if !saw_dot => {
                    saw_dot = true;
                    self.pos += 1;
                }
                _ => break,
            }
        }
        if !saw_digit {
            return Err(self.err("expected number"));
        }
        let text =
            std::str::from_utf8(&self.src[start..self.pos]).map_err(|_| self.err("bad number"))?;
        if saw_dot {
            let f: f64 = text.parse().map_err(|_| self.err("bad float"))?;
            if f.is_nan() {
                return Err(self.err("NaN bound"));
            }
            Ok(Num::Float(f))
        } else {
            match text.parse::<i64>() {
                Ok(i) => Ok(Num::Int(i)),
                // ints beyond i64 fall to float (documented boundary)
                Err(_) => Ok(Num::Float(text.parse().map_err(|_| self.err("bad int"))?)),
            }
        }
    }

    fn string_literal(&mut self) -> EngineResult<String> {
        let quote = self.peek().ok_or_else(|| self.err("expected string"))?;
        if quote != b'\'' && quote != b'"' {
            return Err(self.err("expected quoted string"));
        }
        self.pos += 1;
        let mut out = Vec::new();
        loop {
            match self.peek() {
                None => return Err(self.err("unterminated string")),
                Some(c) if c == quote => {
                    self.pos += 1;
                    break;
                }
                Some(b'\\') => {
                    // escape: \' \" \\ only (SQL-style doubling of the
                    // quote also accepted: '' -> ')
                    self.pos += 1;
                    match self.peek() {
                        Some(b'\\' | b'\'' | b'"') => {
                            out.push(self.src[self.pos]);
                            self.pos += 1;
                        }
                        None => return Err(self.err("dangling escape")),
                        _ => return Err(self.err("unknown escape in string")),
                    }
                }
                Some(c) => {
                    out.push(c);
                    self.pos += 1;
                }
            }
        }
        String::from_utf8(out).map_err(|_| self.err("invalid utf-8 in string literal"))
    }

    fn literal(&mut self) -> EngineResult<MetaValue> {
        self.skip_ws();
        match self.peek() {
            Some(b'\'') | Some(b'"') => Ok(MetaValue::Str(self.string_literal()?)),
            Some(b'-' | b'0'..=b'9') => {
                // int if integral, float if it has a dot
                let start = self.pos;
                let save = self.pos;
                if matches!(self.peek(), Some(b'-')) {
                    self.pos += 1;
                }
                let mut saw_digit = false;
                let mut saw_dot = false;
                while let Some(c) = self.peek() {
                    match c {
                        b'0'..=b'9' => {
                            saw_digit = true;
                            self.pos += 1;
                        }
                        b'.' if !saw_dot => {
                            saw_dot = true;
                            self.pos += 1;
                        }
                        _ => break,
                    }
                }
                let _ = start;
                if !saw_digit {
                    self.pos = save;
                    return Err(self.err("expected literal"));
                }
                let text = std::str::from_utf8(&self.src[save..self.pos])
                    .map_err(|_| self.err("bad literal"))?;
                if saw_dot {
                    let f: f64 = text.parse().map_err(|_| self.err("bad float"))?;
                    if f.is_nan() {
                        return Err(self.err("NaN literal"));
                    }
                    Ok(MetaValue::Float(f))
                } else {
                    Ok(MetaValue::Int(
                        text.parse().map_err(|_| self.err("bad int"))?,
                    ))
                }
            }
            _ => {
                // true / false / null keywords
                if self.eat_keyword("true") {
                    Ok(MetaValue::Bool(true))
                } else if self.eat_keyword("false") {
                    Ok(MetaValue::Bool(false))
                } else if self.eat_keyword("null") {
                    // `field = null` is never useful; SQL says NULL =
                    // NULL is not true. We treat it as a parse error to
                    // keep the loud-errors discipline.
                    Err(self.err("null is only valid with IS [NOT] NULL"))
                } else {
                    Err(self.err("expected literal (number, string, true, false)"))
                }
            }
        }
    }
}

/// Smallest representable step above `n` (exclusive-bound widening
/// for `>`): +inf maps to itself; ints step by one when exactly
/// representable, else one ULP.
fn next_up(n: Num) -> Num {
    match n {
        Num::Int(i) => match i64::try_from(i as i128 + 1) {
            Ok(next) => Num::Int(next),
            Err(_) => Num::Float(f64::INFINITY),
        },
        Num::Float(f) => {
            if f == f64::INFINITY {
                Num::Float(f64::INFINITY)
            } else {
                Num::Float(f.next_up())
            }
        }
    }
}

/// Largest representable step below `n` (exclusive-bound widening
/// for `<`).
fn next_down(n: Num) -> Num {
    match n {
        Num::Int(i) => match i64::try_from(i as i128 - 1) {
            Ok(next) => Num::Int(next),
            Err(_) => Num::Float(f64::NEG_INFINITY),
        },
        Num::Float(f) => {
            if f == f64::NEG_INFINITY {
                Num::Float(f64::NEG_INFINITY)
            } else {
                Num::Float(f.next_down())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::parse::parse_where;
    use super::*;
    use crate::records::{MetaValue, Record};

    fn rec(meta: Vec<(&str, MetaValue)>) -> Record {
        let mut r = Record::new(1, vec![0.1]);
        for (k, v) in meta {
            r = r.with_meta(k, v);
        }
        r
    }

    fn m(s: &str, r: &Record) -> bool {
        parse_where(s).unwrap().matches(r).unwrap()
    }

    #[test]
    fn equality_and_cases() {
        let r = rec(vec![
            ("lang", MetaValue::Str("en".into())),
            ("year", MetaValue::Int(2024)),
        ]);
        assert!(m("lang = 'en'", &r));
        assert!(!m("lang = \"fr\"", &r));
        assert!(m("year = 2024", &r));
        assert!(!m("year = 2024.0", &r)); // Int(2024) != Float(2024.0)
        assert!(m("lang != 'fr'", &r));
        assert!(m("lang <> 'fr'", &r));
    }

    #[test]
    fn comparisons_and_ranges() {
        let r = rec(vec![("year", MetaValue::Int(2024))]);
        assert!(m("year >= 2024", &r));
        assert!(m("year > 2023", &r));
        assert!(!m("year > 2024", &r));
        assert!(m("year < 2025", &r));
        assert!(m("year <= 2024", &r));
        // half-open intervals compare integers correctly via f64
        // (2024 < 2^53)
        assert!(m("year > 2023.5", &r));
    }

    #[test]
    fn in_lists() {
        let r = rec(vec![("lang", MetaValue::Str("en".into()))]);
        assert!(m("lang IN ('en', 'fr')", &r));
        assert!(!m("lang IN ('fr', 'de')", &r));
        assert!(m("lang IN ('de', 'en')", &r));
    }

    #[test]
    fn null_handling() {
        let with = rec(vec![("x", MetaValue::Int(1))]);
        let without = rec(vec![]);
        assert!(m("x IS NULL", &without));
        assert!(!m("x IS NULL", &with));
        assert!(m("x IS NOT NULL", &with));
        assert!(!m("x IS NOT NULL", &without));
    }

    #[test]
    fn boolean_composition_and_precedence() {
        let r = rec(vec![
            ("lang", MetaValue::Str("en".into())),
            ("year", MetaValue::Int(2024)),
        ]);
        // AND binds tighter than OR
        assert!(m("lang = 'en' OR lang = 'fr' AND year = 1999", &r));
        // parens override
        assert!(!m("(lang = 'en' OR lang = 'fr') AND year = 1999", &r));
        // NOT
        assert!(m("NOT lang = 'fr'", &r));
        assert!(!m("NOT (lang = 'en' OR lang = 'fr')", &r));
        assert!(m("NOT NOT lang = 'en'", &r));
        // case-insensitive keywords
        assert!(m("lang = 'en' and year = 2024", &r));
        assert!(!m("lang = 'fr' or year = 1999", &r));
    }

    #[test]
    fn missing_field_semantics() {
        let r = rec(vec![("lang", MetaValue::Str("en".into()))]);
        // absent != value is TRUE (differs)
        assert!(m("missing != 'x'", &r));
        // absent = value is FALSE
        assert!(!m("missing = 'x'", &r));
        // absent IN (...) is FALSE
        assert!(!m("missing IN ('x')", &r));
        // absent > 0 is FALSE (no value to compare)
        assert!(!m("missing > 0", &r));
        // NOT absent = 'x' is TRUE
        assert!(m("NOT missing = 'x'", &r));
    }

    #[test]
    fn empty_where_matches_all() {
        let r = rec(vec![]);
        assert!(m("", &r));
        assert!(m("   ", &r));
    }

    #[test]
    fn type_sensitive_eq() {
        let r = rec(vec![("n", MetaValue::Int(1))]);
        assert!(m("n = 1", &r));
        assert!(!m("n = 1.0", &r));
        let rf = rec(vec![("n", MetaValue::Float(1.0))]);
        assert!(m("n = 1.0", &rf));
        assert!(!m("n = 1", &rf));
    }

    #[test]
    fn bool_literals() {
        let r = rec(vec![("ok", MetaValue::Bool(true))]);
        assert!(m("ok = true", &r));
        assert!(!m("ok = false", &r));
    }

    #[test]
    fn parse_errors_are_loud() {
        for bad in [
            "lang =",
            "lang = 'unterminated",
            "= 'en'",
            "lang IN 'en'",
            "lang IN ()",
            "year >",
            "year > 'notanumber'",
            "lang = null",
            "lang IS",
            "lang IS NOTTL",
            "AND lang = 'en'",
            "lang = 'en' AND",
            "lang = 'en' TRAILING",
            "(lang = 'en'",
            "lang = 'en')",
            "1 = 1",
            "not_a_operator 'x'",
        ] {
            assert!(
                parse_where(bad).is_err(),
                "expected parse error for {bad:?}"
            );
        }
    }

    #[test]
    fn oracle_equality_with_builder() {
        // The string dialect must be exactly equivalent to the
        // builder on every generated case: same records, same
        // verdicts. This is the oracle-equality acceptance test.
        let records = [
            rec(vec![
                ("lang", MetaValue::Str("en".into())),
                ("year", MetaValue::Int(2023)),
                ("score", MetaValue::Float(0.75)),
                ("ok", MetaValue::Bool(true)),
            ]),
            rec(vec![
                ("lang", MetaValue::Str("fr".into())),
                ("year", MetaValue::Int(2024)),
                ("score", MetaValue::Float(0.9)),
                ("ok", MetaValue::Bool(false)),
            ]),
            rec(vec![("lang", MetaValue::Str("de".into()))]),
            rec(vec![]),
        ];
        let cases: Vec<(&str, Filter)> = vec![
            (
                "lang = 'en'",
                Filter::new().and(Predicate::Eq {
                    field: "lang".into(),
                    value: MetaValue::Str("en".into()),
                }),
            ),
            (
                "year >= 2024 AND ok = true",
                Filter::new()
                    .and(Predicate::Range {
                        field: "year".into(),
                        lo: Num::Int(2024),
                        hi: Num::Float(f64::INFINITY),
                    })
                    .and(Predicate::Eq {
                        field: "ok".into(),
                        value: MetaValue::Bool(true),
                    }),
            ),
            (
                "lang IN ('en', 'fr') OR year = 2023",
                Filter::new()
                    .and_filter(parse_where("lang IN ('en', 'fr') OR year = 2023").unwrap()),
            ),
        ];
        for (src, built) in &cases {
            let parsed = parse_where(src).unwrap();
            for (i, r) in records.iter().enumerate() {
                assert_eq!(
                    parsed.matches(r).unwrap(),
                    built.matches(r).unwrap(),
                    "oracle divergence on {src:?} record {i}"
                );
            }
        }
    }
}
