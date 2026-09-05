//! Text tokenizer: split on non-alphanumeric, lowercase, Unicode-aware.
//!
//! The one tokenizer v0 ships. Tantivy's lesson (evidence register
//! §13): tokenization policy is where search quality lives; keep one
//! default, make it predictable, extend deliberately later. Numbers
//! stay (version strings, part numbers); punctuation and whitespace
//! split; everything lowercases. Underscore splits too (learned from
//! the Mojo engine's CI: `install_omendb` must tokenize to
//! `install` + `omendb`, not one blob).

/// Tokenize `text` into lowercase alphanumeric runs.
pub fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            // Lowercase per char; ASCII fast-path is inside char
            // methods anyway.
            cur.extend(ch.to_lowercase());
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Tokenize with positions (for future phrase/proximity features;
/// v0 postings store term frequency only).
pub fn tokenize_with_positions(text: &str) -> Vec<(String, u32)> {
    tokenize(text)
        .into_iter()
        .enumerate()
        .map(|(pos, tok)| (tok, pos as u32))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_split_and_lowercase() {
        assert_eq!(
            tokenize("Hello, Vector-World!"),
            vec!["hello", "vector", "world"]
        );
    }

    #[test]
    fn underscore_splits() {
        assert_eq!(tokenize("install_omendb"), vec!["install", "omendb"]);
    }

    #[test]
    fn numbers_and_versions() {
        assert_eq!(tokenize("v2.1 beta-3"), vec!["v2", "1", "beta", "3"]);
    }

    #[test]
    fn unicode_letters() {
        assert_eq!(
            tokenize("Grüße aus München"),
            vec!["grüße", "aus", "münchen"]
        );
    }

    #[test]
    fn cjk_is_one_blob_per_run() {
        // CJK ideographs are alphanumeric; no dictionary segmentation
        // in v0 — a run becomes one token. Documented limitation.
        let toks = tokenize("向量数据库");
        assert_eq!(toks, vec!["向量数据库"]);
    }

    #[test]
    fn empty_and_punct_only() {
        assert!(tokenize("").is_empty());
        assert!(tokenize("!!! ... ,,, ").is_empty());
    }

    #[test]
    fn whitespace_collapses() {
        assert_eq!(tokenize("  a   b  "), vec!["a", "b"]);
    }

    #[test]
    fn repeated_terms_counted() {
        assert_eq!(tokenize("a a a b"), vec!["a", "a", "a", "b"]);
    }

    #[test]
    fn positions_are_sequential() {
        let toks = tokenize_with_positions("one two three");
        assert_eq!(
            toks,
            vec![("one".into(), 0), ("two".into(), 1), ("three".into(), 2)]
        );
    }
}
