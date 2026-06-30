"""BM25 text index and RRF fusion for hybrid search.

Implements BM25 scoring with an inverted index, plus Reciprocal Rank Fusion
for combining vector search and BM25 results.

BM25 formula (Robertson et al. 1994):
    score(D, Q) = Σ IDF(qi) * (f(qi,D) * (k1+1)) / (f(qi,D) + k1*(1-b+b*|D|/avgdl))

RRF (Cormack et al. 2009):
    rrf_score = Σ 1 / (k + rank_i + 1)
    Default k = 60.
"""

from math import log


struct BM25Result(Copyable, Movable):
    var doc_id: Int
    var score: Float32

    def __init__(out self, doc_id: Int, score: Float32):
        self.doc_id = doc_id
        self.score = score

    def __init__(out self, *, deinit take: Self):
        self.doc_id = take.doc_id
        self.score = take.score


struct BM25Index(Movable):
    """BM25 inverted index for text search."""
    var inverted_index: Dict[String, List[Int]]
    var doc_lengths: List[Int]
    var num_docs: Int
    var total_length: Int
    var k1: Float32
    var b: Float32

    def __init__(out self, k1: Float32 = 1.2, b: Float32 = 0.75):
        self.inverted_index = Dict[String, List[Int]]()
        self.doc_lengths = List[Int]()
        self.num_docs = 0
        self.total_length = 0
        self.k1 = k1
        self.b = b

    def __init__(out self, *, deinit take: Self):
        self.inverted_index = take.inverted_index^
        self.doc_lengths = take.doc_lengths^
        self.num_docs = take.num_docs
        self.total_length = take.total_length
        self.k1 = take.k1
        self.b = take.b

    def avg_doc_length(self) -> Float32:
        if self.num_docs == 0:
            return 0.0
        return Float32(self.total_length) / Float32(self.num_docs)

    def index(mut self, doc_id: Int, text: String) raises:
        """Index a document's text content."""
        var tokens = self._tokenize(text)
        var token_count = len(tokens)

        # Update document length
        while len(self.doc_lengths) <= doc_id:
            self.doc_lengths.append(0)
        self.doc_lengths[doc_id] = token_count
        self.num_docs += 1
        self.total_length += token_count

        # Count term frequencies
        var tf = Dict[String, Int]()
        for i in range(len(tokens)):
            var term = tokens[i]
            if term in tf:
                tf[term] = tf[term] + 1
            else:
                tf[term] = 1

        # Add to inverted index (copy-modify-store for Dict values)
        var terms = List[String]()
        for term in tf:
            terms.append(term)
        for term in terms:
            var count = tf[term]
            var posting = List[Int]()
            if term in self.inverted_index:
                posting = self.inverted_index[term].copy()
            for _ in range(count):
                posting.append(doc_id)
            self.inverted_index[term] = posting^

    def search(mut self, query: String, k: Int = 10) raises -> List[BM25Result]:
        """Search the index with a text query. Returns top-k by BM25 score."""
        var query_tokens = self._tokenize(query)
        if len(query_tokens) == 0 or self.num_docs == 0:
            return List[BM25Result]()

        var avgdl = self.avg_doc_length()
        var scores = Dict[Int, Float32]()

        # Score documents for each query term
        for i in range(len(query_tokens)):
            var term = query_tokens[i]
            if term not in self.inverted_index:
                continue

            var posting = self.inverted_index[term].copy()
            var posting_len = len(posting)

            # Count unique documents in posting list
            var unique_docs = Dict[Int, Bool]()
            for j in range(posting_len):
                unique_docs[posting[j]] = True
            var n = Float32(len(unique_docs))
            var idf = log((Float32(self.num_docs) - n + 0.5) / (n + 0.5) + 1.0)

            # Score each document
            for j in range(posting_len):
                var doc_id = posting[j]
                # Count tf for this doc (simple: count occurrences in posting)
                var tf_count = 0
                for jj in range(posting_len):
                    if posting[jj] == doc_id:
                        tf_count += 1
                var doc_len = Float32(self.doc_lengths[doc_id])
                var tf_norm = (Float32(tf_count) * (self.k1 + 1.0)) / (
                    Float32(tf_count) + self.k1 * (1.0 - self.b + self.b * doc_len / avgdl)
                )
                var score = idf * tf_norm

                if doc_id in scores:
                    scores[doc_id] = scores[doc_id] + score
                else:
                    scores[doc_id] = score

        # Partial sort for top-k
        var doc_ids = List[Int]()
        for doc_id in scores:
            doc_ids.append(doc_id)

        var results = List[BM25Result]()
        var n_results = min(k, len(doc_ids))
        for i in range(n_results):
            var best_idx = i
            for j in range(i + 1, len(doc_ids)):
                if scores[doc_ids[j]] > scores[doc_ids[best_idx]]:
                    best_idx = j
            var tmp = doc_ids[i]
            doc_ids[i] = doc_ids[best_idx]
            doc_ids[best_idx] = tmp
            results.append(BM25Result(doc_ids[i], scores[doc_ids[i]]))

        return results^

    def _tokenize(self, text: String) -> List[String]:
        """Simple whitespace + lowercase tokenization."""
        var lower = text.lower()
        var tokens = List[String]()
        var current = String()
        for i in range(lower.byte_length()):
            var ch = lower[byte=i]
            if ch == 32 or ch == 9 or ch == 10 or ch == 13:  # space/tab/newline/cr
                if current.byte_length() > 0:
                    tokens.append(current)
                    current = String()
            else:
                current += chr(Int(ch))
        if current.byte_length() > 0:
            tokens.append(current)
        return tokens^


def rrf_fusion(
    vector_results: List[BM25Result],
    bm25_results: List[BM25Result],
    limit: Int = 10,
    rrf_k: Int = 60,
    alpha: Float32 = 0.5,
) raises -> List[BM25Result]:
    """Reciprocal Rank Fusion for combining vector and BM25 results.

    rrf_score = Σ 1 / (k + rank + 1)
    Weighted: alpha for vector, (1-alpha) for BM25.
    """
    var scores = Dict[Int, Float32]()

    # Vector results
    for i in range(len(vector_results)):
        var rrf_score = 1.0 / Float32(rrf_k + i + 1)
        var doc_id = vector_results[i].doc_id
        if doc_id in scores:
            scores[doc_id] = scores[doc_id] + alpha * rrf_score
        else:
            scores[doc_id] = alpha * rrf_score

    # BM25 results
    for i in range(len(bm25_results)):
        var rrf_score = 1.0 / Float32(rrf_k + i + 1)
        var doc_id = bm25_results[i].doc_id
        if doc_id in scores:
            scores[doc_id] = scores[doc_id] + (1.0 - alpha) * rrf_score
        else:
            scores[doc_id] = (1.0 - alpha) * rrf_score

    # Partial sort for top-k
    var doc_ids = List[Int]()
    for doc_id in scores:
        doc_ids.append(doc_id)

    var results = List[BM25Result]()
    var n_results = min(limit, len(doc_ids))
    for i in range(n_results):
        var best_idx = i
        for j in range(i + 1, len(doc_ids)):
            if scores[doc_ids[j]] > scores[doc_ids[best_idx]]:
                best_idx = j
        var tmp = doc_ids[i]
        doc_ids[i] = doc_ids[best_idx]
        doc_ids[best_idx] = tmp
        results.append(BM25Result(doc_ids[i], scores[doc_ids[i]]))

    return results^
