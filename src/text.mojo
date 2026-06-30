from std.collections import Dict, List
from std.math import log


@fieldwise_init
struct Posting(ImplicitlyCopyable):
    var doc_id: UInt32
    var freq: UInt32


def tokenize(text: String) -> List[String]:
    var words = List[String]()
    var current = String("")

    for char in text.codepoint_slices():
        var c = ord(char)
        # Very simple tokenizer: split by non-alphanumeric
        var is_alnum = (
            (c >= 97 and c <= 122)
            or (c >= 65 and c <= 90)
            or (c >= 48 and c <= 57)
        )
        if is_alnum:
            # Lowercase
            if c >= 65 and c <= 90:
                current += chr(c + 32)
            else:
                current += chr(c)
        else:
            if current.byte_length() > 0:
                words.append(current^)
                current = String("")

    if current.byte_length() > 0:
        words.append(current^)

    return words^


struct BM25Index(Movable):
    var vocabulary: Dict[String, Int]
    var posting_lists: List[List[Posting]]
    var doc_lengths: List[Float32]
    var avg_doc_length: Float32
    var k1: Float32
    var b: Float32
    var num_docs: Int

    def __init__(out self, k1: Float32 = 1.5, b: Float32 = 0.75):
        self.vocabulary = Dict[String, Int]()
        self.posting_lists = List[List[Posting]]()
        self.doc_lengths = List[Float32]()
        self.avg_doc_length = 0.0
        self.k1 = k1
        self.b = b
        self.num_docs = 0

    def __init__(out self, *, deinit take: Self):
        self.vocabulary = take.vocabulary^
        self.posting_lists = take.posting_lists^
        self.doc_lengths = take.doc_lengths^
        self.avg_doc_length = take.avg_doc_length
        self.k1 = take.k1
        self.b = take.b
        self.num_docs = take.num_docs

    def add_document(mut self, text: String) raises -> UInt32:
        var doc_id = UInt32(self.num_docs)
        var words = tokenize(text)

        # Count frequencies in this doc
        var freqs = Dict[String, UInt32]()
        for i in range(len(words)):
            var w = words[i]
            if w in freqs:
                freqs[w] += 1
            else:
                freqs[w] = 1

        var doc_len = 0
        for entry in freqs.items():
            var w = entry.key
            var f = entry.value
            doc_len += Int(f)

            # Add to inverted index
            if w not in self.vocabulary:
                self.vocabulary[w] = len(self.posting_lists)
                self.posting_lists.append(List[Posting]())

            var word_id = self.vocabulary[w]
            self.posting_lists[word_id].append(Posting(doc_id, f))

        self.doc_lengths.append(Float32(doc_len))
        self.num_docs += 1

        # Update avg doc length
        var total_len: Float32 = 0.0
        for i in range(len(self.doc_lengths)):
            total_len += self.doc_lengths[i]
        self.avg_doc_length = total_len / Float32(self.num_docs)

        return doc_id

    def search(
        ref self, query: String, k: Int = 10
    ) raises -> List[Tuple[UInt32, Float32]]:
        var words = tokenize(query)
        var scores = Dict[UInt32, Float32]()

        for i in range(len(words)):
            var w = words[i]
            if w not in self.vocabulary:
                continue

            var word_id = self.vocabulary[w]
            ref postings = self.posting_lists[word_id]

            # Calculate IDF for this word
            # idf(q_i) = log((N - n(q_i) + 0.5) / (n(q_i) + 0.5) + 1)
            var n_qi = len(postings)
            var idf = log(
                (Float32(self.num_docs) - Float32(n_qi) + 0.5)
                / (Float32(n_qi) + 0.5)
                + 1.0
            )

            for j in range(len(postings)):
                var p = postings[j]
                # score(D, Q) = SUM [ idf(q_i) * (f(q_i, D) * (k1 + 1)) / (f(q_i, D) + k1 * (1 - b + b * (|D| / avgdl))) ]
                var f_qi_D = Float32(p.freq)
                var dl = self.doc_lengths[Int(p.doc_id)]

                var numerator = f_qi_D * (self.k1 + 1.0)
                var denominator = f_qi_D + self.k1 * (
                    1.0 - self.b + self.b * (dl / self.avg_doc_length)
                )

                var word_score = idf * (numerator / denominator)

                if p.doc_id in scores:
                    scores[p.doc_id] += word_score
                else:
                    scores[p.doc_id] = word_score

        # Sort results
        var results = List[Tuple[UInt32, Float32]]()
        for entry in scores.items():
            results.append((entry.key, entry.value))

        @parameter
        def cmp(a: Tuple[UInt32, Float32], b: Tuple[UInt32, Float32]) -> Bool:
            return a[1] > b[1]  # higher score first

        var span = Span[Tuple[UInt32, Float32], MutAnyOrigin](
            ptr=results.unsafe_ptr(), length=len(results)
        )
        sort[cmp](span)

        var top_k = List[Tuple[UInt32, Float32]]()
        for i in range(min(k, len(results))):
            top_k.append(results[i])

        return top_k^
