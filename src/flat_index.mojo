from distance import l2_distance, dot_product, cosine_distance
from store_types import Metric
from hnsw import Candidate


struct FlatIndex[dim: Int](Movable):
    var data: List[Float32]
    var num_elements: Int
    var metric: Metric

    def __init__(out self, metric: Metric = Metric.L2):
        self.data = List[Float32]()
        self.num_elements = 0
        self.metric = metric

    def __init__(out self, *, deinit take: Self):
        self.data = take.data^
        self.num_elements = take.num_elements
        self.metric = take.metric

    @always_inline
    def _distance(self, a: Span[Float32, _], b: Span[Float32, _]) -> Float32:
        if self.metric == Metric.COSINE:
            return cosine_distance[Self.dim](a, b)
        elif self.metric == Metric.DOT:
            return -dot_product[Self.dim](a, b)
        return l2_distance[Self.dim](a, b)

    def reserve(mut self, capacity: Int):
        self.data.reserve(capacity * Self.dim)

    def insert(mut self, vec: Span[Float32, _]):
        for i in range(Self.dim):
            self.data.append(vec[i])
        self.num_elements += 1

    def get_vector(ref self, id: Int) -> Span[Float32, origin_of(self.data)]:
        var start = id * Self.dim
        return Span(ptr=self.data.unsafe_ptr() + start, length=Self.dim)

    def search(ref self, query: Span[Float32, _], k: Int) -> List[Candidate]:
        var results = List[Candidate]()

        for i in range(self.num_elements):
            var vec = self.get_vector(i)
            var dist = self._distance(query, vec)
            results.append(Candidate(UInt32(i), dist))

        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                if results[i].distance > results[j].distance:
                    var temp = results[i]
                    results[i] = results[j]
                    results[j] = temp

        var top_k = List[Candidate]()
        for i in range(min(k, len(results))):
            top_k.append(results[i])
        return top_k^
