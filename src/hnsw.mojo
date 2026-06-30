from std.collections import Set
from std.utils.lock import BlockingSpinLock
from std.sys.intrinsics import prefetch, PrefetchOptions
from distance import l2_distance, dot_product, cosine_distance
from store_types import Metric


@always_inline
def _metric_distance[dim: Int](
    metric: Metric, a: Span[Float32, _], b: Span[Float32, _]
) -> Float32:
    if metric == Metric.COSINE:
        return cosine_distance[dim](a, b)
    elif metric == Metric.DOT:
        return -dot_product[dim](a, b)
    return l2_distance[dim](a, b)
from quantization import SQ8Quantizer


@fieldwise_init
struct Candidate(ImplicitlyCopyable):
    var id: UInt32
    var distance: Float32


struct FlashSQ8ADT[dim: Int](Movable):
    var query: List[Float32]  # [dim] - stored for direct distance
    var scale: Float32
    var min_val: Float32

    def __init__(out self, ref q: SQ8Quantizer[Self.dim]):
        self.query = List[Float32]()
        for _ in range(Self.dim):
            self.query.append(0.0)
        self.scale = q.scale
        self.min_val = q.min_val

    def __init__(out self, *, deinit take: Self):
        self.query = take.query^
        self.scale = take.scale
        self.min_val = take.min_val

    @always_inline
    def build_for_query(
        mut self, query: Span[Float32, _], ref q: SQ8Quantizer[Self.dim]
    ):
        # Store query for direct distance computation using SIMD
        var dst_ptr = self.query.unsafe_ptr()
        var src_ptr = query.unsafe_ptr()
        var d: Int = 0
        while d + 8 <= Self.dim:
            var vec = src_ptr.load[width=8](d)
            dst_ptr.store[width=8](d, vec)
            d += 8
        for i in range(d, Self.dim):
            dst_ptr[i] = src_ptr[i]

    @always_inline
    def distance(ref self, codes: Span[UInt8, _]) -> Float32:
        # Direct SIMD distance computation - avoids table lookup
        var codes_ptr = codes.unsafe_ptr()
        var query_ptr = self.query.unsafe_ptr()
        var scale = self.scale
        var min_val = self.min_val

        # Use 2 accumulators with 8-wide SIMD (optimal for M3 Max)
        var acc0 = SIMD[DType.float32, 8](0.0)
        var acc1 = SIMD[DType.float32, 8](0.0)
        var d: Int = 0
        while d + 16 <= Self.dim:
            var code_vec0 = codes_ptr.load[width=8](d)
            var quantized0 = code_vec0.cast[DType.float32]() * scale + min_val
            var q_vec0 = query_ptr.load[width=8](d)
            var diff0 = quantized0 - q_vec0
            acc0 += diff0 * diff0

            var code_vec1 = codes_ptr.load[width=8](d + 8)
            var quantized1 = code_vec1.cast[DType.float32]() * scale + min_val
            var q_vec1 = query_ptr.load[width=8](d + 8)
            var diff1 = quantized1 - q_vec1
            acc1 += diff1 * diff1
            d += 16

        var dist: Float32 = (acc0 + acc1).reduce_add()
        # Handle remaining dimensions
        for i in range(d, Self.dim):
            var code_val = (
                Float32(Int(codes_ptr.load[1](i)[0])) * scale + min_val
            )
            var diff = code_val - self.query[i]
            dist += diff * diff
        return dist

    @always_inline
    def distance_direct(
        ref self,
        codes: Span[UInt8, _],
        query: Span[Float32, _],
        scale: Float32,
        min_val: Float32,
    ) -> Float32:
        """Alternative distance that computes directly without table lookup."""
        var codes_ptr = codes.unsafe_ptr()
        var query_ptr = query.unsafe_ptr()

        var acc = SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0)
        var d: Int = 0
        while d + 4 <= Self.dim:
            # Load 4 codes
            var code_vec = codes_ptr.load[width=4](d)
            # Convert to float and scale
            var quantized = code_vec.cast[DType.float32]() * scale + min_val
            # Load query values
            var q_vec = query_ptr.load[width=4](d)
            # Compute squared difference
            var diff = quantized - q_vec
            acc += diff * diff
            d += 4

        var dist: Float32 = acc.reduce_add()
        # Handle remaining
        for i in range(d, Self.dim):
            var q_val = Float32(i) * scale + min_val
            var diff = q_val - query[i]
            dist += diff * diff
        return dist


struct MinHeap(Movable):
    alias D = 4  # 4-ary heap for better cache locality
    var data: List[Candidate]

    def __init__(out self):
        self.data = List[Candidate]()

    def __init__(out self, *, deinit take: Self):
        self.data = take.data^

    def len(self) -> Int:
        return len(self.data)

    def reserve(mut self, capacity: Int):
        self.data.reserve(capacity)

    def clear(mut self):
        self.data.clear()

    def push(mut self, c: Candidate):
        self.data.append(c)
        var n = len(self.data)
        var ptr = self.data.unsafe_ptr()
        var i = n - 1
        while i > 0:
            var parent = (i - 1) // Self.D
            var i_dist = (ptr + i)[].distance
            var p_dist = (ptr + parent)[].distance
            if i_dist < p_dist:
                var tmp = (ptr + i)[]
                (ptr + i)[] = (ptr + parent)[]
                (ptr + parent)[] = tmp
                i = parent
            else:
                break

    def pop(mut self) -> Candidate:
        var ptr = self.data.unsafe_ptr()
        var result = ptr[0]
        var n = len(self.data)
        ptr[0] = ptr[n - 1]
        _ = self.data.pop()
        self._sift_down(0)
        return result

    @always_inline
    def peek(ref self) -> Candidate:
        return self.data.unsafe_ptr()[0]

    @always_inline
    def _sift_down(mut self, idx: Int):
        var n = len(self.data)
        var ptr = self.data.unsafe_ptr()
        var pos = idx
        while True:
            var smallest = pos
            var first_child = Self.D * pos + 1
            var pos_dist = (ptr + pos)[].distance
            for c in range(Self.D):
                var child = first_child + c
                if child < n:
                    var child_dist = (ptr + child)[].distance
                    if child_dist < pos_dist:
                        smallest = child
                        pos_dist = child_dist
            if smallest == pos:
                break
            var tmp = (ptr + pos)[]
            (ptr + pos)[] = (ptr + smallest)[]
            (ptr + smallest)[] = tmp
            pos = smallest


struct MaxHeap(Movable):
    alias D = 4  # 4-ary heap for better cache locality
    var data: List[Candidate]

    def __init__(out self):
        self.data = List[Candidate]()

    def __init__(out self, *, deinit take: Self):
        self.data = take.data^

    def len(self) -> Int:
        return len(self.data)

    def reserve(mut self, capacity: Int):
        self.data.reserve(capacity)

    def clear(mut self):
        self.data.clear()

    def push(mut self, c: Candidate):
        self.data.append(c)
        var n = len(self.data)
        var ptr = self.data.unsafe_ptr()
        var i = n - 1
        while i > 0:
            var parent = (i - 1) // Self.D
            var i_dist = (ptr + i)[].distance
            var p_dist = (ptr + parent)[].distance
            if i_dist > p_dist:
                var tmp = (ptr + i)[]
                (ptr + i)[] = (ptr + parent)[]
                (ptr + parent)[] = tmp
                i = parent
            else:
                break

    def pop(mut self) -> Candidate:
        var ptr = self.data.unsafe_ptr()
        var result = ptr[0]
        var n = len(self.data)
        ptr[0] = ptr[n - 1]
        _ = self.data.pop()
        self._sift_down(0)
        return result

    @always_inline
    def peek(ref self) -> Candidate:
        return self.data.unsafe_ptr()[0]

    @always_inline
    def worst_distance(ref self) -> Float32:
        if len(self.data) == 0:
            return 1e30
        return self.data.unsafe_ptr()[0].distance

    def push_bounded(mut self, c: Candidate, max_size: Int):
        var n = len(self.data)
        if n < max_size:
            self.push(c)
        elif c.distance < self.data.unsafe_ptr()[0].distance:
            self.data.unsafe_ptr()[0] = c
            self._sift_down(0)

    @always_inline
    def _sift_down(mut self, idx: Int):
        var n = len(self.data)
        var ptr = self.data.unsafe_ptr()
        var pos = idx
        while True:
            var largest = pos
            var first_child = Self.D * pos + 1
            var pos_dist = (ptr + pos)[].distance
            for c in range(Self.D):
                var child = first_child + c
                if child < n:
                    var child_dist = (ptr + child)[].distance
                    if child_dist > pos_dist:
                        largest = child
                        pos_dist = child_dist
            if largest == pos:
                break
            var tmp = (ptr + pos)[]
            (ptr + pos)[] = (ptr + largest)[]
            (ptr + largest)[] = tmp
            pos = largest


struct HNSWSearchScratch[dim: Int](Movable):
    var adt: FlashSQ8ADT[Self.dim]
    var visit_marks: List[UInt32]
    var visit_generation: UInt32
    var candidates: MinHeap
    var found: MaxHeap
    var reranked: List[Candidate]

    def __init__(out self, ref q: SQ8Quantizer[Self.dim]):
        self.adt = FlashSQ8ADT[Self.dim](q)
        self.visit_marks = List[UInt32]()
        self.visit_generation = 0
        self.candidates = MinHeap()
        self.found = MaxHeap()
        self.reranked = List[Candidate]()

    def __init__(out self, *, deinit take: Self):
        self.adt = take.adt^
        self.visit_marks = take.visit_marks^
        self.visit_generation = take.visit_generation
        self.candidates = take.candidates^
        self.found = take.found^
        self.reranked = take.reranked^

    def prepare_query(
        mut self,
        query: Span[Float32, _],
        ref q: SQ8Quantizer[Self.dim],
        num_elements: Int,
    ):
        self.adt.build_for_query(query, q)
        self.ensure_visit_capacity(num_elements)

    def ensure_visit_capacity(mut self, num_elements: Int):
        self.visit_marks.reserve(num_elements)
        while len(self.visit_marks) < num_elements:
            self.visit_marks.append(0)

    def next_visit_generation(mut self) -> UInt32:
        self.visit_generation += 1
        if self.visit_generation == 0:
            for i in range(len(self.visit_marks)):
                self.visit_marks[i] = 0
            self.visit_generation = 1
        return self.visit_generation

    def begin_layer(mut self, num_elements: Int, ef: Int) -> UInt32:
        self.ensure_visit_capacity(num_elements)
        self.candidates.clear()
        self.found.clear()
        self.candidates.reserve(ef * 2)
        self.found.reserve(ef)
        return self.next_visit_generation()


struct HNSWGraphStorage(Movable):
    var layer_neighbors: List[List[UInt32]]
    var layer_counts: List[List[UInt32]]
    var row_locks: List[
        List[UnsafePointer[BlockingSpinLock, MutExternalOrigin]]
    ]

    def __init__(out self):
        self.layer_neighbors = List[List[UInt32]]()
        self.layer_counts = List[List[UInt32]]()
        self.row_locks = List[
            List[UnsafePointer[BlockingSpinLock, MutExternalOrigin]]
        ]()

    def __init__(out self, *, deinit take: Self):
        self.layer_neighbors = take.layer_neighbors^
        self.layer_counts = take.layer_counts^
        self.row_locks = take.row_locks^

    def __del__(deinit self):
        for layer in range(len(self.row_locks)):
            for row in range(len(self.row_locks[layer])):
                var ptr = self.row_locks[layer][row]
                ptr.destroy_pointee()
                ptr.free()

    def reserve_layers(mut self, capacity: Int):
        self.layer_neighbors.reserve(capacity)
        self.layer_counts.reserve(capacity)
        self.row_locks.reserve(capacity)

    def _new_row_lock(
        self,
    ) -> UnsafePointer[BlockingSpinLock, MutExternalOrigin]:
        var ptr = alloc[BlockingSpinLock](1)
        ptr[] = BlockingSpinLock()
        return ptr

    def load_from_flat(
        mut self,
        layer_neighbors: List[List[UInt32]],
        layer_counts: List[List[UInt32]],
    ):
        for layer in range(len(self.row_locks)):
            for row in range(len(self.row_locks[layer])):
                var ptr = self.row_locks[layer][row]
                ptr.destroy_pointee()
                ptr.free()

        self.layer_neighbors = layer_neighbors.copy()
        self.layer_counts = layer_counts.copy()
        self.row_locks = List[
            List[UnsafePointer[BlockingSpinLock, MutExternalOrigin]]
        ]()
        self.row_locks.reserve(len(self.layer_counts))
        for layer in range(len(self.layer_counts)):
            self.row_locks.append(
                List[UnsafePointer[BlockingSpinLock, MutExternalOrigin]]()
            )
            self.row_locks[layer].reserve(len(self.layer_counts[layer]))
            for _ in range(len(self.layer_counts[layer])):
                self.row_locks[layer].append(self._new_row_lock())

    def layer_count(ref self) -> Int:
        return len(self.layer_counts)

    def node_count(ref self, layer: Int) -> Int:
        return len(self.layer_counts[layer])

    def max_conn_for_layer(
        ref self, layer: Int, layer0_max_conn: Int, upper_max_conn: Int
    ) -> Int:
        return layer0_max_conn if layer == 0 else upper_max_conn

    def slot_capacity(
        ref self, layer: Int, layer0_max_conn: Int, upper_max_conn: Int
    ) -> Int:
        return len(self.layer_neighbors[layer])

    def layer_neighbors_len(ref self, layer: Int) -> Int:
        return len(self.layer_neighbors[layer])

    def layer_counts_len(ref self, layer: Int) -> Int:
        return len(self.layer_counts[layer])

    def row_lock_count(ref self, layer: Int) -> Int:
        return len(self.row_locks[layer])

    def lock_row(mut self, node: Int, layer: Int, owner: Int):
        self.row_locks[layer][node][].lock(owner)

    def unlock_row(mut self, node: Int, layer: Int, owner: Int) -> Bool:
        return self.row_locks[layer][node][].unlock(owner)

    def layer_neighbors_copy(ref self, layer: Int) -> List[UInt32]:
        return self.layer_neighbors[layer].copy()

    def layer_counts_copy(ref self, layer: Int) -> List[UInt32]:
        return self.layer_counts[layer].copy()

    @always_inline
    def row_start(
        ref self,
        node: Int,
        layer: Int,
        layer0_max_conn: Int,
        upper_max_conn: Int,
    ) -> Int:
        return node * self.max_conn_for_layer(
            layer, layer0_max_conn, upper_max_conn
        )

    def count(ref self, node: Int, layer: Int) -> Int:
        return Int(self.layer_counts[layer][node])

    def set_count(mut self, node: Int, layer: Int, count: Int):
        self.layer_counts[layer][node] = UInt32(count)

    def increment_count(mut self, node: Int, layer: Int):
        self.layer_counts[layer][node] += 1

    def neighbor(
        ref self,
        node: Int,
        layer: Int,
        slot: Int,
        layer0_max_conn: Int,
        upper_max_conn: Int,
    ) -> UInt32:
        var start = self.row_start(node, layer, layer0_max_conn, upper_max_conn)
        return self.layer_neighbors[layer][start + slot]

    def set_neighbor(
        mut self,
        node: Int,
        layer: Int,
        slot: Int,
        layer0_max_conn: Int,
        upper_max_conn: Int,
        neighbor_id: UInt32,
    ):
        var start = self.row_start(node, layer, layer0_max_conn, upper_max_conn)
        self.layer_neighbors[layer][start + slot] = neighbor_id

    @always_inline
    @always_inline
    def neighbors_of(
        ref self,
        node: Int,
        layer: Int,
        layer0_max_conn: Int,
        upper_max_conn: Int,
    ) -> Span[UInt32, origin_of(self.layer_neighbors[layer])]:
        var max_conn = self.max_conn_for_layer(
            layer, layer0_max_conn, upper_max_conn
        )
        var start = node * max_conn
        var count_ptr = self.layer_counts[layer].unsafe_ptr()
        var count = Int(count_ptr[node])
        var neighbors_ptr = self.layer_neighbors[layer].unsafe_ptr()
        return Span(ptr=neighbors_ptr + start, length=count)

    def ensure_node_storage(
        mut self,
        node_id: UInt32,
        level: Int,
        layer0_max_conn: Int,
        upper_max_conn: Int,
    ):
        while len(self.layer_neighbors) <= level:
            self.layer_neighbors.append(List[UInt32]())
            self.layer_counts.append(List[UInt32]())
            self.row_locks.append(
                List[UnsafePointer[BlockingSpinLock, MutExternalOrigin]]()
            )

        for lc in range(level + 1):
            var max_conn = self.max_conn_for_layer(
                lc, layer0_max_conn, upper_max_conn
            )
            while len(self.layer_counts[lc]) < Int(node_id) + 1:
                self.layer_counts[lc].append(0)
                self.row_locks[lc].append(self._new_row_lock())
                for _ in range(max_conn):
                    self.layer_neighbors[lc].append(0)


struct HNSWIndex[dim: Int](Movable):
    var data: List[Float32]
    var codes: List[UInt8]
    var quantizer: SQ8Quantizer[Self.dim]
    var graph: HNSWGraphStorage
    var node_levels: List[Int]
    var ep: UInt32
    var max_level: Int
    var M: Int
    var M0: Int
    var ef_construction: Int
    var alpha: Float32
    var mL: Float32
    var num_elements: Int
    var use_f32_construction: Bool
    var metric: Metric
    var visit_marks: List[UInt32]
    var visit_generation: UInt32
    var build_profile_enabled: Bool
    var build_profile_inserts: Int
    var build_profile_total_ns: Int
    var build_profile_storage_ns: Int
    var build_profile_quantization_ns: Int
    var build_profile_query_prep_ns: Int
    var build_profile_connect_ns: Int
    var build_profile_upper_descent_ns: Int
    var build_profile_layer_search_ns: Int
    var build_profile_neighbor_select_ns: Int
    var build_profile_link_prune_ns: Int

    def __init__(
        out self,
        M: Int = 16,
        ef_construction: Int = 100,
        min_val: Float32 = 0.0,
        max_val: Float32 = 255.0,
        alpha: Float32 = 1.0,
        use_f32_construction: Bool = True,
        metric: Metric = Metric.L2,
    ):
        self.data = List[Float32]()
        self.codes = List[UInt8]()
        self.quantizer = SQ8Quantizer[Self.dim](min_val, max_val)
        self.graph = HNSWGraphStorage()
        self.node_levels = List[Int]()
        self.ep = 0
        self.max_level = 0
        self.M = M
        self.M0 = M * 2
        self.ef_construction = ef_construction
        self.alpha = alpha
        from std.math import log

        self.mL = Float32(1.0 / log(Float64(M)))
        self.num_elements = 0
        self.use_f32_construction = use_f32_construction
        self.metric = metric
        self.visit_marks = List[UInt32]()
        self.visit_generation = 0
        self.build_profile_enabled = False
        self.build_profile_inserts = 0
        self.build_profile_total_ns = 0
        self.build_profile_storage_ns = 0
        self.build_profile_quantization_ns = 0
        self.build_profile_query_prep_ns = 0
        self.build_profile_connect_ns = 0
        self.build_profile_upper_descent_ns = 0
        self.build_profile_layer_search_ns = 0
        self.build_profile_neighbor_select_ns = 0
        self.build_profile_link_prune_ns = 0

    def __init__(out self, *, deinit take: Self):
        self.data = take.data^
        self.codes = take.codes^
        self.quantizer = take.quantizer.copy()
        self.graph = take.graph^
        self.node_levels = take.node_levels^
        self.ep = take.ep
        self.max_level = take.max_level
        self.M = take.M
        self.M0 = take.M0
        self.ef_construction = take.ef_construction
        self.alpha = take.alpha
        self.mL = take.mL
        self.num_elements = take.num_elements
        self.use_f32_construction = take.use_f32_construction
        self.metric = take.metric
        self.visit_marks = take.visit_marks^
        self.visit_generation = take.visit_generation
        self.build_profile_enabled = take.build_profile_enabled
        self.build_profile_inserts = take.build_profile_inserts
        self.build_profile_total_ns = take.build_profile_total_ns
        self.build_profile_storage_ns = take.build_profile_storage_ns
        self.build_profile_quantization_ns = take.build_profile_quantization_ns
        self.build_profile_query_prep_ns = take.build_profile_query_prep_ns
        self.build_profile_connect_ns = take.build_profile_connect_ns
        self.build_profile_upper_descent_ns = (
            take.build_profile_upper_descent_ns
        )
        self.build_profile_layer_search_ns = take.build_profile_layer_search_ns
        self.build_profile_neighbor_select_ns = (
            take.build_profile_neighbor_select_ns
        )
        self.build_profile_link_prune_ns = take.build_profile_link_prune_ns

    def reserve(mut self, capacity: Int):
        self.data.reserve(capacity * Self.dim)
        self.codes.reserve(capacity * Self.dim)
        self.node_levels.reserve(capacity)
        self.visit_marks.reserve(capacity)
        self.graph.reserve_layers(8)

    def load_from_data(
        mut self,
        data: List[Float32],
        codes: List[UInt8],
        layer_neighbors: List[List[UInt32]],
        layer_counts: List[List[UInt32]],
        node_levels: List[Int],
        ep: UInt32,
        max_level: Int,
        num_elements: Int,
        use_f32_construction: Bool = True,
    ) raises:
        self.data = data.copy()
        self.codes = codes.copy()
        self.graph.load_from_flat(layer_neighbors, layer_counts)
        self.node_levels = node_levels.copy()
        self.ep = ep
        self.max_level = max_level
        self.num_elements = num_elements
        self.use_f32_construction = use_f32_construction
        self.visit_marks = List[UInt32]()
        self.visit_marks.reserve(num_elements)
        for _ in range(num_elements):
            self.visit_marks.append(0)
        self.visit_generation = 0
        self.build_profile_enabled = False
        self.build_profile_inserts = 0
        self.build_profile_total_ns = 0
        self.build_profile_storage_ns = 0
        self.build_profile_quantization_ns = 0
        self.build_profile_query_prep_ns = 0
        self.build_profile_connect_ns = 0
        self.build_profile_upper_descent_ns = 0
        self.build_profile_layer_search_ns = 0
        self.build_profile_neighbor_select_ns = 0
        self.build_profile_link_prune_ns = 0

    def reset_build_profile(mut self):
        self.build_profile_inserts = 0
        self.build_profile_total_ns = 0
        self.build_profile_storage_ns = 0
        self.build_profile_quantization_ns = 0
        self.build_profile_query_prep_ns = 0
        self.build_profile_connect_ns = 0
        self.build_profile_upper_descent_ns = 0
        self.build_profile_layer_search_ns = 0
        self.build_profile_neighbor_select_ns = 0
        self.build_profile_link_prune_ns = 0

    def enable_build_profile(mut self):
        self.reset_build_profile()
        self.build_profile_enabled = True

    def disable_build_profile(mut self):
        self.build_profile_enabled = False

    @always_inline
    def get_vector(ref self, id: Int) -> Span[Float32, origin_of(self.data)]:
        var start = id * Self.dim
        return Span(ptr=self.data.unsafe_ptr() + start, length=Self.dim)

    @always_inline
    def get_codes(ref self, id: Int) -> Span[UInt8, origin_of(self.codes)]:
        var start = id * Self.dim
        return Span(ptr=self.codes.unsafe_ptr() + start, length=Self.dim)

    @always_inline
    def _get_neighbors(
        ref self, node: Int, layer: Int
    ) -> Span[UInt32, origin_of(self.graph.layer_neighbors[layer])]:
        return self.graph.neighbors_of(node, layer, self.M0, self.M)

    def _add_neighbor(mut self, node: Int, layer: Int, neighbor_id: UInt32):
        var max_conn = self.M0 if layer == 0 else self.M
        var count = self.graph.count(node, layer)
        var scan_count = min(count, max_conn)
        # Simple check for duplicates
        for i in range(scan_count):
            if (
                self.graph.neighbor(node, layer, i, self.M0, self.M)
                == neighbor_id
            ):
                return

        if count >= max_conn:
            var neighbor_node = Int(neighbor_id)
            if (
                len(self.data) < (node + 1) * Self.dim
                or len(self.data) < (neighbor_node + 1) * Self.dim
            ):
                return

            var data_ptr = self.data.unsafe_ptr()
            var node_vec = Span[Float32, ImmutAnyOrigin](
                ptr=data_ptr + node * Self.dim, length=Self.dim
            )

            var scored = List[Candidate]()
            for i in range(max_conn):
                var nbr_id = self.graph.neighbor(
                    node, layer, i, self.M0, self.M
                )
                var nbr_node = Int(nbr_id)
                if len(self.data) < (nbr_node + 1) * Self.dim:
                    continue
                var nbr_vec = Span[Float32, ImmutAnyOrigin](
                    ptr=data_ptr + nbr_node * Self.dim, length=Self.dim
                )
                scored.append(
                    Candidate(nbr_id, _metric_distance[Self.dim](self.metric, node_vec, nbr_vec))
                )

            var new_vec = Span[Float32, ImmutAnyOrigin](
                ptr=data_ptr + neighbor_node * Self.dim, length=Self.dim
            )
            scored.append(
                Candidate(neighbor_id, _metric_distance[Self.dim](self.metric, node_vec, new_vec))
            )

            @parameter
            def cmp(a: Candidate, b: Candidate) -> Bool:
                return a.distance < b.distance

            var scored_span = Span[Candidate, MutAnyOrigin](
                ptr=scored.unsafe_ptr(), length=len(scored)
            )
            sort[cmp](scored_span)

            self.graph.set_count(node, layer, min(max_conn, len(scored)))
            for i in range(self.graph.count(node, layer)):
                self.graph.set_neighbor(
                    node, layer, i, self.M0, self.M, scored[i].id
                )
            return

        self.graph.set_neighbor(
            node, layer, count, self.M0, self.M, neighbor_id
        )
        self.graph.increment_count(node, layer)

    def _neighbor_count(ref self, node: Int, layer: Int) -> Int:
        return self.graph.count(node, layer)

    def _prune_neighbors(mut self, node: Int, layer: Int, max_conn: Int):
        var count = self.graph.count(node, layer)
        if count <= max_conn:
            return

        var nbr_ids = List[UInt32]()
        for i in range(count):
            nbr_ids.append(self.graph.neighbor(node, layer, i, self.M0, self.M))

        var data_ptr = self.data.unsafe_ptr()
        var node_vec = Span[Float32, ImmutAnyOrigin](
            ptr=data_ptr + node * Self.dim, length=Self.dim
        )

        var scored = List[Candidate]()
        for i in range(len(nbr_ids)):
            var nbr_id = Int(nbr_ids[i])
            var nbr_vec = Span[Float32, ImmutAnyOrigin](
                ptr=data_ptr + nbr_id * Self.dim, length=Self.dim
            )
            var dist = _metric_distance[Self.dim](self.metric, node_vec, nbr_vec)
            scored.append(Candidate(nbr_ids[i], dist))

        @parameter
        def cmp(a: Candidate, b: Candidate) -> Bool:
            return a.distance < b.distance

        var scored_span = Span[Candidate, MutAnyOrigin](
            ptr=scored.unsafe_ptr(), length=len(scored)
        )
        sort[cmp](scored_span)

        # Write sorted, truncated neighbors directly to storage.
        # Avoids calling _add_neighbor which would recompute all distances.
        var keep = min(max_conn, len(scored))
        self.graph.set_count(node, layer, keep)
        for i in range(keep):
            self.graph.set_neighbor(
                node, layer, i, self.M0, self.M, scored[i].id
            )

    def _random_level(mut self) -> Int:
        from std.math import log
        from std.random import random_float64

        var r = random_float64()
        if r == 0.0:
            r = 1e-10
        var level = Int(-log(r) * Float64(self.mL))
        if level < 0:
            level = 0
        return level

    def _ensure_layer_storage(mut self, node_id: UInt32, level: Int):
        self.graph.ensure_node_storage(node_id, level, self.M0, self.M)

    def _connect_existing_node(
        mut self, new_id: UInt32, vec: Span[Float32, _], level: Int
    ):
        if new_id == 0:
            self.ep = 0
            return

        from std.time import perf_counter_ns

        var connect_start = 0
        if self.build_profile_enabled:
            connect_start = Int(perf_counter_ns())

        var current_ep = self.ep

        if self.use_f32_construction:
            # Keep graph construction independent of traversal quantization.
            for lc in range(self.max_level, level, -1):
                var upper_start = 0
                if self.build_profile_enabled:
                    upper_start = Int(perf_counter_ns())
                current_ep = self._greedy_descent_f32(vec, current_ep, lc)
                if self.build_profile_enabled:
                    self.build_profile_upper_descent_ns += (
                        Int(perf_counter_ns()) - upper_start
                    )

            for lc in range(min(level, self.max_level), -1, -1):
                var max_conn = self.M0 if lc == 0 else self.M
                var search_start = 0
                if self.build_profile_enabled:
                    search_start = Int(perf_counter_ns())
                var results = self._search_layer_f32_scratch(
                    vec, [current_ep], ef=self.ef_construction, layer=lc
                )
                if self.build_profile_enabled:
                    self.build_profile_layer_search_ns += (
                        Int(perf_counter_ns()) - search_start
                    )

                var select_start = 0
                if self.build_profile_enabled:
                    select_start = Int(perf_counter_ns())
                var selected = self._select_neighbors(results^, max_conn)
                if self.build_profile_enabled:
                    self.build_profile_neighbor_select_ns += (
                        Int(perf_counter_ns()) - select_start
                    )

                var link_start = 0
                if self.build_profile_enabled:
                    link_start = Int(perf_counter_ns())
                for i in range(len(selected)):
                    var nbr_id = selected[i]
                    self._add_neighbor(Int(new_id), lc, nbr_id)
                    self._add_neighbor(Int(nbr_id), lc, new_id)
                    if self._neighbor_count(Int(nbr_id), lc) > max_conn:
                        self._prune_neighbors(Int(nbr_id), lc, max_conn)
                if self.build_profile_enabled:
                    self.build_profile_link_prune_ns += (
                        Int(perf_counter_ns()) - link_start
                    )

                if len(selected) > 0:
                    current_ep = selected[0]

            if level > self.max_level:
                self.max_level = level
                self.ep = new_id

            if self.build_profile_enabled:
                self.build_profile_connect_ns += (
                    Int(perf_counter_ns()) - connect_start
                )
            return

        var adt = FlashSQ8ADT[Self.dim](self.quantizer)
        var query_prep_start = 0
        if self.build_profile_enabled:
            query_prep_start = Int(perf_counter_ns())
        adt.build_for_query(vec, self.quantizer)
        if self.build_profile_enabled:
            self.build_profile_query_prep_ns += (
                Int(perf_counter_ns()) - query_prep_start
            )

        # 1. Search from max_level down to level + 1 to find entry point
        for lc in range(self.max_level, level, -1):
            var upper_start = 0
            if self.build_profile_enabled:
                upper_start = Int(perf_counter_ns())
            current_ep = self._greedy_descent_sq8(adt, current_ep, lc)
            if self.build_profile_enabled:
                self.build_profile_upper_descent_ns += (
                    Int(perf_counter_ns()) - upper_start
                )

        # 2. Insert into layers from min(level, max_level) down to 0
        for lc in range(min(level, self.max_level), -1, -1):
            var max_conn = self.M0 if lc == 0 else self.M
            var search_start = 0
            if self.build_profile_enabled:
                search_start = Int(perf_counter_ns())
            var results = self._search_layer_sq8_scratch(
                vec, adt, [current_ep], ef=self.ef_construction, layer=lc
            )
            if self.build_profile_enabled:
                self.build_profile_layer_search_ns += (
                    Int(perf_counter_ns()) - search_start
                )

            var select_start = 0
            if self.build_profile_enabled:
                select_start = Int(perf_counter_ns())
            var selected = self._select_neighbors(results^, max_conn)
            if self.build_profile_enabled:
                self.build_profile_neighbor_select_ns += (
                    Int(perf_counter_ns()) - select_start
                )

            var link_start = 0
            if self.build_profile_enabled:
                link_start = Int(perf_counter_ns())
            for i in range(len(selected)):
                var nbr_id = selected[i]
                self._add_neighbor(Int(new_id), lc, nbr_id)
                self._add_neighbor(Int(nbr_id), lc, new_id)
                if self._neighbor_count(Int(nbr_id), lc) > max_conn:
                    self._prune_neighbors(Int(nbr_id), lc, max_conn)
            if self.build_profile_enabled:
                self.build_profile_link_prune_ns += (
                    Int(perf_counter_ns()) - link_start
                )

            if len(selected) > 0:
                current_ep = selected[0]

        if level > self.max_level:
            self.max_level = level
            self.ep = new_id

        if self.build_profile_enabled:
            self.build_profile_connect_ns += (
                Int(perf_counter_ns()) - connect_start
            )

    def insert(mut self, vec: Span[Float32, _]):
        if self.build_profile_enabled:
            self._insert_profiled(vec)
            return

        var new_id = UInt32(self.num_elements)
        for i in range(Self.dim):
            self.data.append(vec[i])
        self.quantizer.quantize(vec, self.codes)
        self.num_elements += 1
        self.visit_marks.append(0)

        var level = self._random_level() if new_id > 0 else 0
        self.node_levels.append(level)
        self._ensure_layer_storage(new_id, level)
        self._connect_existing_node(new_id, vec, level)

    def insert_batch_serial(
        mut self,
        vectors: Span[Float32, _],
        count: Int,
        ef_multiplier: Int = 1,
    ):
        self.reserve(self.num_elements + count)
        var original_ef = self.ef_construction
        var multiplier = max(ef_multiplier, 1)
        self.ef_construction = original_ef * multiplier
        for row in range(count):
            var vec = Span(
                ptr=vectors.unsafe_ptr() + row * Self.dim, length=Self.dim
            )
            self.insert(vec)
        self.ef_construction = original_ef

    def _insert_profiled(mut self, vec: Span[Float32, _]):
        from std.time import perf_counter_ns

        var total_start = Int(perf_counter_ns())
        var new_id = UInt32(self.num_elements)

        var storage_start = Int(perf_counter_ns())
        for i in range(Self.dim):
            self.data.append(vec[i])
        self.build_profile_storage_ns += Int(perf_counter_ns()) - storage_start

        var quant_start = Int(perf_counter_ns())
        self.quantizer.quantize(vec, self.codes)
        self.build_profile_quantization_ns += (
            Int(perf_counter_ns()) - quant_start
        )

        storage_start = Int(perf_counter_ns())
        self.num_elements += 1
        self.visit_marks.append(0)

        var level = self._random_level() if new_id > 0 else 0
        self.node_levels.append(level)
        self._ensure_layer_storage(new_id, level)
        self.build_profile_storage_ns += Int(perf_counter_ns()) - storage_start

        self._connect_existing_node(new_id, vec, level)
        self.build_profile_inserts += 1
        self.build_profile_total_ns += Int(perf_counter_ns()) - total_start

    def _greedy_descent_sq8(
        ref self,
        ref adt: FlashSQ8ADT[Self.dim],
        entry_id: UInt32,
        layer: Int,
    ) -> UInt32:
        var current = entry_id
        var current_dist = adt.distance(self.get_codes(Int(current)))
        var improved = True

        while improved:
            improved = False
            var neighbors = self._get_neighbors(Int(current), layer)
            for i in range(len(neighbors)):
                var candidate = neighbors[i]
                var dist = adt.distance(self.get_codes(Int(candidate)))
                if dist < current_dist:
                    current_dist = dist
                    current = candidate
                    improved = True

        return current

    def _greedy_descent_f32(
        ref self,
        query: Span[Float32, _],
        entry_id: UInt32,
        layer: Int,
    ) -> UInt32:
        var current = entry_id
        var current_dist = _metric_distance[Self.dim](
            self.metric, query, self.get_vector(Int(current))
        )
        var improved = True

        while improved:
            improved = False
            var neighbors = self._get_neighbors(Int(current), layer)
            for i in range(len(neighbors)):
                var candidate = neighbors[i]
                var dist = _metric_distance[Self.dim](
                    self.metric, query, self.get_vector(Int(candidate))
                )
                if dist < current_dist:
                    current_dist = dist
                    current = candidate
                    improved = True

        return current

    def _search_layer_sq8(
        ref self,
        query: Span[Float32, _],
        ref adt: FlashSQ8ADT[Self.dim],
        entry_points: List[UInt32],
        ef: Int,
        layer: Int,
    ) -> List[Candidate]:
        var candidates = MinHeap()
        var found = MaxHeap()
        candidates.reserve(ef * 2)
        found.reserve(ef)
        var visited = Set[UInt32]()

        for i in range(len(entry_points)):
            var ep_id = entry_points[i]
            if ep_id in visited:
                continue
            visited.add(ep_id)
            var codes = self.get_codes(Int(ep_id))
            var dist = adt.distance(codes)
            candidates.push(Candidate(ep_id, dist))
            found.push(Candidate(ep_id, dist))

        while candidates.len() > 0:
            var best = candidates.peek()
            if best.distance > found.worst_distance():
                break
            _ = candidates.pop()

            var neighbors = self._get_neighbors(Int(best.id), layer)

            # VSAG Speculative Prefetch: prefetch the next candidate's neighbors
            if candidates.len() > 0:
                var next_best = candidates.peek()
                var _ = self._get_neighbors(Int(next_best.id), layer)
                # prefetch next neighbors ptr (stub)
                # from std.sys import prefetch
                # prefetch(next_neighbors.unsafe_ptr())

            for i in range(len(neighbors)):
                var n = neighbors[i]
                if n in visited:
                    continue
                visited.add(n)

                var codes = self.get_codes(Int(n))
                var dist = adt.distance(codes)

                if dist < found.worst_distance() or found.len() < ef:
                    candidates.push(Candidate(n, dist))
                    found.push_bounded(Candidate(n, dist), ef)

        var result = List[Candidate]()
        result.reserve(ef)
        while found.len() > 0:
            result.append(found.pop())
        return result^

    def _next_visit_generation(mut self) -> UInt32:
        self.visit_generation += 1
        if self.visit_generation == 0:
            for i in range(len(self.visit_marks)):
                self.visit_marks[i] = 0
            self.visit_generation = 1
        return self.visit_generation

    def _search_layer_sq8_scratch(
        mut self,
        query: Span[Float32, _],
        ref adt: FlashSQ8ADT[Self.dim],
        entry_points: List[UInt32],
        ef: Int,
        layer: Int,
    ) -> List[Candidate]:
        var candidates = MinHeap()
        var found = MaxHeap()
        candidates.reserve(ef * 2)
        found.reserve(ef)
        var generation = self._next_visit_generation()

        for i in range(len(entry_points)):
            var ep_id = entry_points[i]
            var ep_idx = Int(ep_id)
            if (
                ep_idx >= len(self.visit_marks)
                or self.visit_marks[ep_idx] == generation
            ):
                continue
            self.visit_marks[ep_idx] = generation
            var codes = self.get_codes(ep_idx)
            var dist = adt.distance(codes)
            candidates.push(Candidate(ep_id, dist))
            found.push(Candidate(ep_id, dist))

        while candidates.len() > 0:
            var best = candidates.peek()
            if best.distance > found.worst_distance():
                break
            _ = candidates.pop()

            var neighbors = self._get_neighbors(Int(best.id), layer)

            if candidates.len() > 0:
                var next_best = candidates.peek()
                var _ = self._get_neighbors(Int(next_best.id), layer)

            for i in range(len(neighbors)):
                var n = neighbors[i]
                var n_idx = Int(n)
                if (
                    n_idx >= len(self.visit_marks)
                    or self.visit_marks[n_idx] == generation
                ):
                    continue
                self.visit_marks[n_idx] = generation

                var codes = self.get_codes(n_idx)
                var dist = adt.distance(codes)

                if dist < found.worst_distance() or found.len() < ef:
                    candidates.push(Candidate(n, dist))
                    found.push_bounded(Candidate(n, dist), ef)

        var result = List[Candidate]()
        result.reserve(ef)
        while found.len() > 0:
            result.append(found.pop())
        return result^

    def _search_layer_f32_scratch(
        mut self,
        query: Span[Float32, _],
        entry_points: List[UInt32],
        ef: Int,
        layer: Int,
    ) -> List[Candidate]:
        var candidates = MinHeap()
        var found = MaxHeap()
        candidates.reserve(ef * 2)
        found.reserve(ef)
        var generation = self._next_visit_generation()

        for i in range(len(entry_points)):
            var ep_id = entry_points[i]
            var ep_idx = Int(ep_id)
            if (
                ep_idx >= len(self.visit_marks)
                or self.visit_marks[ep_idx] == generation
            ):
                continue
            self.visit_marks[ep_idx] = generation
            var dist = _metric_distance[Self.dim](self.metric, query, self.get_vector(ep_idx))
            candidates.push(Candidate(ep_id, dist))
            found.push(Candidate(ep_id, dist))

        while candidates.len() > 0:
            var best = candidates.peek()
            if best.distance > found.worst_distance():
                break
            _ = candidates.pop()

            var neighbors = self._get_neighbors(Int(best.id), layer)

            if candidates.len() > 0:
                var next_best = candidates.peek()
                var _ = self._get_neighbors(Int(next_best.id), layer)

            for i in range(len(neighbors)):
                var n = neighbors[i]
                var n_idx = Int(n)
                if (
                    n_idx >= len(self.visit_marks)
                    or self.visit_marks[n_idx] == generation
                ):
                    continue
                self.visit_marks[n_idx] = generation

                var dist = _metric_distance[Self.dim](self.metric, query, self.get_vector(n_idx))

                if dist < found.worst_distance() or found.len() < ef:
                    candidates.push(Candidate(n, dist))
                    found.push_bounded(Candidate(n, dist), ef)

        var result = List[Candidate]()
        result.reserve(ef)
        while found.len() > 0:
            result.append(found.pop())
        return result^

    def _search_layer_sq8_bitmap(
        ref self,
        query: Span[Float32, _],
        ref adt: FlashSQ8ADT[Self.dim],
        entry_points: List[UInt32],
        ef: Int,
        layer: Int,
    ) -> List[Candidate]:
        var candidates = MinHeap()
        var found = MaxHeap()
        candidates.reserve(ef * 2)
        found.reserve(ef)
        var visited = List[UInt8]()
        visited.reserve(self.num_elements)
        for _ in range(self.num_elements):
            visited.append(0)

        for i in range(len(entry_points)):
            var ep_id = entry_points[i]
            var ep_idx = Int(ep_id)
            if ep_idx >= len(visited) or visited[ep_idx] != 0:
                continue
            visited[ep_idx] = 1
            var codes = self.get_codes(ep_idx)
            var dist = adt.distance(codes)
            candidates.push(Candidate(ep_id, dist))
            found.push(Candidate(ep_id, dist))

        while candidates.len() > 0:
            var best = candidates.peek()
            if best.distance > found.worst_distance():
                break
            _ = candidates.pop()

            var neighbors = self._get_neighbors(Int(best.id), layer)

            if candidates.len() > 0:
                var next_best = candidates.peek()
                var _ = self._get_neighbors(Int(next_best.id), layer)

            for i in range(len(neighbors)):
                var n = neighbors[i]
                var n_idx = Int(n)
                if n_idx >= len(visited) or visited[n_idx] != 0:
                    continue
                visited[n_idx] = 1

                var codes = self.get_codes(n_idx)
                var dist = adt.distance(codes)

                if dist < found.worst_distance() or found.len() < ef:
                    candidates.push(Candidate(n, dist))
                    found.push_bounded(Candidate(n, dist), ef)

        var result = List[Candidate]()
        result.reserve(ef)
        while found.len() > 0:
            result.append(found.pop())
        return result^

    def _search_layer_sq8_external_scratch(
        ref self,
        query: Span[Float32, _],
        mut scratch: HNSWSearchScratch[Self.dim],
        entry_points: List[UInt32],
        ef: Int,
        layer: Int,
    ) -> List[Candidate]:
        var generation = scratch.begin_layer(self.num_elements, ef)
        var n_visit = len(scratch.visit_marks)

        for i in range(len(entry_points)):
            var ep_id = entry_points[i]
            var ep_idx = Int(ep_id)
            if ep_idx >= n_visit or scratch.visit_marks[ep_idx] == generation:
                continue
            scratch.visit_marks[ep_idx] = generation
            var codes = self.get_codes(ep_idx)
            var dist = scratch.adt.distance(codes)
            scratch.candidates.push(Candidate(ep_id, dist))
            scratch.found.push(Candidate(ep_id, dist))

        while scratch.candidates.len() > 0:
            var best = scratch.candidates.peek()
            if best.distance > scratch.found.worst_distance():
                break
            _ = scratch.candidates.pop()

            var neighbors = self._get_neighbors(Int(best.id), layer)
            var n_neighbors = len(neighbors)
            var visit_ptr = scratch.visit_marks.unsafe_ptr()

            for i in range(n_neighbors):
                var n = neighbors[i]
                var n_idx = Int(n)
                if n_idx >= n_visit or visit_ptr[n_idx] == generation:
                    continue
                visit_ptr[n_idx] = generation

                var codes = self.get_codes(n_idx)
                var dist = scratch.adt.distance(codes)

                if (
                    dist < scratch.found.worst_distance()
                    or scratch.found.len() < ef
                ):
                    scratch.candidates.push(Candidate(n, dist))
                    scratch.found.push_bounded(Candidate(n, dist), ef)

        var result = List[Candidate]()
        result.reserve(ef)
        while scratch.found.len() > 0:
            result.append(scratch.found.pop())
        return result^

    def _search_layer_sq8_external_scratch_filtered(
        ref self,
        query: Span[Float32, _],
        mut scratch: HNSWSearchScratch[Self.dim],
        entry_points: List[UInt32],
        allowed: List[UInt8],
        ef: Int,
        layer: Int,
    ) -> List[Candidate]:
        # ACORN-1 filtered search with 2-hop expansion (Weaviate optimization).
        # When a neighbor fails the filter, expand to its neighbors (2-hop)
        # to maintain graph connectivity through filtered-out bridge nodes.
        var generation = scratch.begin_layer(self.num_elements, ef)
        var filtered = MaxHeap()
        filtered.reserve(ef)

        for i in range(len(entry_points)):
            var ep_id = entry_points[i]
            var ep_idx = Int(ep_id)
            if (
                ep_idx >= len(scratch.visit_marks)
                or scratch.visit_marks[ep_idx] == generation
            ):
                continue
            scratch.visit_marks[ep_idx] = generation
            var codes = self.get_codes(ep_idx)
            var dist = scratch.adt.distance(codes)
            scratch.candidates.push(Candidate(ep_id, dist))
            scratch.found.push(Candidate(ep_id, dist))
            if ep_idx < len(allowed) and allowed[ep_idx] != 0:
                filtered.push(Candidate(ep_id, dist))

        while scratch.candidates.len() > 0:
            var best = scratch.candidates.peek()
            if best.distance > scratch.found.worst_distance():
                break
            _ = scratch.candidates.pop()

            var neighbors = self._get_neighbors(Int(best.id), layer)

            if scratch.candidates.len() > 0:
                var next_best = scratch.candidates.peek()
                var _ = self._get_neighbors(Int(next_best.id), layer)

            for i in range(len(neighbors)):
                var n = neighbors[i]
                var n_idx = Int(n)
                if (
                    n_idx >= len(scratch.visit_marks)
                    or scratch.visit_marks[n_idx] == generation
                ):
                    continue
                scratch.visit_marks[n_idx] = generation

                var codes = self.get_codes(n_idx)
                var dist = scratch.adt.distance(codes)

                if (
                    scratch.found.len() < ef
                    or dist < scratch.found.worst_distance()
                ):
                    scratch.candidates.push(Candidate(n, dist))
                    scratch.found.push_bounded(Candidate(n, dist), ef)
                    if n_idx < len(allowed) and allowed[n_idx] != 0:
                        filtered.push_bounded(Candidate(n, dist), ef)
                    else:
                        # ACORN 2-hop: neighbor failed filter, expand to its neighbors
                        var second_hop = self._get_neighbors(n_idx, layer)
                        for j in range(len(second_hop)):
                            var sh = second_hop[j]
                            var sh_idx = Int(sh)
                            if (
                                sh_idx >= len(scratch.visit_marks)
                                or scratch.visit_marks[sh_idx] == generation
                            ):
                                continue
                            scratch.visit_marks[sh_idx] = generation
                            var sh_codes = self.get_codes(sh_idx)
                            var sh_dist = scratch.adt.distance(sh_codes)
                            if (
                                scratch.found.len() < ef
                                or sh_dist < scratch.found.worst_distance()
                            ):
                                scratch.candidates.push(Candidate(sh, sh_dist))
                                scratch.found.push_bounded(Candidate(sh, sh_dist), ef)
                                if sh_idx < len(allowed) and allowed[sh_idx] != 0:
                                    filtered.push_bounded(Candidate(sh, sh_dist), ef)

        var result = List[Candidate]()
        result.reserve(ef)
        while filtered.len() > 0:
            result.append(filtered.pop())
        return result^

    def _search_layer_f32_bitmap(
        ref self,
        query: Span[Float32, _],
        entry_points: List[UInt32],
        ef: Int,
        layer: Int,
    ) -> List[Candidate]:
        var candidates = MinHeap()
        var found = MaxHeap()
        candidates.reserve(ef * 2)
        found.reserve(ef)
        var visited = List[UInt8]()
        visited.reserve(self.num_elements)
        for _ in range(self.num_elements):
            visited.append(0)

        for i in range(len(entry_points)):
            var ep_id = entry_points[i]
            var ep_idx = Int(ep_id)
            if ep_idx >= len(visited) or visited[ep_idx] != 0:
                continue
            visited[ep_idx] = 1
            var vec = self.get_vector(ep_idx)
            var dist = _metric_distance[Self.dim](self.metric, query, vec)
            candidates.push(Candidate(ep_id, dist))
            found.push(Candidate(ep_id, dist))

        while candidates.len() > 0:
            var best = candidates.peek()
            if best.distance > found.worst_distance():
                break
            _ = candidates.pop()

            var neighbors = self._get_neighbors(Int(best.id), layer)

            if candidates.len() > 0:
                var next_best = candidates.peek()
                var _ = self._get_neighbors(Int(next_best.id), layer)

            for i in range(len(neighbors)):
                var n = neighbors[i]
                var n_idx = Int(n)
                if n_idx >= len(visited) or visited[n_idx] != 0:
                    continue
                visited[n_idx] = 1

                var vec = self.get_vector(n_idx)
                var dist = _metric_distance[Self.dim](self.metric, query, vec)

                if dist < found.worst_distance() or found.len() < ef:
                    candidates.push(Candidate(n, dist))
                    found.push_bounded(Candidate(n, dist), ef)

        var result = List[Candidate]()
        result.reserve(ef)
        while found.len() > 0:
            result.append(found.pop())
        return result^

    def _select_neighbors(
        mut self, var candidates: List[Candidate], M: Int
    ) -> List[UInt32]:
        # Simple heuristic neighbor selection (Algorithm 4 from HNSW paper)
        # 1. Sort candidates by distance (closest first)
        @parameter
        def cmp(a: Candidate, b: Candidate) -> Bool:
            return a.distance < b.distance

        var span = Span[Candidate, MutAnyOrigin](
            ptr=candidates.unsafe_ptr(), length=len(candidates)
        )
        sort[cmp](span)

        var result = List[Candidate]()
        var data_ptr = self.data.unsafe_ptr()

        for i in range(len(candidates)):
            if len(result) >= M:
                break

            var c = candidates[i]
            var is_good = True

            # Check if c is closer to query than to any already selected neighbor
            var c_vec = Span[Float32, ImmutAnyOrigin](
                ptr=data_ptr + Int(c.id) * Self.dim, length=Self.dim
            )

            for j in range(len(result)):
                var r = result[j]
                var r_vec = Span[Float32, ImmutAnyOrigin](
                    ptr=data_ptr + Int(r.id) * Self.dim, length=Self.dim
                )
                var dist_c_r = _metric_distance[Self.dim](self.metric, c_vec, r_vec)
                if dist_c_r * self.alpha < c.distance:
                    is_good = False
                    break

            if is_good:
                result.append(c)

        var final_ids = List[UInt32]()
        for i in range(len(result)):
            final_ids.append(result[i].id)

        return final_ids^

    def search(
        ref self, query: Span[Float32, _], k: Int, ef_search: Int = 100
    ) -> List[Candidate]:
        var scratch = HNSWSearchScratch[Self.dim](self.quantizer)
        return self.search_with_scratch(query, scratch, k, ef_search)

    def search_with_scratch(
        ref self,
        query: Span[Float32, _],
        mut scratch: HNSWSearchScratch[Self.dim],
        k: Int,
        ef_search: Int = 100,
    ) -> List[Candidate]:
        if self.num_elements == 0:
            var empty = List[Candidate]()
            return empty^

        scratch.prepare_query(query, self.quantizer, self.num_elements)

        var current_ep = self.ep

        for lc in range(self.max_level, 0, -1):
            var results = self._search_layer_sq8_external_scratch(
                query, scratch, [current_ep], ef=1, layer=lc
            )
            if len(results) > 0:
                current_ep = results[0].id

        var results = self._search_layer_sq8_external_scratch(
            query, scratch, [current_ep], ef=ef_search, layer=0
        )

        var n_results = len(results)
        scratch.reranked.clear()
        scratch.reranked.reserve(n_results)
        var query_ptr = query.unsafe_ptr()
        var data_ptr = self.data.unsafe_ptr()
        var rerank_ptr = scratch.reranked.unsafe_ptr()
        for i in range(n_results):
            var c = results[i]
            var vec_ptr = data_ptr + Int(c.id) * Self.dim
            # Inline SIMD L2 distance
            var acc0 = SIMD[DType.float32, 8](0.0)
            var acc1 = SIMD[DType.float32, 8](0.0)
            var d: Int = 0
            while d + 16 <= Self.dim:
                var da0 = vec_ptr.load[width=8](d) - query_ptr.load[width=8](d)
                acc0 += da0 * da0
                var da1 = vec_ptr.load[width=8](d + 8) - query_ptr.load[
                    width=8
                ](d + 8)
                acc1 += da1 * da1
                d += 16
            var dist: Float32 = (acc0 + acc1).reduce_add()
            for j in range(d, Self.dim):
                var diff = vec_ptr.load[1](j)[0] - query_ptr.load[1](j)[0]
                dist += diff * diff
            scratch.reranked.append(Candidate(c.id, dist))

        @parameter
        def cmp(a: Candidate, b: Candidate) -> Bool:
            return a.distance < b.distance

        var span = Span[Candidate, MutAnyOrigin](
            ptr=scratch.reranked.unsafe_ptr(), length=len(scratch.reranked)
        )
        sort[cmp](span)

        var top_k = List[Candidate]()
        top_k.reserve(min(k, len(scratch.reranked)))
        for i in range(min(k, len(scratch.reranked))):
            top_k.append(scratch.reranked[i])
        return top_k^

    def search_filtered(
        ref self,
        query: Span[Float32, _],
        allowed: List[UInt8],
        k: Int,
        ef_search: Int = 100,
    ) raises -> List[Candidate]:
        var scratch = HNSWSearchScratch[Self.dim](self.quantizer)
        return self.search_filtered_with_scratch(
            query, allowed, scratch, k, ef_search
        )

    def _count_eligible(ref self, allowed: List[UInt8]) -> Int:
        """Count non-zero entries in the allow-list bitmap."""
        var count = 0
        for i in range(self.num_elements):
            if allowed[i] != 0:
                count += 1
        return count

    def search_exact_filtered(
        ref self,
        query: Span[Float32, _],
        allowed: List[UInt8],
        k: Int,
    ) raises -> List[Candidate]:
        """Exact brute-force search over eligible items using F32 distances.

        Linear scan of all items, computing L2 distance for eligible ones.
        Returns guaranteed-correct top-k.
        """
        if self.num_elements == 0:
            var empty = List[Candidate]()
            return empty^
        if len(allowed) < self.num_elements:
            raise Error("filtered search allow-list is shorter than index")

        var found = MaxHeap()
        found.reserve(k + 1)

        for i in range(self.num_elements):
            if allowed[i] != 0:
                var vec = self.get_vector(i)
                var dist = _metric_distance[Self.dim](self.metric, query, vec)
                found.push_bounded(Candidate(UInt32(i), dist), k)

        var result = List[Candidate]()
        result.reserve(found.len())
        while found.len() > 0:
            result.append(found.pop())
        return result^

    def search_filtered_with_scratch(
        ref self,
        query: Span[Float32, _],
        allowed: List[UInt8],
        mut scratch: HNSWSearchScratch[Self.dim],
        k: Int,
        ef_search: Int = 100,
    ) raises -> List[Candidate]:
        if self.num_elements == 0:
            var empty = List[Candidate]()
            return empty^
        if len(allowed) < self.num_elements:
            raise Error("filtered search allow-list is shorter than index")

        # Route based on selectivity: exact for restrictive predicates,
        # HNSW for broad predicates where graph traversal stays accurate.
        var eligible = self._count_eligible(allowed)
        if eligible <= k:
            return self.search_exact_filtered(query, allowed, k)
        var selectivity = Float64(eligible) / Float64(self.num_elements)
        if selectivity < 0.2:
            return self.search_exact_filtered(query, allowed, k)

        scratch.prepare_query(query, self.quantizer, self.num_elements)

        var current_ep = self.ep

        for lc in range(self.max_level, 0, -1):
            var results = self._search_layer_sq8_external_scratch(
                query, scratch, [current_ep], ef=1, layer=lc
            )
            if len(results) > 0:
                current_ep = results[0].id

        var results = self._search_layer_sq8_external_scratch_filtered(
            query, scratch, [current_ep], allowed, ef=ef_search, layer=0
        )

        scratch.reranked.clear()
        scratch.reranked.reserve(len(results))
        for i in range(len(results)):
            var c = results[i]
            var vec = self.get_vector(Int(c.id))
            var dist = _metric_distance[Self.dim](self.metric, query, vec)
            scratch.reranked.append(Candidate(c.id, dist))

        @parameter
        def cmp(a: Candidate, b: Candidate) -> Bool:
            return a.distance < b.distance

        var span = Span[Candidate, MutAnyOrigin](
            ptr=scratch.reranked.unsafe_ptr(), length=len(scratch.reranked)
        )
        sort[cmp](span)

        var top_k = List[Candidate]()
        top_k.reserve(min(k, len(scratch.reranked)))
        for i in range(min(k, len(scratch.reranked))):
            top_k.append(scratch.reranked[i])
        return top_k^

    def search_f32(
        ref self, query: Span[Float32, _], k: Int, ef_search: Int = 100
    ) -> List[Candidate]:
        if self.num_elements == 0:
            var empty = List[Candidate]()
            return empty^

        var current_ep = self.ep

        for lc in range(self.max_level, 0, -1):
            var results = self._search_layer_f32_bitmap(
                query, [current_ep], ef=1, layer=lc
            )
            if len(results) > 0:
                current_ep = results[0].id

        var results = self._search_layer_f32_bitmap(
            query, [current_ep], ef=ef_search, layer=0
        )

        @parameter
        def cmp(a: Candidate, b: Candidate) -> Bool:
            return a.distance < b.distance

        var span = Span[Candidate, MutAnyOrigin](
            ptr=results.unsafe_ptr(), length=len(results)
        )
        sort[cmp](span)

        var top_k = List[Candidate]()
        top_k.reserve(min(k, len(results)))
        for i in range(min(k, len(results))):
            top_k.append(results[i])
        return top_k^
