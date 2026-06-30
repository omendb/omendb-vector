from std.collections import Dict, List, Set
from std.utils import Variant
from std.math import min
from hnsw import HNSWIndex, Candidate
from distance import l2_distance, dot_product, cosine_distance
from store_types import Metric

# PropertyValue can hold common types for graph properties
comptime PropertyValue = Variant[String, Int, Float64, Bool]


@fieldwise_init
struct GraphDirection(Equatable, ImplicitlyCopyable):
    var _value: Int

    comptime OUTGOING = GraphDirection(_value=0)
    comptime INCOMING = GraphDirection(_value=1)
    comptime BOTH = GraphDirection(_value=2)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)


@fieldwise_init
struct Node(Copyable, Movable):
    var id: UInt64
    var label: String
    var properties: Dict[String, PropertyValue]
    var vector_id: Optional[UInt64]

    def __init__(
        out self, id: UInt64, label: String, vector_id: Optional[UInt64] = None
    ):
        self.id = id
        self.label = label
        self.properties = Dict[String, PropertyValue]()
        self.vector_id = vector_id

    def __init__(out self, *, copy: Self):
        self.id = copy.id
        self.label = copy.label
        self.properties = copy.properties.copy()
        self.vector_id = copy.vector_id

    def __init__(out self, *, deinit take: Self):
        self.id = take.id
        self.label = take.label^
        self.properties = take.properties^
        self.vector_id = take.vector_id


@fieldwise_init
struct Edge(Copyable, Movable):
    var id: UInt64
    var src: UInt64
    var dst: UInt64
    var edge_type: String
    var properties: Dict[String, PropertyValue]
    var weight: Optional[Float32]

    def __init__(
        out self,
        id: UInt64,
        src: UInt64,
        dst: UInt64,
        edge_type: String,
        weight: Optional[Float32] = None,
    ):
        self.id = id
        self.src = src
        self.dst = dst
        self.edge_type = edge_type
        self.properties = Dict[String, PropertyValue]()
        self.weight = weight

    def __init__(out self, *, copy: Self):
        self.id = copy.id
        self.src = copy.src
        self.dst = copy.dst
        self.edge_type = copy.edge_type
        self.properties = copy.properties.copy()
        self.weight = copy.weight

    def __init__(out self, *, deinit take: Self):
        self.id = take.id
        self.src = take.src
        self.dst = take.dst
        self.edge_type = take.edge_type^
        self.properties = take.properties^
        self.weight = take.weight


struct GraphIndex(Movable):
    """CSR Adjacency storage for fast graph traversal."""

    var out_offsets: List[UInt64]
    var out_neighbors: List[UInt64]
    var out_edge_ids: List[UInt64]

    var in_offsets: List[UInt64]
    var in_neighbors: List[UInt64]
    var in_edge_ids: List[UInt64]

    def __init__(out self):
        self.out_offsets = List[UInt64]()
        self.out_offsets.append(0)
        self.out_neighbors = List[UInt64]()
        self.out_edge_ids = List[UInt64]()

        self.in_offsets = List[UInt64]()
        self.in_offsets.append(0)
        self.in_neighbors = List[UInt64]()
        self.in_edge_ids = List[UInt64]()

    def __init__(out self, *, deinit take: Self):
        self.out_offsets = take.out_offsets^
        self.out_neighbors = take.out_neighbors^
        self.out_edge_ids = take.out_edge_ids^
        self.in_offsets = take.in_offsets^
        self.in_neighbors = take.in_neighbors^
        self.in_edge_ids = take.in_edge_ids^

    def out_neighbors_of(
        ref self, node_id: UInt64
    ) -> Span[UInt64, origin_of(self.out_neighbors)]:
        var idx = Int(node_id)
        if idx >= len(self.out_offsets) - 1:
            return Span[UInt64, origin_of(self.out_neighbors)]()
        var start = Int(self.out_offsets[idx])
        var end = Int(self.out_offsets[idx + 1])
        return Span(
            ptr=self.out_neighbors.unsafe_ptr() + start, length=end - start
        )

    def in_neighbors_of(
        ref self, node_id: UInt64
    ) -> Span[UInt64, origin_of(self.in_neighbors)]:
        var idx = Int(node_id)
        if idx >= len(self.in_offsets) - 1:
            return Span[UInt64, origin_of(self.in_neighbors)]()
        var start = Int(self.in_offsets[idx])
        var end = Int(self.in_offsets[idx + 1])
        return Span(
            ptr=self.in_neighbors.unsafe_ptr() + start, length=end - start
        )


from std.python import Python, PythonObject


struct NodeStore(Movable):
    var db: PythonObject
    var json: PythonObject

    def __init__(out self, path: String) raises:
        var sqlite3 = Python.import_module("sqlite3")
        self.json = Python.import_module("json")
        self.db = sqlite3.connect(path)
        self._init_db()

    def __init__(out self, *, deinit take: Self):
        self.db = take.db
        self.json = take.json

    def _init_db(mut self) raises:
        var cursor = self.db.cursor()
        _ = cursor.execute(
            "CREATE TABLE IF NOT EXISTS nodes (id INTEGER PRIMARY KEY, label"
            " TEXT, vector_id INTEGER, properties TEXT)"
        )
        _ = cursor.execute(
            "CREATE TABLE IF NOT EXISTS edges (id INTEGER PRIMARY KEY, src"
            " INTEGER, dst INTEGER, edge_type TEXT, weight REAL, properties"
            " TEXT)"
        )
        _ = self.db.commit()

    def save_node(mut self, node: Node) raises:
        var cursor = self.db.cursor()
        # Convert properties Dict to JSON string
        var props_py = Python.dict()
        for item in node.properties.items():
            var key = item.key
            var val = item.value
            if val.isa[String]():
                props_py[key] = val.unsafe_get[String]()
            elif val.isa[Int]():
                props_py[key] = val.unsafe_get[Int]()
            elif val.isa[Float64]():
                props_py[key] = val.unsafe_get[Float64]()
            elif val.isa[Bool]():
                props_py[key] = val.unsafe_get[Bool]()

        var props_json = self.json.dumps(props_py)
        var vid = Int(-1)
        if node.vector_id:
            vid = Int(node.vector_id.value())

        var params = Python.list()
        params.append(Int(node.id))
        params.append(node.label)
        params.append(vid)
        params.append(props_json)

        var builtins = Python.import_module("builtins")
        _ = cursor.execute(
            (
                "INSERT OR REPLACE INTO nodes (id, label, vector_id,"
                " properties) VALUES (?, ?, ?, ?)"
            ),
            builtins.tuple(params),
        )
        _ = self.db.commit()

    def load_nodes(mut self) raises -> List[Node]:
        var cursor = self.db.cursor()
        var rows = cursor.execute(
            "SELECT id, label, vector_id, properties FROM nodes"
        ).fetchall()
        var result = List[Node]()
        var builtins = Python.import_module("builtins")

        for i in range(Int(py=builtins.len(rows))):
            var row = rows[i]
            var id = UInt64(Int(py=row[0]))
            var label = String(row[1])
            var vid_raw = Int(py=row[2])
            var vid = Optional[UInt64](None)
            if vid_raw != -1:
                vid = Optional[UInt64](UInt64(vid_raw))

            var node = Node(id, label, vid)

            var props_json = String(row[3])
            if props_json.byte_length() > 2:  # not empty {}
                var props_py = self.json.loads(props_json)
                var items_list = builtins.list(props_py.items())
                # items_list is a list of tuples
                for j in range(Int(py=builtins.len(items_list))):
                    var item = items_list[j]
                    var key = String(item[0])
                    var val = item[1]
                    # This is tricky: we need to guess the type for Variant
                    if builtins.isinstance(val, builtins.str):
                        node.properties[key] = PropertyValue(String(val))
                    elif builtins.isinstance(
                        val, builtins.bool
                    ):  # check bool before int (bool is int in python)
                        node.properties[key] = PropertyValue(Bool(val))
                    elif builtins.isinstance(val, builtins.int):
                        node.properties[key] = PropertyValue(Int(py=val))
                    elif builtins.isinstance(val, builtins.float):
                        node.properties[key] = PropertyValue(Float64(py=val))

            result.append(node^)
        return result^

    def close(mut self):
        _ = self.db.close()


struct PropertyGraph(Movable):
    var nodes: List[Node]
    var edges: List[Edge]
    var index: GraphIndex
    var external_to_node: Dict[String, UInt64]
    var node_external_ids: List[String]
    var vector_to_node: Dict[UInt64, UInt64]
    # Delta buffers for incremental updates before re-building CSR
    var _pending_edges: List[Edge]
    var store: Optional[NodeStore]

    def __init__(out self):
        self.nodes = List[Node]()
        self.edges = List[Edge]()
        self.index = GraphIndex()
        self.external_to_node = Dict[String, UInt64]()
        self.node_external_ids = List[String]()
        self.vector_to_node = Dict[UInt64, UInt64]()
        self._pending_edges = List[Edge]()
        self.store = None

    def __init__(out self, *, deinit take: Self):
        self.nodes = take.nodes^
        self.edges = take.edges^
        self.index = take.index^
        self.external_to_node = take.external_to_node^
        self.node_external_ids = take.node_external_ids^
        self.vector_to_node = take.vector_to_node^
        self._pending_edges = take._pending_edges^
        self.store = take.store^

    def set_store(mut self, path: String) raises:
        self.store = Optional(NodeStore(path))

    def add_node(
        mut self, label: String, vector_id: Optional[UInt64] = None
    ) -> UInt64:
        var id = UInt64(len(self.nodes))
        var node = Node(id, label, vector_id)

        # If store is active, persist immediately
        if self.store:
            try:
                self.store.value().save_node(node)
            except:
                pass  # Log error in real implementation

        self.nodes.append(node^)
        self.node_external_ids.append("")
        if vector_id:
            self.vector_to_node[vector_id.value()] = id
        # Ensure CSR offsets are ready for this node
        while len(self.index.out_offsets) <= Int(id) + 1:
            self.index.out_offsets.append(
                self.index.out_offsets[len(self.index.out_offsets) - 1]
            )
        while len(self.index.in_offsets) <= Int(id) + 1:
            self.index.in_offsets.append(
                self.index.in_offsets[len(self.index.in_offsets) - 1]
            )
        return id

    def add_node_with_id(
        mut self,
        external_id: String,
        label: String,
        vector_id: Optional[UInt64] = None,
    ) raises -> UInt64:
        if external_id in self.external_to_node:
            return self.external_to_node[external_id]

        var node_id = self.add_node(label, vector_id)
        self.external_to_node[external_id] = node_id
        self.node_external_ids[Int(node_id)] = external_id
        return node_id

    def node_id_for(ref self, external_id: String) raises -> Optional[UInt64]:
        if external_id in self.external_to_node:
            return Optional[UInt64](self.external_to_node[external_id])
        return Optional[UInt64](None)

    def node_id_for_vector(
        ref self, vector_id: UInt64
    ) raises -> Optional[UInt64]:
        if vector_id in self.vector_to_node:
            return Optional[UInt64](self.vector_to_node[vector_id])
        return Optional[UInt64](None)

    def remove_node_by_external_id(
        mut self, external_id: String
    ) raises -> Bool:
        if external_id not in self.external_to_node:
            return False

        var node_id = self.external_to_node.pop(external_id)
        var idx = Int(node_id)
        if idx >= len(self.nodes):
            raise Error("graph external node mapping out of range")

        self.node_external_ids[idx] = ""
        if self.nodes[idx].vector_id:
            var vector_id = self.nodes[idx].vector_id.value()
            if vector_id in self.vector_to_node:
                _ = self.vector_to_node.pop(vector_id)
            self.nodes[idx].vector_id = Optional[UInt64](None)

        var removed_edge = False
        var i = 0
        while i < len(self.edges):
            if self.edges[i].src == node_id or self.edges[i].dst == node_id:
                _ = self.edges.pop(i)
                removed_edge = True
            else:
                i += 1

        i = 0
        while i < len(self._pending_edges):
            if (
                self._pending_edges[i].src == node_id
                or self._pending_edges[i].dst == node_id
            ):
                _ = self._pending_edges.pop(i)
                removed_edge = True
            else:
                i += 1

        if removed_edge:
            for j in range(len(self.edges)):
                self.edges[j].id = UInt64(j)
            for j in range(len(self._pending_edges)):
                self._pending_edges[j].id = UInt64(len(self.edges) + j)
            self._rebuild_index()

        return True

    def external_id_for(ref self, node_id: UInt64) -> Optional[String]:
        var idx = Int(node_id)
        if idx >= len(self.node_external_ids):
            return Optional[String](None)

        var external_id = self.node_external_ids[idx]
        if external_id.byte_length() == 0:
            return Optional[String](None)

        return Optional[String](external_id)

    def add_edge(
        mut self,
        src: UInt64,
        dst: UInt64,
        edge_type: String,
        weight: Optional[Float32] = None,
    ) raises -> UInt64:
        self._validate_edge(src, dst, edge_type)
        var existing = self.edge_id_for(src, dst, edge_type)
        if existing:
            return existing.value()

        var id = UInt64(len(self.edges) + len(self._pending_edges))
        var e = Edge(id, src, dst, edge_type, weight)
        self._pending_edges.append(e^)
        # For Phase 2, we rebuild CSR immediately for simplicity.
        # In production, this would be batched or use a dynamic adjacency list.
        self._flush_deltas()
        return id

    def edge_id_for(
        ref self, src: UInt64, dst: UInt64, edge_type: String
    ) raises -> Optional[UInt64]:
        self._validate_edge(src, dst, edge_type)
        for i in range(len(self.edges)):
            ref e = self.edges[i]
            if e.src == src and e.dst == dst and e.edge_type == edge_type:
                return Optional[UInt64](e.id)
        for i in range(len(self._pending_edges)):
            ref e = self._pending_edges[i]
            if e.src == src and e.dst == dst and e.edge_type == edge_type:
                return Optional[UInt64](e.id)
        return Optional[UInt64](None)

    def remove_edge(
        mut self, src: UInt64, dst: UInt64, edge_type: String
    ) raises -> Bool:
        self._validate_edge(src, dst, edge_type)
        for i in range(len(self.edges)):
            ref e = self.edges[i]
            if e.src == src and e.dst == dst and e.edge_type == edge_type:
                _ = self.edges.pop(i)
                for j in range(len(self.edges)):
                    self.edges[j].id = UInt64(j)
                self._rebuild_index()
                return True
        return False

    def _validate_edge(
        ref self, src: UInt64, dst: UInt64, edge_type: String
    ) raises:
        if Int(src) >= len(self.nodes):
            raise Error("edge source node does not exist")
        if Int(dst) >= len(self.nodes):
            raise Error("edge destination node does not exist")
        if src == dst:
            raise Error("self edges are not supported")
        if edge_type.byte_length() == 0:
            raise Error("edge_type must not be empty")

    def _validate_node(ref self, node_id: UInt64) raises:
        if Int(node_id) >= len(self.nodes):
            raise Error("node does not exist")

    def _edge_type_matches(
        ref self, edge_id: UInt64, edge_type: Optional[String]
    ) -> Bool:
        if not edge_type:
            return True
        return self.edges[Int(edge_id)].edge_type == edge_type.value()

    def neighbors(
        ref self,
        node_id: UInt64,
        direction: GraphDirection = GraphDirection.OUTGOING,
        edge_type: Optional[String] = None,
    ) raises -> List[UInt64]:
        self._validate_node(node_id)
        var result = List[UInt64]()
        var seen = Set[UInt64]()
        var idx = Int(node_id)

        if (
            direction == GraphDirection.OUTGOING
            or direction == GraphDirection.BOTH
        ):
            var start = Int(self.index.out_offsets[idx])
            var end = Int(self.index.out_offsets[idx + 1])
            for i in range(start, end):
                var edge_id = self.index.out_edge_ids[i]
                if self._edge_type_matches(edge_id, edge_type):
                    var neighbor = self.index.out_neighbors[i]
                    if neighbor not in seen:
                        seen.add(neighbor)
                        result.append(neighbor)

        if (
            direction == GraphDirection.INCOMING
            or direction == GraphDirection.BOTH
        ):
            var start = Int(self.index.in_offsets[idx])
            var end = Int(self.index.in_offsets[idx + 1])
            for i in range(start, end):
                var edge_id = self.index.in_edge_ids[i]
                if self._edge_type_matches(edge_id, edge_type):
                    var neighbor = self.index.in_neighbors[i]
                    if neighbor not in seen:
                        seen.add(neighbor)
                        result.append(neighbor)

        return result^

    def traverse(
        ref self,
        root: UInt64,
        direction: GraphDirection = GraphDirection.OUTGOING,
        max_hops: Int = 10,
        edge_type: Optional[String] = None,
    ) raises -> List[UInt64]:
        self._validate_node(root)
        if max_hops < 0:
            raise Error("max_hops must be non-negative")

        var visited = Set[UInt64]()
        var distances = Dict[UInt64, Int]()
        var queue = List[UInt64]()
        var results = List[UInt64]()

        visited.add(root)
        distances[root] = 0
        queue.append(root)

        while len(queue) > 0:
            var u = queue.pop(0)
            results.append(u)

            var dist_u = distances[u]
            if dist_u < max_hops:
                var nbrs = self.neighbors(u, direction, edge_type)
                for i in range(len(nbrs)):
                    var v = nbrs[i]
                    if v not in visited:
                        visited.add(v)
                        distances[v] = dist_u + 1
                        queue.append(v)

        return results^

    def has_path(
        ref self,
        from_id: String,
        to_id: String,
        direction: GraphDirection = GraphDirection.OUTGOING,
        max_depth: Int = 10,
        edge_type: Optional[String] = None,
    ) raises -> Bool:
        var path = self.shortest_path(
            from_id, to_id, direction, max_depth, edge_type
        )
        return len(path) > 0

    def shortest_path(
        ref self,
        from_id: String,
        to_id: String,
        direction: GraphDirection = GraphDirection.OUTGOING,
        max_depth: Int = 10,
        edge_type: Optional[String] = None,
    ) raises -> List[String]:
        if max_depth < 0:
            raise Error("max_depth must be non-negative")

        var start = self.node_id_for(from_id)
        var target = self.node_id_for(to_id)
        var empty = List[String]()
        if not start or not target:
            return empty^

        var start_node = start.value()
        var target_node = target.value()
        if start_node == target_node:
            var result = List[String]()
            result.append(from_id)
            return result^

        var visited = Set[UInt64]()
        var distances = Dict[UInt64, Int]()
        var parents = Dict[UInt64, UInt64]()
        var queue = List[UInt64]()

        visited.add(start_node)
        distances[start_node] = 0
        queue.append(start_node)

        while len(queue) > 0:
            var u = queue.pop(0)
            var dist_u = distances[u]
            if dist_u >= max_depth:
                continue

            var nbrs = self.neighbors(u, direction, edge_type)
            for i in range(len(nbrs)):
                var v = nbrs[i]
                if v in visited:
                    continue

                visited.add(v)
                parents[v] = u
                distances[v] = dist_u + 1

                if v == target_node:
                    return self._external_path(start_node, target_node, parents)

                queue.append(v)

        return empty^

    def _external_path(
        ref self,
        start_node: UInt64,
        target_node: UInt64,
        parents: Dict[UInt64, UInt64],
    ) raises -> List[String]:
        var reversed = List[UInt64]()
        var current = target_node
        reversed.append(current)
        while current != start_node:
            current = parents[current]
            reversed.append(current)

        var result = List[String]()
        var i = len(reversed) - 1
        while i >= 0:
            var external_id = self.external_id_for(reversed[i])
            if not external_id:
                raise Error("path contains node without external ID")
            result.append(external_id.value())
            i -= 1
        return result^

    def _flush_deltas(mut self):
        if len(self._pending_edges) == 0:
            return

        # 1. Collect pending edges
        # List.pop(0) is slow, but for small graphs it's fine.
        # Actually, let's just move them all.
        while len(self._pending_edges) > 0:
            self.edges.append(self._pending_edges.pop())

        self._rebuild_index()

    def _rebuild_index(mut self):
        # 2. Reset offsets
        self.index.out_offsets.clear()
        self.index.out_offsets.append(0)
        for _ in range(len(self.nodes)):
            self.index.out_offsets.append(0)

        self.index.in_offsets.clear()
        self.index.in_offsets.append(0)
        for _ in range(len(self.nodes)):
            self.index.in_offsets.append(0)

        # 3. Count degrees
        for i in range(len(self.edges)):
            ref e = self.edges[i]
            self.index.out_offsets[Int(e.src) + 1] += 1
            self.index.in_offsets[Int(e.dst) + 1] += 1

        # 4. Prefix sum to get actual offsets
        for i in range(1, len(self.index.out_offsets)):
            self.index.out_offsets[i] += self.index.out_offsets[i - 1]
        for i in range(1, len(self.index.in_offsets)):
            self.index.in_offsets[i] += self.index.in_offsets[i - 1]

        # 5. Fill neighbors (using temporary counts)
        self.index.out_neighbors.clear()
        self.index.out_edge_ids.clear()
        for _ in range(
            Int(self.index.out_offsets[len(self.index.out_offsets) - 1])
        ):
            self.index.out_neighbors.append(0)
            self.index.out_edge_ids.append(0)

        self.index.in_neighbors.clear()
        self.index.in_edge_ids.clear()
        for _ in range(
            Int(self.index.in_offsets[len(self.index.in_offsets) - 1])
        ):
            self.index.in_neighbors.append(0)
            self.index.in_edge_ids.append(0)

        var cur_out = List[UInt64]()
        var cur_in = List[UInt64]()
        for _ in range(len(self.nodes)):
            cur_out.append(0)
            cur_in.append(0)

        for i in range(len(self.edges)):
            ref e = self.edges[i]
            var out_pos = Int(self.index.out_offsets[Int(e.src)]) + Int(
                cur_out[Int(e.src)]
            )
            self.index.out_neighbors[out_pos] = e.dst
            self.index.out_edge_ids[out_pos] = e.id
            cur_out[Int(e.src)] += 1

            var in_pos = Int(self.index.in_offsets[Int(e.dst)]) + Int(
                cur_in[Int(e.dst)]
            )
            self.index.in_neighbors[in_pos] = e.src
            self.index.in_edge_ids[in_pos] = e.id
            cur_in[Int(e.dst)] += 1

    def bfs(ref self, root: UInt64, max_hops: Int = 10) raises -> List[UInt64]:
        return self.traverse(root, max_hops=max_hops)

    def joint_search[
        dim: Int
    ](
        ref self,
        ref v_index: HNSWIndex[dim],
        query: Span[Float32, _],
        k: Int,
        graph_expansion: Int = 2,
    ) raises -> List[Candidate]:
        """
        1. Perform initial vector search
        2. Expand results through graph
        3. Re-score and return top-k
        """
        var initial_results = v_index.search(query, k=k)
        var visited = Set[UInt64]()
        var candidates = List[Candidate]()

        # Add initial results to candidates
        for i in range(len(initial_results)):
            var c = initial_results[i]
            candidates.append(c)
            var maybe_node = self.node_id_for_vector(UInt64(c.id))
            if maybe_node:
                visited.add(maybe_node.value())

        # Graph expansion
        for i in range(len(initial_results)):
            var maybe_root = self.node_id_for_vector(
                UInt64(initial_results[i].id)
            )
            if not maybe_root:
                continue
            var root = maybe_root.value()
            var nbrs = self.bfs(root, max_hops=graph_expansion)
            for j in range(len(nbrs)):
                var node_id = nbrs[j]
                if node_id not in visited:
                    visited.add(node_id)
                    ref node = self.nodes[Int(node_id)]
                    if not node.vector_id:
                        continue
                    var vector_id = node.vector_id.value()
                    # Score expanded candidate
                    var v_vec = v_index.get_vector(Int(vector_id))
                    var dist = v_index._distance(query, v_vec)
                    candidates.append(Candidate(UInt32(vector_id), dist))

        # Sort candidates
        @parameter
        def cmp(a: Candidate, b: Candidate) -> Bool:
            return a.distance < b.distance

        var span = Span[Candidate, MutAnyOrigin](
            ptr=candidates.unsafe_ptr(), length=len(candidates)
        )
        sort[cmp](span)

        var top_k = List[Candidate]()
        for i in range(min(k, len(candidates))):
            top_k.append(candidates[i])

        return top_k^
