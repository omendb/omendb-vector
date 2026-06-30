"""Dynamic flat index for arbitrary dense dimensions.

This module provides a runtime-dimensional flat index that supports
any positive dimension up to DENSE_DYNAMIC_FLAT_MAX_DIM (8192).
It stores vectors in a flat Float32 array with runtime dimension loops.
"""

from std.collections import Dict, List


struct DynamicFlatIndex(Movable):
    """A flat index that supports arbitrary dimensions at runtime.

    Unlike FlatIndex[dim] which requires compile-time dimension,
    this stores the dimension as a runtime value.
    """

    var data: List[Float32]
    var dim: Int
    var num_elements: Int

    def __init__(out self, dim: Int):
        self.data = List[Float32]()
        self.dim = dim
        self.num_elements = 0

    def __init__(out self, *, deinit take: Self):
        self.data = take.data^
        self.dim = take.dim
        self.num_elements = take.num_elements

    def reserve(mut self, capacity: Int):
        self.data.reserve(capacity * self.dim)

    def insert(mut self, vec: Span[Float32, _]):
        for i in range(self.dim):
            self.data.append(vec[i])
        self.num_elements += 1

    def get_vector(ref self, id: Int) -> Span[Float32, origin_of(self.data)]:
        var start = id * self.dim
        return Span(ptr=self.data.unsafe_ptr() + start, length=self.dim)

    def search(ref self, query: Span[Float32, _], k: Int) -> List[UInt32]:
        """Search for k nearest neighbors using L2 distance.

        Returns list of element IDs sorted by distance (ascending).
        """
        # Compute distances for all elements
        var distances = List[Float32]()
        distances.reserve(self.num_elements)
        for i in range(self.num_elements):
            var vec = self.get_vector(i)
            var sum: Float32 = 0.0
            for j in range(self.dim):
                var diff = query[j] - vec[j]
                sum += diff * diff
            distances.append(sum)

        # Simple selection sort for top-k (good enough for small k)
        var indices = List[UInt32]()
        indices.reserve(self.num_elements)
        for i in range(self.num_elements):
            indices.append(UInt32(i))

        # Partial sort to find top-k
        for i in range(min(k, self.num_elements)):
            var min_idx = i
            for j in range(i + 1, self.num_elements):
                if (
                    distances[Int(indices[j])]
                    < distances[Int(indices[min_idx])]
                ):
                    min_idx = j
            # Swap
            var temp = indices[i]
            indices[i] = indices[min_idx]
            indices[min_idx] = temp

        # Return top-k
        var result = List[UInt32]()
        result.reserve(min(k, self.num_elements))
        for i in range(min(k, self.num_elements)):
            result.append(indices[i])
        return result^

    def search_with_distances(
        ref self, query: Span[Float32, _], k: Int
    ) raises -> List[UInt32]:
        """Search for k nearest neighbors, returning IDs sorted by distance.

        Returns indices sorted by distance (ascending).
        """
        # Compute distances for all elements
        var distances = List[Float32]()
        distances.reserve(self.num_elements)
        for i in range(self.num_elements):
            var vec = self.get_vector(i)
            var sum: Float32 = 0.0
            for j in range(self.dim):
                var diff = query[j] - vec[j]
                sum += diff * diff
            distances.append(sum)

        # Simple selection sort for top-k
        var indices = List[UInt32]()
        indices.reserve(self.num_elements)
        for i in range(self.num_elements):
            indices.append(UInt32(i))

        # Partial sort to find top-k
        for i in range(min(k, self.num_elements)):
            var min_idx = i
            for j in range(i + 1, self.num_elements):
                if (
                    distances[Int(indices[j])]
                    < distances[Int(indices[min_idx])]
                ):
                    min_idx = j
            # Swap
            var temp = indices[i]
            indices[i] = indices[min_idx]
            indices[min_idx] = temp

        # Return top-k
        var result = List[UInt32]()
        var count = min(k, self.num_elements)
        result.reserve(count)
        for i in range(count):
            result.append(indices[i])
        return result^
