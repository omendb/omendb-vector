"""
Memory budget tracking for OmenDB-Mojo.

Estimates memory usage of vector data, SQ8 codes, HNSW graph, and
metadata structures. Enables production deployment with predictable
memory consumption.

Usage:
    var budget = MemoryBudget(max_bytes=1024 * 1024 * 1024)  # 1GB
    budget.record_vector_insert(dim=128)
    budget.record_code_insert(dim=128)
    if budget.is_over_budget():
        reject_insert()
"""


struct MemoryBudget(Copyable, Movable):
    """Tracks estimated memory usage across store components."""

    var max_bytes: Int
    var vector_bytes: Int
    var code_bytes: Int
    var graph_bytes: Int
    var metadata_bytes: Int
    var text_bytes: Int
    var other_bytes: Int

    def __init__(out self, max_bytes: Int = 0):
        """max_bytes=0 means no limit."""
        self.max_bytes = max_bytes
        self.vector_bytes = 0
        self.code_bytes = 0
        self.graph_bytes = 0
        self.metadata_bytes = 0
        self.text_bytes = 0
        self.other_bytes = 0

    def total_bytes(self) -> Int:
        """Total estimated memory usage in bytes."""
        return (
            self.vector_bytes
            + self.code_bytes
            + self.graph_bytes
            + self.metadata_bytes
            + self.text_bytes
            + self.other_bytes
        )

    def is_over_budget(self) -> Bool:
        """True if usage exceeds max_bytes (no limit if max_bytes==0)."""
        if self.max_bytes <= 0:
            return False
        return self.total_bytes() > self.max_bytes

    def remaining_bytes(self) -> Int:
        """Bytes remaining before hitting the budget limit. -1 if no limit."""
        if self.max_bytes <= 0:
            return -1
        return max(0, self.max_bytes - self.total_bytes())

    def can_insert_vector(self, dim: Int) -> Bool:
        """Check if inserting a vector would exceed the budget."""
        if self.max_bytes <= 0:
            return True
        var needed = dim * 4 + dim  # f32 vector + SQ8 codes
        return self.total_bytes() + needed <= self.max_bytes

    # --- Recording operations ---

    def record_vector_insert(mut self, dim: Int):
        """Record memory for one f32 vector (dim * 4 bytes)."""
        self.vector_bytes += dim * 4

    def record_code_insert(mut self, dim: Int):
        """Record memory for one SQ8 code set (dim bytes)."""
        self.code_bytes += dim

    def record_vector_delete(mut self, dim: Int):
        """Record memory freed by deleting one vector."""
        self.vector_bytes = max(0, self.vector_bytes - dim * 4)
        self.code_bytes = max(0, self.code_bytes - dim)

    def record_graph_neighbor(mut self, layer: Int):
        """Record memory for one graph neighbor entry (~8 bytes: id + distance).
        """
        self.graph_bytes += 8

    def record_metadata(mut self, key_len: Int, value_len: Int):
        """Record memory for a metadata entry."""
        self.metadata_bytes += key_len + value_len + 64  # 64 bytes overhead

    def record_text_doc(mut self, text_len: Int):
        """Record memory for a text document."""
        self.text_bytes += text_len + 128  # 128 bytes overhead

    def record_other(mut self, bytes: Int):
        """Record miscellaneous memory usage."""
        self.other_bytes += bytes

    # --- Bulk estimation ---

    @staticmethod
    def estimate_hnsw_index(num_vectors: Int, dim: Int, M: Int) -> Int:
        """Estimate memory for an HNSW index with given parameters."""
        var vector_bytes = num_vectors * dim * 4
        var code_bytes = num_vectors * dim
        var graph_bytes = (
            num_vectors * M * 2 * 8
        )  # M neighbors * 2 layers * 8 bytes
        var level_bytes = num_vectors * 8  # Int per node
        var lock_bytes = num_vectors * 64  # spinlock per node
        return (
            vector_bytes + code_bytes + graph_bytes + level_bytes + lock_bytes
        )

    @staticmethod
    def estimate_store_metadata(
        num_vectors: Int, avg_id_len: Int, avg_meta_len: Int
    ) -> Int:
        """Estimate memory for store metadata structures."""
        var id_bytes = num_vectors * (avg_id_len + 64)  # String overhead
        var meta_bytes = num_vectors * (avg_meta_len + 64)
        var flag_bytes = num_vectors * 3  # deleted, text_only, etc.
        return id_bytes + meta_bytes + flag_bytes

    def summary(self) -> String:
        """Human-readable memory usage summary."""
        var total = self.total_bytes()
        var result = "Memory Budget: "
        result += _format_bytes(total)
        if self.max_bytes > 0:
            result += " / " + _format_bytes(self.max_bytes)
            var pct = Int(Float64(total) / Float64(self.max_bytes) * 100.0)
            result += " (" + String(pct) + "%)"
        result += "\n"
        result += "  vectors:   " + _format_bytes(self.vector_bytes) + "\n"
        result += "  codes:     " + _format_bytes(self.code_bytes) + "\n"
        result += "  graph:     " + _format_bytes(self.graph_bytes) + "\n"
        result += "  metadata:  " + _format_bytes(self.metadata_bytes) + "\n"
        result += "  text:      " + _format_bytes(self.text_bytes) + "\n"
        result += "  other:     " + _format_bytes(self.other_bytes)
        return result


def _format_bytes(bytes: Int) -> String:
    """Format bytes as human-readable string."""
    if bytes < 1024:
        return String(bytes) + " B"
    elif bytes < 1024 * 1024:
        return String(bytes / 1024) + " KB"
    elif bytes < 1024 * 1024 * 1024:
        return String(bytes / (1024 * 1024)) + " MB"
    else:
        return String(bytes / (1024 * 1024 * 1024)) + " GB"
