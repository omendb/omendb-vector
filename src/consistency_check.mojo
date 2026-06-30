"""
Consistency checks for VectorStore invariants.

Verifies that the canonical item state, HNSW index, text index, and graph
edges are mutually consistent. Run after mutations, after recovery, and
as a pre-flush gate.
"""
from collections import Dict
from store import VectorStore


struct ConsistencyViolation(Copyable, Movable):
    var code: String
    var detail: String

    def __init__(out self, code: String, detail: String):
        self.code = code
        self.detail = detail

    def __str__(self) -> String:
        return self.code + ": " + self.detail


struct ConsistencyReport(Copyable, Movable):
    var violations: List[ConsistencyViolation]
    var checked: Int

    def __init__(out self):
        self.violations = List[ConsistencyViolation]()
        self.checked = 0

    def ok(ref self) -> Bool:
        return len(self.violations) == 0

    def add(mut self, code: String, detail: String):
        self.violations.append(ConsistencyViolation(code, detail))

    def summary(ref self) -> String:
        if self.ok():
            return "CONSISTENT (" + String(self.checked) + " checks passed)"
        var s = String(len(self.violations)) + " VIOLATIONS:\n"
        for i in range(len(self.violations)):
            s += "  " + self.violations[i].__str__() + "\n"
        return s


def verify_consistency[
    dim: Int
](ref store: VectorStore[dim]) raises -> ConsistencyReport:
    """Run all consistency checks on the store. Returns a report with violations.
    """
    var report = ConsistencyReport()
    var n = len(store.vector_external_ids)

    # --- Parallel array length checks ---
    if len(store.metadata) != n:
        report.add(
            "METADATA_LENGTH",
            "len(metadata)="
            + String(len(store.metadata))
            + " != len(vector_external_ids)="
            + String(n),
        )
    if len(store.source_spans) != n:
        report.add(
            "SOURCE_SPANS_LENGTH",
            "len(source_spans)="
            + String(len(store.source_spans))
            + " != len(vector_external_ids)="
            + String(n),
        )
    if len(store.deleted) != n:
        report.add(
            "DELETED_LENGTH",
            "len(deleted)="
            + String(len(store.deleted))
            + " != len(vector_external_ids)="
            + String(n),
        )
    # text_only is sparse: only populated when items are marked text_only.
    # Items beyond len(text_only) are implicitly not text-only.
    # This is fine; search checks `id < len(text_only) and text_only[id]`.
    if len(store.text_only) > n:
        report.add(
            "TEXT_ONLY_LENGTH",
            "len(text_only)="
            + String(len(store.text_only))
            + " > len(vector_external_ids)="
            + String(n),
        )
    report.checked += 4

    # --- Deleted count check ---
    var actual_deleted = 0
    for i in range(len(store.deleted)):
        if store.deleted[i]:
            actual_deleted += 1
    if actual_deleted != store.deleted_count:
        report.add(
            "DELETED_COUNT",
            "actual_deleted="
            + String(actual_deleted)
            + " != deleted_count="
            + String(store.deleted_count),
        )
    report.checked += 1

    # --- Text-only count check ---
    var actual_text_only = 0
    for i in range(len(store.text_only)):
        if store.text_only[i]:
            actual_text_only += 1
    if actual_text_only != store.text_only_count:
        report.add(
            "TEXT_ONLY_COUNT",
            "actual_text_only="
            + String(actual_text_only)
            + " != text_only_count="
            + String(store.text_only_count),
        )
    report.checked += 1

    # --- Dual-status check: no item can be both deleted and text_only ---
    var dual_count = 0
    var check_len = min(len(store.deleted), len(store.text_only))
    for i in range(check_len):
        if store.deleted[i] and store.text_only[i]:
            dual_count += 1
    if dual_count > 0:
        report.add(
            "DUAL_STATUS",
            String(dual_count) + " items are both deleted and text_only",
        )
    report.checked += 1

    # --- HNSW index bound check ---
    var hnsw_size = store.index.num_elements
    if hnsw_size < n:
        report.add(
            "HNSW_TOO_SMALL",
            "index.num_elements="
            + String(hnsw_size)
            + " < len(vector_external_ids)="
            + String(n),
        )
    report.checked += 1

    # --- Live item HNSW coverage ---
    var live_count = 0
    var missing_hnsw = 0
    for i in range(n):
        if i < len(store.deleted) and store.deleted[i]:
            continue
        if i < len(store.text_only) and store.text_only[i]:
            continue
        live_count += 1
        if i >= hnsw_size:
            missing_hnsw += 1
    if missing_hnsw > 0:
        report.add(
            "LIVE_ITEM_MISSING_HNSW",
            String(missing_hnsw)
            + " of "
            + String(live_count)
            + " live items have no HNSW entry",
        )
    report.checked += 1

    # --- External-to-vector map consistency ---
    # Round-trip check: for each live item, external_to_vector must map back
    # to the correct index
    var map_errors = 0
    for i in range(n):
        if i < len(store.deleted) and store.deleted[i]:
            continue
        var ext_id = store.vector_external_ids[i]
        if ext_id not in store.external_to_vector:
            map_errors += 1
        elif Int(store.external_to_vector[ext_id]) != i:
            map_errors += 1
    if map_errors > 0:
        report.add(
            "EXTERNAL_MAP_CORRUPT",
            String(map_errors)
            + " live items have incorrect external_to_vector mapping",
        )
    report.checked += 1

    # --- Text index consistency ---
    if store.options.text_enabled:
        var text_doc_count = len(store.text_doc_vector_ids)
        if text_doc_count != len(store.text_doc_texts):
            report.add(
                "TEXT_DOC_LENGTH",
                "text_doc_vector_ids="
                + String(text_doc_count)
                + " != text_doc_texts="
                + String(len(store.text_doc_texts)),
            )
        # Each text doc's vector_id should be < n
        for i in range(text_doc_count):
            if Int(store.text_doc_vector_ids[i]) >= n:
                report.add(
                    "TEXT_DOC_INVALID_VECTOR_ID",
                    "text_doc["
                    + String(i)
                    + "].vector_id="
                    + String(Int(store.text_doc_vector_ids[i]))
                    + " >= n="
                    + String(n),
                )
                break
        report.checked += 2
    else:
        report.checked += 2

    # --- Graph edge consistency ---
    if store.options.graph_enabled:
        var node_count = len(store.graph.node_external_ids)
        var edge_count = len(store.graph.edges)
        var broken_edges = 0
        var removed_node_edges = 0
        for i in range(edge_count):
            ref edge = store.graph.edges[i]
            if Int(edge.src) >= node_count:
                broken_edges += 1
            elif store.graph.node_external_ids[Int(edge.src)] == "":
                removed_node_edges += 1
            if Int(edge.dst) >= node_count:
                broken_edges += 1
            elif store.graph.node_external_ids[Int(edge.dst)] == "":
                removed_node_edges += 1
        if broken_edges > 0:
            report.add(
                "GRAPH_BROKEN_EDGES",
                String(broken_edges)
                + " edge endpoints reference missing nodes",
            )
        if removed_node_edges > 0:
            report.add(
                "GRAPH_REMOVED_NODE_EDGES",
                String(removed_node_edges)
                + " edge endpoints reference removed (empty-id) nodes",
            )
        report.checked += 1
    else:
        report.checked += 1

    return report^
