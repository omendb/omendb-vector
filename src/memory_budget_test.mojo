"""
Tests for memory budget tracking.
"""
from std.testing import assert_equal, assert_true, assert_false
from memory_budget import MemoryBudget


def test_budget_basic() raises:
    """Basic budget tracking works."""
    var budget = MemoryBudget(max_bytes=1024)
    assert_equal(budget.total_bytes(), 0)
    assert_false(budget.is_over_budget())
    assert_equal(budget.remaining_bytes(), 1024)

    budget.record_vector_insert(dim=128)
    assert_equal(budget.vector_bytes, 128 * 4)
    assert_equal(budget.total_bytes(), 128 * 4)

    budget.record_code_insert(dim=128)
    assert_equal(budget.code_bytes, 128)
    assert_equal(budget.total_bytes(), 128 * 4 + 128)

    print("PASS test_budget_basic")


def test_budget_over_limit() raises:
    """is_over_budget returns true when exceeding limit."""
    var budget = MemoryBudget(max_bytes=100)
    assert_false(budget.is_over_budget())

    budget.record_vector_insert(dim=32)  # 128 bytes
    assert_true(budget.is_over_budget())
    assert_equal(budget.remaining_bytes(), 0)

    print("PASS test_budget_over_limit")


def test_budget_no_limit() raises:
    """max_bytes=0 means no limit."""
    var budget = MemoryBudget(max_bytes=0)
    assert_false(budget.is_over_budget())
    assert_equal(budget.remaining_bytes(), -1)

    for i in range(1000):
        budget.record_vector_insert(dim=128)

    assert_false(budget.is_over_budget())
    print("PASS test_budget_no_limit")


def test_can_insert_vector() raises:
    """can_insert_vector checks if insert would exceed budget."""
    var budget = MemoryBudget(max_bytes=1000)

    # 128 * 4 + 128 = 640 bytes
    assert_true(budget.can_insert_vector(dim=128))
    budget.record_vector_insert(dim=128)
    budget.record_code_insert(dim=128)

    # 640 bytes used, 360 remaining
    # Another 128-dim insert would need 640 bytes — over budget
    assert_false(budget.can_insert_vector(dim=128))

    print("PASS test_can_insert_vector")


def test_budget_delete() raises:
    """Deleting a vector frees tracked memory."""
    var budget = MemoryBudget(max_bytes=1000)

    budget.record_vector_insert(dim=128)
    budget.record_code_insert(dim=128)
    var used_after_insert = budget.total_bytes()

    budget.record_vector_delete(dim=128)
    assert_equal(budget.vector_bytes, 0)
    assert_equal(budget.code_bytes, 0)
    assert_equal(budget.total_bytes(), 0)

    print("PASS test_budget_delete")


def test_budget_estimate_hnsw() raises:
    """estimate_hnsw_index gives reasonable estimates."""
    var est = MemoryBudget.estimate_hnsw_index(num_vectors=1000, dim=128, M=16)
    # vectors: 1000 * 128 * 4 = 512000
    # codes: 1000 * 128 = 128000
    # graph: 1000 * 16 * 2 * 8 = 256000
    # levels: 1000 * 8 = 8000
    # locks: 1000 * 64 = 64000
    # total: 968000
    assert_true(est > 900000)
    assert_true(est < 1100000)

    print("PASS test_budget_estimate_hnsw")


def test_budget_summary() raises:
    """summary() returns readable string."""
    var budget = MemoryBudget(max_bytes=1024 * 1024)
    budget.record_vector_insert(dim=128)
    var summary = budget.summary()
    assert_true(summary.find("Memory Budget:") >= 0)
    assert_true(summary.find("vectors:") >= 0)

    print("PASS test_budget_summary")


def main() raises:
    test_budget_basic()
    test_budget_over_limit()
    test_budget_no_limit()
    test_can_insert_vector()
    test_budget_delete()
    test_budget_estimate_hnsw()
    test_budget_summary()
