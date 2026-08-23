"""
Tests for immutable segment binary format and typed pointer access.
"""
from std.testing import assert_equal, assert_true
from mmap_segment import (
    SegmentHeader,
    HEADER_SIZE,
    SEGMENT_MAGIC,
    SEGMENT_VERSION,
    open_immutable_segment,
    save_hnsw_immutable_segment,
    load_segment_header,
)


def test_segment_header_roundtrip() raises:
    """SegmentHeader create preserves all fields."""
    var header = SegmentHeader.create(1000, 128)
    assert_equal(header.magic, SEGMENT_MAGIC)
    assert_equal(header.version, SEGMENT_VERSION)
    assert_equal(header.num_elements, 1000)
    assert_equal(header.dim, 128)
    assert_equal(header.vectors_offset, HEADER_SIZE)
    assert_equal(header.vectors_length, 1000 * 128 * 4)
    assert_equal(header.codes_offset, HEADER_SIZE + 1000 * 128 * 4)
    assert_equal(header.codes_length, 1000 * 128)


def test_segment_header_sizes() raises:
    """Header offsets are correct for various element/dim combinations."""
    var h1 = SegmentHeader.create(10, 64)
    assert_equal(h1.vectors_offset, HEADER_SIZE)

    var h2 = SegmentHeader.create(100_000, 128)
    assert_equal(h2.vectors_offset, HEADER_SIZE)
    assert_equal(h2.codes_offset, HEADER_SIZE + 100_000 * 128 * 4)


def test_save_and_load_segment() raises:
    """Write a small segment and load it back with typed access."""
    var test_path = "/tmp/omendb_vector_mmap_test_segment.seg"
    comptime dim = 4
    var n = 3

    var data = List[Float32]()
    data.append(1.0)
    data.append(2.0)
    data.append(3.0)
    data.append(4.0)
    data.append(5.0)
    data.append(6.0)
    data.append(7.0)
    data.append(8.0)
    data.append(9.0)
    data.append(10.0)
    data.append(11.0)
    data.append(12.0)

    var codes = List[UInt8]()
    for i in range(n * dim):
        codes.append(UInt8(i * 10))

    save_hnsw_immutable_segment(test_path, data, codes, n, dim)

    var header = load_segment_header(test_path)
    assert_equal(header.num_elements, n)
    assert_equal(header.dim, dim)

    var segment = open_immutable_segment(test_path)
    assert_true(segment.is_valid(), "segment should be valid")
    assert_equal(segment.num_elements(), n)
    assert_equal(segment.dim(), dim)

    for i in range(n):
        var vec_ptr = segment.vector_ptr(i)
        for d in range(dim):
            var expected = Float32(i * dim + d + 1)
            assert_equal(vec_ptr[d], expected)

    for i in range(n):
        var code_ptr = segment.code_ptr(i)
        for d in range(dim):
            var expected = UInt8((i * dim + d) * 10)
            assert_equal(code_ptr[d], expected)

    print("PASS test_save_and_load_segment")


def test_segment_larger_dataset() raises:
    """Segment works with 1000 vectors of dim 128."""
    var test_path = "/tmp/omendb_vector_mmap_test_large.seg"
    comptime dim = 128
    var n = 1000

    var data = List[Float32]()
    for i in range(n * dim):
        data.append(Float32(i % 1000) * 0.001)

    var codes = List[UInt8]()
    for i in range(n * dim):
        codes.append(UInt8(i % 256))

    save_hnsw_immutable_segment(test_path, data, codes, n, dim)

    var segment = open_immutable_segment(test_path)
    assert_true(segment.is_valid())
    assert_equal(segment.num_elements(), n)
    assert_equal(segment.dim(), dim)

    var vec_0 = segment.vector_ptr(0)
    assert_equal(vec_0[0], data[0])
    assert_equal(vec_0[dim - 1], data[dim - 1])

    var vec_500 = segment.vector_ptr(500)
    assert_equal(vec_500[0], data[500 * dim])

    var vec_999 = segment.vector_ptr(999)
    assert_equal(vec_999[0], data[999 * dim])

    print("PASS test_segment_larger_dataset")


def test_segment_missing_file() raises:
    """Opening a nonexistent file raises an explicit error."""
    var raised = False
    try:
        _ = open_immutable_segment("/tmp/nonexistent_omendb_vector_segment.seg")
    except e:
        raised = True
        assert_true(String(e).find("not found") >= 0, "error: " + String(e))
    assert_true(raised, "should raise for missing file")


def main() raises:
    test_segment_header_roundtrip()
    print("PASS test_segment_header_roundtrip")
    test_segment_header_sizes()
    print("PASS test_segment_header_sizes")
    test_save_and_load_segment()
    test_segment_larger_dataset()
    test_segment_missing_file()
    print("PASS test_segment_missing_file")
