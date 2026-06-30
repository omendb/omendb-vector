from std.testing import (
    assert_equal,
    assert_true,
    assert_almost_equal,
    TestSuite,
)
from distance import l2_distance, dot_product, cosine_distance
from std.memory import UnsafePointer


def test_l2_distance() raises:
    var a: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var b: List[Float32] = [4.0, 3.0, 2.0, 1.0]
    var a_span = Span(ptr=a.unsafe_ptr(), length=4)
    var b_span = Span(ptr=b.unsafe_ptr(), length=4)
    var dist = l2_distance[4](a_span, b_span)
    assert_almost_equal(
        dist, 20.0
    )  # (1-4)^2 + (2-3)^2 + (3-2)^2 + (4-1)^2 = 9 + 1 + 1 + 9 = 20


def test_dot_product() raises:
    var a: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var b: List[Float32] = [4.0, 3.0, 2.0, 1.0]
    var a_span = Span(ptr=a.unsafe_ptr(), length=4)
    var b_span = Span(ptr=b.unsafe_ptr(), length=4)
    var dist = dot_product[4](a_span, b_span)
    assert_almost_equal(dist, 20.0)  # 4 + 6 + 6 + 4 = 20


def test_cosine_distance() raises:
    var a: List[Float32] = [1.0, 0.0, 0.0, 0.0]
    var b: List[Float32] = [0.0, 1.0, 0.0, 0.0]
    var a_span = Span(ptr=a.unsafe_ptr(), length=4)
    var b_span = Span(ptr=b.unsafe_ptr(), length=4)
    var dist = cosine_distance[4](a_span, b_span)
    assert_almost_equal(dist, 1.0)  # 1 - 0

    var c: List[Float32] = [1.0, 0.0, 0.0, 0.0]
    var d: List[Float32] = [1.0, 0.0, 0.0, 0.0]
    var c_span = Span(ptr=c.unsafe_ptr(), length=4)
    var d_span = Span(ptr=d.unsafe_ptr(), length=4)
    var dist2 = cosine_distance[4](c_span, d_span)
    assert_almost_equal(dist2, 0.0)  # 1 - 1


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
