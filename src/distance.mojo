from std.algorithm import vectorize
from std.sys.info import simd_width_of

comptime simd_width = simd_width_of[DType.float32]()


@always_inline
def l2_distance[dim: Int](a: Span[Float32, _], b: Span[Float32, _]) -> Float32:
    var a_ptr = a.unsafe_ptr()
    var b_ptr = b.unsafe_ptr()

    # Use 2 accumulators with 8-wide SIMD
    var acc0 = SIMD[DType.float32, 8](0.0)
    var acc1 = SIMD[DType.float32, 8](0.0)
    var d: Int = 0
    while d + 16 <= dim:
        var da0 = a_ptr.load[width=8](d)
        var db0 = b_ptr.load[width=8](d)
        var diff0 = da0 - db0
        acc0 += diff0 * diff0

        var da1 = a_ptr.load[width=8](d + 8)
        var db1 = b_ptr.load[width=8](d + 8)
        var diff1 = da1 - db1
        acc1 += diff1 * diff1
        d += 16

    var dist: Float32 = (acc0 + acc1).reduce_add()
    for i in range(d, dim):
        var diff = a_ptr.load[1](i)[0] - b_ptr.load[1](i)[0]
        dist += diff * diff
    return dist


@always_inline
def dot_product[dim: Int](a: Span[Float32, _], b: Span[Float32, _]) -> Float32:
    var acc: Float32 = 0.0

    def body[w: Int](i: Int) {mut acc, read a, read b}:
        var da = a.unsafe_ptr().load[width=w](i)
        var db = b.unsafe_ptr().load[width=w](i)
        acc += (da * db).reduce_add()

    vectorize[simd_width](dim, body)
    return acc


@always_inline
def cosine_distance[
    dim: Int
](a: Span[Float32, _], b: Span[Float32, _]) -> Float32:
    var dot: Float32 = 0.0
    var norm_a: Float32 = 0.0
    var norm_b: Float32 = 0.0

    def body[w: Int](i: Int) {mut dot, mut norm_a, mut norm_b, read a, read b}:
        var da = a.unsafe_ptr().load[width=w](i)
        var db = b.unsafe_ptr().load[width=w](i)
        dot += (da * db).reduce_add()
        norm_a += (da * da).reduce_add()
        norm_b += (db * db).reduce_add()

    vectorize[simd_width](dim, body)
    from std.math import sqrt

    var denom = sqrt(norm_a) * sqrt(norm_b)
    if denom == 0.0:
        return 0.0
    return 1.0 - (dot / denom)
