from std.math import min, max


@fieldwise_init
struct SQ8Quantizer[dim: Int](Copyable):
    var min_val: Float32
    var max_val: Float32
    var scale: Float32
    var inv_scale: Float32

    def __init__(out self, min_val: Float32, max_val: Float32):
        self.min_val = min_val
        self.max_val = max_val
        self.scale = (max_val - min_val) / 255.0
        if self.scale == 0:
            self.scale = 1.0
        self.inv_scale = 1.0 / self.scale

    def __init__(out self, *, copy: Self):
        self.min_val = copy.min_val
        self.max_val = copy.max_val
        self.scale = copy.scale
        self.inv_scale = copy.inv_scale

    def quantize(ref self, vec: Span[Float32, _], mut out_codes: List[UInt8]):
        for i in range(Self.dim):
            var val = vec[i]
            var q = Int((val - self.min_val) * self.inv_scale)
            if q < 0:
                q = 0
            if q > 255:
                q = 255
            out_codes.append(UInt8(q))

    def dequantize(ref self, codes: Span[UInt8, _], mut out_vec: List[Float32]):
        for i in range(Self.dim):
            var val = Float32(codes[i]) * self.scale + self.min_val
            out_vec.append(val)
