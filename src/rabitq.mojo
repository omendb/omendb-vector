"""
RaBitQ 1-bit vector quantization.

Encodes each dimension to {-1, +1} via randomized rotation + sign threshold.
Distance is estimated via XNOR + popcount (Hamming distance on packed bits).

Reference: RaBitQ, Jian et al. 2024 — arXiv:2405.12497
SymphonyQG, SIGMOD 2025 — arXiv:2411.12229
"""

from std.memory import Span
from std.sys import size_of
from std.math import sqrt


# ── popcount helper ───────────────────────────────────────────────

def _popcount_u64(x: UInt64) -> Int:
    """Population count (number of set bits) in a 64-bit word."""
    var v = x
    v = v - ((v >> 1) & 0x5555555555555555)
    v = (v & 0x3333333333333333) + ((v >> 2) & 0x3333333333333333)
    v = (v + (v >> 4)) & 0x0F0F0F0F0F0F0F0F
    return Int((v * 0x0101010101010101) >> 56)


# ── RaBitQCode ────────────────────────────────────────────────────

struct RaBitQCode(Copyable, Movable):
    """A single 1-bit quantized vector: packed bits + norm correction factor."""
    var bits: List[UInt64]          # packed 1-bit signs, 64 dims per word
    var norm_factor: Float32         # ||x|| / sqrt(D) — error-bound correction

    def __init__(out self):
        self.bits = List[UInt64]()
        self.norm_factor = 0.0

    def __init__(out self, *, deinit take: Self):
        self.bits = take.bits^
        self.norm_factor = take.norm_factor

    def __init__(out self, *, copy: Self):
        var b = List[UInt64]()
        for i in range(len(copy.bits)):
            b.append(copy.bits[i])
        self.bits = b^
        self.norm_factor = copy.norm_factor


# ── RaBitQEncoder ─────────────────────────────────────────────────

struct RaBitQEncoder[dim: Int](Copyable):
    """Encodes F32 vectors to 1-bit RaBitQ codes.

    Rotation: a dim×dim randomized orthogonal matrix (simplified as
    pairwise shuffle + sign flip — a single iteration of Kac walk).
    Threshold: per-dimension zero-centering after rotation.
    """

    var rotation_signs: List[Float32]    # ±1 per dim for pair shuffling
    var rotation_pairs: List[Int]         # paired dim index for shuffling
    var centroid: List[Float32]           # mean vector (subtracted pre-rotation)
    var rot_matrix: List[Float32]         # D×D random orthogonal matrix (row-major)
    var _fitted: Bool                     # True after fit() called

    # Precomputed: number of UInt64 words needed
    comptime words_per_vec: Int = (Self.dim + 63) // 64

    def __init__(out self):
        self.rotation_signs = List[Float32]()
        self.rotation_pairs = List[Int]()
        self.centroid = List[Float32]()
        self.rot_matrix = List[Float32]()
        self._fitted = False

    def __init__(out self, *, deinit take: Self):
        self.rotation_signs = take.rotation_signs^
        self.rotation_pairs = take.rotation_pairs^
        self.centroid = take.centroid^
        self.rot_matrix = take.rot_matrix^
        self._fitted = take._fitted

    def __init__(out self, *, copy: Self):
        var s = List[Float32]()
        var p = List[Int]()
        var c = List[Float32]()
        var rm = List[Float32]()
        for i in range(len(copy.rotation_signs)):
            s.append(copy.rotation_signs[i])
        for i in range(len(copy.rotation_pairs)):
            p.append(copy.rotation_pairs[i])
        for i in range(len(copy.centroid)):
            c.append(copy.centroid[i])
        for i in range(len(copy.rot_matrix)):
            rm.append(copy.rot_matrix[i])
        self.rotation_signs = s^
        self.rotation_pairs = p^
        self.centroid = c^
        self.rot_matrix = rm^
        self._fitted = copy._fitted

    def fit(mut self, data: Span[Float32, _], n: Int):
        """Learn centroid + random orthogonal rotation matrix.

        Uses randomized Gram-Schmidt: generate D random vectors from N(0,1),
        orthogonalize, and store as D×D row-major matrix.
        """
        # Compute centroid
        self.centroid = List[Float32]()
        self.centroid.reserve(Self.dim)
        for d in range(Self.dim):
            var s: Float32 = 0.0
            for i in range(n):
                s += data[i * Self.dim + d]
            self.centroid.append(s / Float32(n))

        # Generate random matrix via Gram-Schmidt orthogonalization
        self.rot_matrix = List[Float32]()
        self.rot_matrix.reserve(Self.dim * Self.dim)

        # Seed matrix with pseudorandom values (deterministic from dimension)
        var tmp = List[Float32]()
        for i in range(Self.dim * Self.dim):
            var r = Float32((i * 7919 + 6271) % 10007) / 10007.0 - 0.5
            tmp.append(r)

        # Gram-Schmidt orthogonalize rows
        for i in range(Self.dim):
            # Copy row i from tmp
            var row_start = i * Self.dim
            for j in range(Self.dim):
                self.rot_matrix.append(tmp[row_start + j])
            # Subtract projections onto previous rows
            for k in range(i):
                var k_start = k * Self.dim
                var dot: Float64 = 0.0
                for j in range(Self.dim):
                    dot += Float64(self.rot_matrix[row_start + j]) * Float64(self.rot_matrix[k_start + j])
                for j in range(Self.dim):
                    self.rot_matrix[row_start + j] -= Float32(dot) * self.rot_matrix[k_start + j]
            # Normalize
            var norm_sq: Float64 = 0.0
            for j in range(Self.dim):
                norm_sq += Float64(self.rot_matrix[row_start + j]) * Float64(self.rot_matrix[row_start + j])
            var inv_norm = Float32(1.0 / sqrt(norm_sq))
            for j in range(Self.dim):
                self.rot_matrix[row_start + j] *= inv_norm

    def _rotate(ref self, vec: Span[Float32, _]) -> List[Float32]:
        """Apply orthogonal rotation: result = R * (vec - centroid)."""
        var result = List[Float32]()
        result.reserve(Self.dim)
        for _ in range(Self.dim):
            result.append(0.0)

        for i in range(Self.dim):
            var row_start = i * Self.dim
            var dot: Float64 = 0.0
            for j in range(Self.dim):
                dot += Float64(self.rot_matrix[row_start + j]) * Float64(vec[j] - self.centroid[j])
            result[i] = Float32(dot)

        return result^

    def encode_first(mut self, vector: Span[Float32, _]) raises -> RaBitQCode:
        """Encode the first vector — auto-fits the encoder if not already.

        Subsequent vectors should use encode() (non-mutating).
        """
        if not self._fitted:
            self.fit(vector, 1)
            self._fitted = True
        return self.encode(vector)

    def encode(ref self, vector: Span[Float32, _]) raises -> RaBitQCode:
        """Encode one vector to RaBitQ 1-bit code.

        Encoder must be fitted via fit() first (auto-called by encode_first()).

        Steps:
          1. Subtract centroid
          2. Apply random rotation (Kac walk)
          3. Sign threshold: bit = 1 if rotated[i] >= 0, else 0
          4. Compute norm factor for distance correction
        """
        var rotated = self._rotate(vector)
        # Compute L2 norm of rotated vector
        var norm_sq: Float32 = 0.0
        for d in range(Self.dim):
            norm_sq += rotated[d] * rotated[d]

        var code = RaBitQCode()
        code.norm_factor = sqrt(norm_sq / Float32(Self.dim))

        # Pack bits: 64 dimensions per UInt64 word
        var word_count = Self.words_per_vec
        code.bits.reserve(word_count)
        for _ in range(word_count):
            code.bits.append(0)

        for d in range(Self.dim):
            if rotated[d] >= 0.0:
                var word_idx = d // 64
                var bit_idx = d % 64
                code.bits[word_idx] = code.bits[word_idx] | (UInt64(1) << bit_idx)

        return code^


# ── RaBitQ distance estimation ────────────────────────────────────

def rabitq_hamming_distance[words_per_vec: Int](
    a: RaBitQCode, b: RaBitQCode
) -> Int:
    """Compute Hamming distance between two RaBitQ codes via popcount."""
    var dist = 0
    for i in range(words_per_vec):
        var xor_val = a.bits[i] ^ b.bits[i]
        dist += _popcount_u64(xor_val)
    return dist


def encode_first(mut self, vector: Span[Float32, _]) raises -> RaBitQCode:
    """Encode the first vector — auto-fits the encoder if not already.

    Subsequent vectors should use encode() (non-mutating).
    """
    if not self._fitted:
        self.fit(vector, 1)
        self._fitted = True
    return self.encode(vector)
