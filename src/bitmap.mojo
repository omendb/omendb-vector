"""Bitmap — plain bitset for metadata filter evaluation.

Uses List[UInt64] words. Supports set/test/clear/intersect/union/complement/popcount.
Sufficient for ~10M docs (125KB per bitmap at 1M). No Roaring — keep it simple.
"""


struct Bitmap(Movable, Copyable):
    var bits: List[UInt64]
    var size: Int  # number of bits

    def __init__(out self, size: Int):
        self.size = size
        var words = (size + 63) // 64
        self.bits = List[UInt64](capacity=words)
        for _ in range(words):
            self.bits.append(0)

    def __init__(out self, *, deinit take: Self):
        self.size = take.size
        self.bits = take.bits^

    def set(mut self, idx: Int):
        if idx < 0 or idx >= self.size:
            return
        self.bits[idx // 64] |= (UInt64(1) << UInt64(idx % 64))

    def clear(mut self, idx: Int):
        if idx < 0 or idx >= self.size:
            return
        self.bits[idx // 64] &= ~(UInt64(1) << UInt64(idx % 64))

    def test(ref self, idx: Int) -> Bool:
        if idx < 0 or idx >= self.size:
            return False
        return (self.bits[idx // 64] & (UInt64(1) << UInt64(idx % 64))) != 0

    def popcount(ref self) -> Int:
        var count = 0
        for i in range(len(self.bits)):
            count += _popcount_u64(self.bits[i])
        return count

    def resized(ref self, new_size: Int) -> Bitmap:
        """Create a new bitmap with new_size, copying existing words."""
        var cap = max(new_size, self.size)
        var result = Bitmap(cap)
        # Copy existing words (word-level, not bit-level)
        var old_words = len(self.bits)
        for i in range(old_words):
            result.bits[i] = self.bits[i]
        return result^

    def intersect(ref self, ref other: Bitmap) -> Bitmap:
        var min_words = min(len(self.bits), len(other.bits))
        var result = Bitmap(self.size)
        for i in range(min_words):
            result.bits[i] = self.bits[i] & other.bits[i]
        return result^

    def union(ref self, ref other: Bitmap) -> Bitmap:
        var max_words = max(len(self.bits), len(other.bits))
        var result = Bitmap(max(self.size, other.size))
        for i in range(max_words):
            var a = self.bits[i] if i < len(self.bits) else UInt64(0)
            var b = other.bits[i] if i < len(other.bits) else UInt64(0)
            result.bits[i] = a | b
        return result^

    def complement(ref self) -> Bitmap:
        var result = Bitmap(self.size)
        var full_words = self.size // 64
        var remainder = self.size % 64
        for i in range(full_words):
            result.bits[i] = ~self.bits[i]
        if remainder > 0 and full_words < len(self.bits):
            var mask = (UInt64(1) << UInt64(remainder)) - 1
            result.bits[full_words] = ~self.bits[full_words] & mask
        return result^

    def is_empty(ref self) -> Bool:
        for i in range(len(self.bits)):
            if self.bits[i] != 0:
                return False
        return True

    def and_not(ref self, ref other: Bitmap) -> Bitmap:
        """Bits in self that are not in other."""
        var min_words = min(len(self.bits), len(other.bits))
        var result = Bitmap(self.size)
        for i in range(min_words):
            result.bits[i] = self.bits[i] & ~other.bits[i]
        for i in range(min_words, len(self.bits)):
            result.bits[i] = self.bits[i]
        return result^

    def to_bytes(ref self) -> List[UInt8]:
        """Convert to byte array for HNSW interface compatibility."""
        var result = List[UInt8](capacity=self.size)
        for i in range(self.size):
            if self.test(i):
                result.append(1)
            else:
                result.append(0)
        return result^

    @staticmethod
    def from_bytes(ref bytes: List[UInt8]) -> Bitmap:
        """Create bitmap from byte array (HNSW allowlist format)."""
        var bm = Bitmap(len(bytes))
        for i in range(len(bytes)):
            if bytes[i] != 0:
                bm.set(i)
        return bm^




def _popcount_u64(x: UInt64) -> Int:
    """Population count (number of set bits) for a UInt64."""
    var v = x
    v -= (v >> 1) & UInt64(0x5555555555555555)
    v = (v & UInt64(0x3333333333333333)) + ((v >> 2) & UInt64(0x3333333333333333))
    v = (v + (v >> 4)) & UInt64(0x0F0F0F0F0F0F0F0F)
    return Int((v * UInt64(0x0101010101010101)) >> 56)


def test_bitmap():
    print("=== Bitmap tests ===")
    var bm = Bitmap(1000)
    assert not bm.test(0)
    bm.set(0)
    assert bm.test(0)
    bm.set(999)
    assert bm.test(999)
    assert bm.popcount() == 2
    bm.clear(0)
    assert not bm.test(0)
    assert bm.popcount() == 1

    var a = Bitmap(256)
    var b = Bitmap(256)
    a.set(0)
    a.set(1)
    a.set(100)
    b.set(1)
    b.set(100)
    b.set(200)
    var c = a.intersect(b)
    assert c.popcount() == 2
    assert c.test(1)
    assert c.test(100)
    assert not c.test(0)

    var d = a.union(b)
    assert d.popcount() == 4

    var small = Bitmap(64)
    small.set(0)
    var comp = small.complement()
    assert not comp.test(0)
    assert comp.test(1)
    assert comp.test(63)
    assert comp.popcount() == 63

    var large = Bitmap(100000)
    for i in range(0, 100000, 2):
        large.set(i)
    assert large.popcount() == 50000

    print("All Bitmap tests passed!")


def main():
    test_bitmap()
