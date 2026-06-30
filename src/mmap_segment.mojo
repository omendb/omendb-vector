"""
Memory-mapped immutable segment for zero-copy reads of persisted HNSW data.

Uses mmap(2) for zero-copy access. The segment file format is:
  - 64-byte header (magic, version, counts, offsets)
  - f32 vectors (num_elements * dim * 4 bytes)
  - SQ8 codes (num_elements * dim bytes)

The segment is verified on open: magic, version, and length must match.
"""
from std.memory import UnsafePointer, Span
from mmap_syscalls import MmapRegion, mmap_file_readonly


# --- Binary format constants ---

comptime SEGMENT_MAGIC: UInt32 = 0x4F4D454E  # "OMEN"
comptime SEGMENT_VERSION: UInt32 = 1
comptime HEADER_SIZE: Int = 64


struct SegmentHeader(Copyable, Movable):
    """Fixed-layout 64-byte header for an immutable segment file."""

    var magic: UInt32
    var version: UInt32
    var num_elements: Int
    var dim: Int
    var vectors_offset: Int
    var vectors_length: Int
    var codes_offset: Int
    var codes_length: Int

    def __init__(
        out self,
        magic: UInt32,
        version: UInt32,
        num_elements: Int,
        dim: Int,
        vectors_offset: Int,
        vectors_length: Int,
        codes_offset: Int,
        codes_length: Int,
    ):
        self.magic = magic
        self.version = version
        self.num_elements = num_elements
        self.dim = dim
        self.vectors_offset = vectors_offset
        self.vectors_length = vectors_length
        self.codes_offset = codes_offset
        self.codes_length = codes_length

    @staticmethod
    def create(num_elements: Int, dim: Int) -> Self:
        var vectors_offset = HEADER_SIZE
        var vectors_length = num_elements * dim * 4
        var codes_offset = vectors_offset + vectors_length
        var codes_length = num_elements * dim
        return Self(
            SEGMENT_MAGIC,
            SEGMENT_VERSION,
            num_elements,
            dim,
            vectors_offset,
            vectors_length,
            codes_offset,
            codes_length,
        )


struct ImmutableSegment(Movable):
    """
    An mmap-backed immutable HNSW segment with typed pointer access.

    Zero-copy reads: vector_ptr(id) and code_ptr(id) return Spans
    pointing directly into the mmap'd file region.
    """

    var region: MmapRegion
    var header: SegmentHeader

    def __init__(out self):
        self.region = MmapRegion()
        self.header = SegmentHeader.create(0, 0)

    def __del__(deinit self):
        pass

    def is_valid(self) -> Bool:
        return self.region.valid and self.header.num_elements > 0

    def vector_ptr(ref self, id: Int) -> Span[Float32, MutAnyOrigin]:
        """Span over the Float32 values for vector `id` (zero-copy into mmap).
        """
        var base = self.region.as_uint8_ptr() + self.header.vectors_offset
        var float_ptr = base.bitcast[Float32]()
        return Span[Float32, MutAnyOrigin](
            ptr=float_ptr + id * self.header.dim, length=self.header.dim
        )

    def code_ptr(ref self, id: Int) -> Span[UInt8, MutAnyOrigin]:
        """Span over the SQ8 code bytes for vector `id` (zero-copy into mmap).
        """
        var base = self.region.as_uint8_ptr() + self.header.codes_offset
        return Span[UInt8, MutAnyOrigin](
            ptr=base + id * self.header.dim, length=self.header.dim
        )

    def num_elements(self) -> Int:
        return self.header.num_elements

    def dim(self) -> Int:
        return self.header.dim


def save_hnsw_immutable_segment(
    path: String,
    data: List[Float32],
    codes: List[UInt8],
    num_elements: Int,
    dim: Int,
) raises:
    """Write HNSW vector data and SQ8 codes as an immutable segment file."""
    if len(data) != num_elements * dim:
        raise Error(
            "data length mismatch: expected "
            + String(num_elements * dim)
            + ", got "
            + String(len(data))
        )
    if len(codes) != num_elements * dim:
        raise Error(
            "codes length mismatch: expected "
            + String(num_elements * dim)
            + ", got "
            + String(len(codes))
        )

    var header = SegmentHeader.create(num_elements, dim)

    with open(path, "w") as f:
        var hdr_buf = List[UInt8](capacity=HEADER_SIZE)
        for _ in range(HEADER_SIZE):
            hdr_buf.append(0)
        _write_u32(hdr_buf, 0, header.magic)
        _write_u32(hdr_buf, 4, header.version)
        _write_int(hdr_buf, 8, header.num_elements)
        _write_int(hdr_buf, 16, header.dim)
        _write_int(hdr_buf, 24, header.vectors_offset)
        _write_int(hdr_buf, 32, header.vectors_length)
        _write_int(hdr_buf, 40, header.codes_offset)
        _write_int(hdr_buf, 48, header.codes_length)
        f.write_all(
            Span(ptr=hdr_buf.unsafe_ptr().bitcast[Byte](), length=HEADER_SIZE)
        )
        f.write_all(
            Span(
                ptr=data.unsafe_ptr().bitcast[Byte](),
                length=header.vectors_length,
            )
        )
        f.write_all(
            Span(
                ptr=codes.unsafe_ptr().bitcast[Byte](),
                length=header.codes_length,
            )
        )


def load_segment_header(path: String) raises -> SegmentHeader:
    """Read and validate the 64-byte header."""
    var b: List[UInt8]
    try:
        with open(path, "r") as f:
            b = f.read_bytes()
    except:
        raise Error("immutable segment file not found: " + path)

    if len(b) < HEADER_SIZE:
        raise Error("immutable segment file too small: " + path)

    var magic = _read_u32(b, 0)
    if magic != SEGMENT_MAGIC:
        raise Error("immutable segment bad magic at " + path)

    var version = _read_u32(b, 4)
    if version != SEGMENT_VERSION:
        raise Error("immutable segment unsupported version: " + String(version))

    return SegmentHeader.create(_read_int(b, 8), _read_int(b, 16))


def open_immutable_segment(path: String) raises -> ImmutableSegment:
    """
    Open an immutable segment file via mmap(2). Zero-copy reads.

    Validates the header, mmaps the file, and returns a segment with
    typed Span-based access into the mmap'd region.
    """
    var header = load_segment_header(path)
    if header.num_elements <= 0:
        raise Error("immutable segment has no elements")
    if header.dim <= 0:
        raise Error("immutable segment has no dimension")

    var expected_size = (
        header.vectors_offset + header.vectors_length + header.codes_length
    )

    # mmap the entire file read-only
    var region = mmap_file_readonly(path)

    if region.length < expected_size:
        raise Error(
            "immutable segment truncated: expected "
            + String(expected_size)
            + " bytes, got "
            + String(region.length)
        )

    var segment = ImmutableSegment()
    segment.region = region^
    segment.header = header.copy()

    return segment^


# --- Little-endian integer I/O helpers ---


def _write_u32(mut buf: List[UInt8], offset: Int, val: UInt32):
    var v = val
    for i in range(4):
        buf[offset + i] = UInt8(v & 0xFF)
        v >>= 8


def _write_int(mut buf: List[UInt8], offset: Int, val: Int):
    var v = UInt64(val)
    for i in range(8):
        buf[offset + i] = UInt8(v & 0xFF)
        v >>= 8


def _read_u32(ref buf: List[UInt8], offset: Int) -> UInt32:
    var result: UInt32 = 0
    for i in range(4):
        result |= UInt32(buf[offset + i]) << UInt32(i * 8)
    return result


def _read_int(ref buf: List[UInt8], offset: Int) -> Int:
    var result: UInt64 = 0
    for i in range(8):
        result |= UInt64(buf[offset + i]) << UInt64(i * 8)
    return Int(result)
