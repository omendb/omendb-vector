"""
Raw mmap syscall wrappers for zero-copy file access.

This module uses ONLY external_call — no Python imports. This avoids
the external_call["open"] signature conflict that occurs when Python's
open() is also in scope.

Note on cleanup: munmap is intentionally NOT called in MmapRegion.__del__.
When Python is imported, Mojo's __del__ runs before Python's cleanup, and
munmap can cause segfaults when Python's faulthandler/signal handlers
reference the now-unmapped memory. Since mmap regions are process-lifetime
immutable data, the OS reclaims them on exit.
"""
from std.ffi import external_call, c_int
from std.memory import UnsafePointer


comptime PROT_READ: c_int = 1
comptime MAP_PRIVATE: c_int = 2
comptime O_RDONLY: c_int = 0


@always_inline
def _sys_open(
    path: UnsafePointer[mut=False, UInt8, _], flags: c_int, mode: c_int
) -> c_int:
    return external_call["open", c_int](path, flags, mode)


@always_inline
def _sys_close(fd: c_int) -> c_int:
    return external_call["close", c_int](fd)


@always_inline
def _sys_lseek(fd: c_int, offset: Int, whence: c_int) -> Int:
    return external_call["lseek", Int](fd, offset, whence)


@always_inline
def _sys_mmap(
    length: Int, prot: c_int, flags: c_int, fd: c_int, offset: Int
) -> UnsafePointer[UInt8, MutExternalOrigin]:
    var no_address: Optional[UnsafePointer[UInt8, MutExternalOrigin]] = None
    return external_call[
        "mmap",
        UnsafePointer[UInt8, MutExternalOrigin],
    ](no_address, length, prot, flags, fd, offset)


@always_inline
def _sys_munmap(
    ptr: UnsafePointer[UInt8, MutExternalOrigin], length: Int
) -> c_int:
    return external_call["munmap", c_int](ptr, length)


struct MmapRegion(Movable):
    """RAII wrapper over an mmap'd file region. Read-only, private mapping."""

    var ptr: UnsafePointer[UInt8, MutExternalOrigin]
    var length: Int
    var valid: Bool

    def __init__(out self):
        self.ptr = UnsafePointer[UInt8, MutExternalOrigin].unsafe_dangling()
        self.length = 0
        self.valid = False

    def __del__(deinit self):
        # Intentionally no munmap: process-lifetime data, OS reclaims on exit.
        # Calling munmap here crashes when Python is imported due to cleanup ordering.
        self.valid = False
        self.length = 0

    def as_float32_ptr(self) -> UnsafePointer[Float32, MutExternalOrigin]:
        return self.ptr.bitcast[Float32]()

    def as_uint8_ptr(self) -> UnsafePointer[UInt8, MutExternalOrigin]:
        return self.ptr


def mmap_file_readonly(path: String) raises -> MmapRegion:
    """Open a file and mmap it read-only. Returns region (auto-reclaimed on exit).
    """
    var fd = _sys_open(path.unsafe_ptr(), O_RDONLY, c_int(0))
    if Int(fd) < 0:
        raise Error("mmap: cannot open " + path)

    var file_size = _sys_lseek(fd, 0, c_int(2))  # SEEK_END
    _ = _sys_lseek(fd, 0, c_int(0))  # SEEK_SET

    if file_size <= 0:
        _ = _sys_close(fd)
        raise Error("mmap: empty file " + path)

    var mapped = _sys_mmap(file_size, PROT_READ, MAP_PRIVATE, fd, 0)
    _ = _sys_close(fd)

    if Int(mapped) == -1:
        raise Error("mmap failed for " + path)

    var region = MmapRegion()
    region.ptr = mapped
    region.length = file_size
    region.valid = True

    return region^
