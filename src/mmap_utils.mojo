from std.ffi import external_call, c_int
from std.memory import UnsafePointer

# Low-level mmap wrappers for future disk-backed/zero-copy storage modes.
# Current P2 persistence intentionally uses native buffered List load.

comptime PROT_READ: c_int = 1
comptime PROT_WRITE: c_int = 2
comptime MAP_SHARED: c_int = 1
comptime MAP_PRIVATE: c_int = 2

comptime O_RDONLY: c_int = 0
comptime O_RDWR: c_int = 2
comptime O_CREAT: c_int = 0x0200  # macOS
comptime O_TRUNC: c_int = 0x0400  # macOS


@always_inline
def sys_mmap(
    length: Int, prot: c_int, flags: c_int, fd: c_int, offset: Int
) -> UnsafePointer[UInt8, MutExternalOrigin]:
    var no_address: Optional[UnsafePointer[UInt8, MutExternalOrigin]] = None
    return external_call[
        "mmap",
        UnsafePointer[UInt8, MutExternalOrigin],
    ](no_address, length, prot, flags, fd, offset)


@always_inline
def sys_munmap(
    ptr: UnsafePointer[UInt8, MutExternalOrigin], length: Int
) -> c_int:
    return external_call["munmap", c_int](ptr, length)


@always_inline
def sys_close(fd: c_int) -> c_int:
    return external_call["close", c_int](fd)


@always_inline
def sys_lseek(fd: c_int, offset: Int, whence: Int) -> Int:
    return external_call["lseek", Int](fd, offset, whence)


@always_inline
def sys_ftruncate(fd: c_int, length: Int) -> c_int:
    return external_call["ftruncate", c_int](fd, length)
