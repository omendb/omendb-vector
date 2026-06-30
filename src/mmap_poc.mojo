from std.ffi import external_call, c_int
from std.memory import UnsafePointer


comptime PROT_READ: c_int = 1
comptime MAP_PRIVATE: c_int = 2
comptime O_RDONLY: c_int = 0


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
def sys_open(path: UnsafePointer[mut=False, UInt8, _], flags: c_int) -> c_int:
    return external_call["open", c_int](path, flags)


@always_inline
def sys_close(fd: c_int) -> c_int:
    return external_call["close", c_int](fd)


@always_inline
def sys_lseek(fd: c_int, offset: Int, whence: Int) -> Int:
    return external_call["lseek", Int](fd, offset, whence)


def main() raises:
    var test_path = "/tmp/omendb_mmap_test.bin"

    var fd = sys_open(test_path.unsafe_ptr(), O_RDONLY)
    if fd < 0:
        print("FAIL: could not open", test_path)
        return

    var file_size = sys_lseek(fd, 0, 2)
    if file_size <= 0:
        print("FAIL: could not determine file size")
        _ = sys_close(fd)
        return
    _ = sys_lseek(fd, 0, 0)

    print("File size:", file_size)

    var mapped = sys_mmap(file_size, PROT_READ, MAP_PRIVATE, fd, 0)
    _ = sys_close(fd)

    if Int(mapped) == -1:
        print("FAIL: mmap returned MAP_FAILED")
        return

    var first = Int(mapped[0])
    var mid = Int(mapped[file_size // 2])
    var last = Int(mapped[file_size - 1])
    print("byte[0]:", first)
    print("byte[128]:", mid)
    print("byte[255]:", last)

    _ = sys_munmap(mapped, file_size)

    if first == 0 and mid == 128 and last == 255:
        print("PASS: mmap proof-of-concept complete")
    else:
        print("WARN: unexpected byte values (but mmap worked)")
