"""
Test that mmap_syscalls works and can coexist with Python imports.
"""
from std.testing import assert_equal, assert_true
from std.ffi import c_int
from mmap_syscalls import (
    MmapRegion,
    mmap_file_readonly,
    _sys_open,
    _sys_close,
    O_RDONLY,
)
from std.python import Python


def test_mmap_syscalls_basic() raises:
    """Open a file via syscall and mmap it."""
    var path = "/tmp/omendb_mmap_syscalls_test.bin"

    # Create test file with known content
    var data = List[UInt8]()
    for i in range(256):
        data.append(UInt8(i))

    with open(path, "w") as f:
        f.write_all(Span(ptr=data.unsafe_ptr().bitcast[Byte](), length=256))

    # mmap the file
    var region = mmap_file_readonly(path)
    assert_true(region.valid, "region should be valid")
    assert_equal(region.length, 256)

    # Read bytes via the mmap pointer
    var ptr = region.as_uint8_ptr()
    assert_equal(ptr[0], 0)
    assert_equal(ptr[128], 128)
    assert_equal(ptr[255], 255)

    # Read as float32 (reinterpret first 4 bytes: 0,1,2,3 → little-endian float)
    var f32_ptr = region.as_float32_ptr()
    var v0 = f32_ptr[0]
    # Just verify we can read it without crashing
    print("f32[0] = " + String(v0))


def test_mmap_with_python_coexistence() raises:
    """Verify mmap_syscalls works even when Python is imported."""
    var path = "/tmp/omendb_mmap_syscalls_test2.bin"

    # Create test file
    var data = List[UInt8]()
    for i in range(64):
        data.append(UInt8(i))

    with open(path, "w") as f:
        f.write_all(Span(ptr=data.unsafe_ptr().bitcast[Byte](), length=64))

    # Use Python (this is what triggers the open() conflict)
    var os = Python.import_module("os")
    var file_exists = Bool(py=os.path.exists(path))
    assert_true(file_exists, "file should exist per Python os.path")

    # Now mmap it — this is the path that was failing before
    var region = mmap_file_readonly(path)
    assert_true(region.valid)
    assert_equal(region.length, 64)
    assert_equal(region.as_uint8_ptr()[0], 0)
    assert_equal(region.as_uint8_ptr()[63], 63)


def main() raises:
    test_mmap_syscalls_basic()
    print("PASS test_mmap_syscalls_basic")
    test_mmap_with_python_coexistence()
    print("PASS test_mmap_with_python_coexistence")
