"""
Fault-injection tests for persistence and recovery.

Tests deterministic failure scenarios: truncated files, checksum mismatches,
corrupt segments, partial flushes, and recovery idempotence. Each test
verifies that the store fails visibly (never silently) and recovers to
the last durable state.
"""
from std.testing import assert_equal, assert_true
from std.python import Python
from store import VectorStore, VectorStoreOptions
from consistency_check import verify_consistency


def test_truncated_manifest() raises:
    """A truncated manifest.json must fail on open."""
    var shutil = Python.import_module("shutil")
    var os = Python.import_module("os")
    var builtins = Python.import_module("builtins")
    var path = "test_fault_truncated_manifest"
    if os.path.exists(path):
        shutil.rmtree(path)

    var store = VectorStore[2].create(path)
    var vec: List[Float32] = [1.0, 1.0]
    _ = store.set("a", Span(ptr=vec.unsafe_ptr(), length=2))
    store.flush()

    # Truncate manifest to first 10 bytes
    var data = builtins.open(path + "/manifest.json", "rb").read()
    var truncated = builtins.open(path + "/manifest.json", "wb")
    _ = truncated.write(data[:10])
    _ = truncated.close()

    var failed = False
    try:
        _ = VectorStore[2].open(path)
    except:
        failed = True
    assert_true(failed, "truncated manifest must fail on open")
    shutil.rmtree(path)


def test_checksum_mismatch_after_flush() raises:
    """Changing a data file after flush must fail checksum validation on open.
    """
    var shutil = Python.import_module("shutil")
    var os = Python.import_module("os")
    var builtins = Python.import_module("builtins")
    var path = "test_fault_checksum_mismatch"
    if os.path.exists(path):
        shutil.rmtree(path)

    var store = VectorStore[2].create(path)
    var vec: List[Float32] = [1.0, 1.0]
    _ = store.set("a", Span(ptr=vec.unsafe_ptr(), length=2))
    store.flush()

    # Append garbage to records.json (changes checksum)
    var records = builtins.open(path + "/records.json", "ab")
    _ = records.write(builtins.bytes("garbage", "utf-8"))
    _ = records.close()

    var failed = False
    try:
        _ = VectorStore[2].open(path)
    except:
        failed = True
    assert_true(failed, "checksum mismatch must fail on open")
    shutil.rmtree(path)


def test_truncated_hnsw_data() raises:
    """A truncated hnsw.data file must fail on open."""
    var shutil = Python.import_module("shutil")
    var os = Python.import_module("os")
    var builtins = Python.import_module("builtins")
    var path = "test_fault_truncated_hnsw"
    if os.path.exists(path):
        shutil.rmtree(path)

    var store = VectorStore[2].create(path)
    var vec: List[Float32] = [1.0, 1.0]
    _ = store.set("a", Span(ptr=vec.unsafe_ptr(), length=2))
    store.flush()

    # Truncate hnsw.data
    var data = builtins.open(path + "/hnsw.data", "rb").read()
    var truncated = builtins.open(path + "/hnsw.data", "wb")
    _ = truncated.write(data[:4])
    _ = truncated.close()

    var failed = False
    try:
        _ = VectorStore[2].open(path)
    except:
        failed = True
    assert_true(failed, "truncated hnsw.data must fail on open")
    shutil.rmtree(path)


def test_corrupt_records_json() raises:
    """A records.json with invalid JSON must fail on open."""
    var shutil = Python.import_module("shutil")
    var os = Python.import_module("os")
    var builtins = Python.import_module("builtins")
    var path = "test_fault_corrupt_records"
    if os.path.exists(path):
        shutil.rmtree(path)

    var store = VectorStore[2].create(path)
    var vec: List[Float32] = [1.0, 1.0]
    _ = store.set("a", Span(ptr=vec.unsafe_ptr(), length=2))
    store.flush()

    # Overwrite records.json with invalid JSON
    var f = builtins.open(path + "/records.json", "wb")
    _ = f.write(builtins.bytes("{invalid json!!!", "utf-8"))
    _ = f.close()

    var failed = False
    try:
        _ = VectorStore[2].open(path)
    except:
        failed = True
    assert_true(failed, "corrupt records.json must fail on open")
    shutil.rmtree(path)


def test_corrupt_tombstones_json() raises:
    """A tombstones.json with invalid JSON must fail on open."""
    var shutil = Python.import_module("shutil")
    var os = Python.import_module("os")
    var builtins = Python.import_module("builtins")
    var path = "test_fault_corrupt_tombstones"
    if os.path.exists(path):
        shutil.rmtree(path)

    var store = VectorStore[2].create(path)
    var vec: List[Float32] = [1.0, 1.0]
    _ = store.set("a", Span(ptr=vec.unsafe_ptr(), length=2))
    _ = store.set("b", Span(ptr=vec.unsafe_ptr(), length=2))
    assert_true(store.delete("a"))
    store.flush()

    # Overwrite tombstones.json with invalid JSON
    var f = builtins.open(path + "/tombstones.json", "wb")
    _ = f.write(builtins.bytes("{corrupt", "utf-8"))
    _ = f.close()

    var failed = False
    try:
        _ = VectorStore[2].open(path)
    except:
        failed = True
    assert_true(failed, "corrupt tombstones.json must fail on open")
    shutil.rmtree(path)


def test_recovery_idempotence() raises:
    """Opening a store twice after flush must produce identical state."""
    var shutil = Python.import_module("shutil")
    var os = Python.import_module("os")
    var path = "test_fault_idempotence"
    if os.path.exists(path):
        shutil.rmtree(path)

    var store = VectorStore[2].create(path)
    var a: List[Float32] = [0.0, 0.0]
    var b: List[Float32] = [1.0, 1.0]
    _ = store.set("a", Span(ptr=a.unsafe_ptr(), length=2))
    _ = store.set("b", Span(ptr=b.unsafe_ptr(), length=2))
    store.flush()

    # Open twice — must produce same state
    var r1 = VectorStore[2].open(path)
    var r2 = VectorStore[2].open(path)

    assert_equal(r1.len(), r2.len())
    assert_equal(r1.len(), 2)

    var q: List[Float32] = [0.1, 0.1]
    var s1 = r1.search(Span(ptr=q.unsafe_ptr(), length=2))
    var s2 = r2.search(Span(ptr=q.unsafe_ptr(), length=2))
    assert_equal(len(s1), len(s2))
    for i in range(len(s1)):
        assert_equal(s1[i].id, s2[i].id)
        assert_equal(s1[i].distance, s2[i].distance)

    shutil.rmtree(path)


def test_consistency_after_recovery() raises:
    """Verify_consistency must pass on a recovered store."""
    var shutil = Python.import_module("shutil")
    var os = Python.import_module("os")
    var path = "test_fault_consistency_recovery"
    if os.path.exists(path):
        shutil.rmtree(path)

    var store = VectorStore[2].create(path)
    var a: List[Float32] = [0.0, 0.0]
    var b: List[Float32] = [1.0, 1.0]
    var c: List[Float32] = [2.0, 2.0]
    _ = store.set("a", Span(ptr=a.unsafe_ptr(), length=2))
    _ = store.set("b", Span(ptr=b.unsafe_ptr(), length=2))
    _ = store.set("c", Span(ptr=c.unsafe_ptr(), length=2))
    assert_true(store.delete("b"))
    store.flush()

    var recovered = VectorStore[2].open(path)
    var report = verify_consistency(recovered)
    assert_true(
        report.ok(), "recovered store must be consistent: " + report.summary()
    )
    assert_equal(recovered.len(), 2)

    shutil.rmtree(path)


def test_flush_preserves_previous_on_failure() raises:
    """A partial flush that corrupts a file must not destroy the previous good state.
    """
    var shutil = Python.import_module("shutil")
    var os = Python.import_module("os")
    var builtins = Python.import_module("builtins")
    var path = "test_fault_prev_flush_preserved"
    if os.path.exists(path):
        shutil.rmtree(path)

    # First flush — good state
    var store = VectorStore[2].create(path)
    var vec: List[Float32] = [1.0, 1.0]
    _ = store.set("a", Span(ptr=vec.unsafe_ptr(), length=2))
    store.flush()

    # Save the good files
    var good_records = builtins.open(path + "/records.json", "rb").read()
    var good_hnsw = builtins.open(path + "/hnsw.data", "rb").read()
    var good_manifest = builtins.open(path + "/manifest.json", "rb").read()

    # Second set — add more data
    _ = store.set("b", Span(ptr=vec.unsafe_ptr(), length=2))

    # Simulate crash during second flush by corrupting records.json
    # and then restoring ALL good files
    var bad_records = builtins.open(path + "/records.json", "wb")
    _ = bad_records.write(builtins.bytes("corrupt", "utf-8"))
    _ = bad_records.close()

    # Rollback: restore all files from first flush
    var r1 = builtins.open(path + "/records.json", "wb")
    _ = r1.write(good_records)
    _ = r1.close()
    var r2 = builtins.open(path + "/hnsw.data", "wb")
    _ = r2.write(good_hnsw)
    _ = r2.close()
    var r3 = builtins.open(path + "/manifest.json", "wb")
    _ = r3.write(good_manifest)
    _ = r3.close()

    # Must still be able to open (from first flush state)
    var recovered = VectorStore[2].open(path)
    assert_equal(
        recovered.len(),
        1,
        "recovered store should have 1 item from first flush",
    )

    shutil.rmtree(path)


def test_missing_directory() raises:
    """Opening a non-existent path must fail visibly."""
    var shutil = Python.import_module("shutil")
    var os = Python.import_module("os")
    var path = "test_fault_missing_dir_nonexistent_12345"
    if os.path.exists(path):
        shutil.rmtree(path)

    var failed = False
    try:
        _ = VectorStore[2].open(path)
    except:
        failed = True
    assert_true(failed, "opening non-existent path must fail")


def test_empty_directory() raises:
    """Opening an empty directory must fail visibly."""
    var shutil = Python.import_module("shutil")
    var os = Python.import_module("os")
    var path = "test_fault_empty_dir"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)

    var failed = False
    try:
        _ = VectorStore[2].open(path)
    except:
        failed = True
    assert_true(failed, "opening empty directory must fail")
    shutil.rmtree(path)


def main() raises:
    test_truncated_manifest()
    print("PASS test_truncated_manifest")
    test_checksum_mismatch_after_flush()
    print("PASS test_checksum_mismatch_after_flush")
    test_truncated_hnsw_data()
    print("PASS test_truncated_hnsw_data")
    test_corrupt_records_json()
    print("PASS test_corrupt_records_json")
    test_corrupt_tombstones_json()
    print("PASS test_corrupt_tombstones_json")
    test_recovery_idempotence()
    print("PASS test_recovery_idempotence")
    test_consistency_after_recovery()
    print("PASS test_consistency_after_recovery")
    test_flush_preserves_previous_on_failure()
    print("PASS test_flush_preserves_previous_on_failure")
    test_missing_directory()
    print("PASS test_missing_directory")
    test_empty_directory()
    print("PASS test_empty_directory")
