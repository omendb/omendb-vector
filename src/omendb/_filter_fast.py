"""Fast C-based metadata filter evaluation.

Uses cJSON for JSON parsing in C, avoiding Python dict access overhead.
"""

import ctypes
import json
from pathlib import Path
from typing import Any

_lib = None

def _get_lib():
    global _lib
    if _lib is None:
        lib_path = Path(__file__).parent / "libfilterfast.dylib"
        if not lib_path.exists():
            lib_path = Path(__file__).parent / "libfilterfast.so"
        if not lib_path.exists():
            raise RuntimeError(f"libfilterfast not found at {lib_path}")
        _lib = ctypes.CDLL(str(lib_path))
        # int filter_batch_fast(
        #   const char **metadata_jsons, int num_items, const int *deleted,
        #   const char *filter_json, int *out_indices, int max_out)
        _lib.filter_batch_fast.argtypes = [
            ctypes.POINTER(ctypes.c_char_p),  # metadata_jsons
            ctypes.c_int,                      # num_items
            ctypes.POINTER(ctypes.c_int),      # deleted
            ctypes.c_char_p,                   # filter_json
            ctypes.POINTER(ctypes.c_int),      # out_indices
            ctypes.c_int,                      # max_out
        ]
        _lib.filter_batch_fast.restype = ctypes.c_int
    return _lib


def filter_ids_c(
    ids: list[str],
    metadata_jsons: list[str | None],
    deleted: list[bool],
    filter_dict: dict[str, Any],
) -> list[str]:
    """Evaluate filter in C, return matching IDs."""
    lib = _get_lib()
    n = len(ids)

    # Build C arrays
    meta_arr = (ctypes.c_char_p * n)()
    del_arr = (ctypes.c_int * n)()
    for i in range(n):
        meta_arr[i] = metadata_jsons[i].encode("utf-8") if metadata_jsons[i] else None
        del_arr[i] = 1 if deleted[i] else 0

    filter_json = json.dumps(filter_dict).encode("utf-8")

    # Output array
    max_out = n
    out_arr = (ctypes.c_int * max_out)()

    count = lib.filter_batch_fast(
        meta_arr, n, del_arr, filter_json, out_arr, max_out
    )

    if count < 0:
        raise ValueError("filter_batch_fast failed")

    return [ids[out_arr[i]] for i in range(count)]
