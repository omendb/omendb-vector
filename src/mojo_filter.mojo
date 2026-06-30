"""Native filter matching using single Python call.

Passes raw Mojo data to Python for minimal Mojo↔Python overhead.
Python builds the lists itself (fast) and evaluates the filter.
"""

from std.python import Python, PythonObject


def filter_ids_native(
    ids: PythonObject,
    metadata: List[Optional[String]],
    deleted: List[Bool],
    filter_obj: PythonObject,
) raises -> PythonObject:
    """Evaluate filter — delegates to Python for minimal overhead."""
    var py_none = Python.none()
    var parsed_list = Python.list()

    # Use orjson for 4x faster JSON parsing, fall back to json
    var use_orjson = False
    var json_mod = py_none
    var orjson_mod = py_none
    try:
        orjson_mod = Python.import_module("orjson")
        use_orjson = True
    except:
        json_mod = Python.import_module("json")

    # Build Python list of parsed metadata in one pass
    for i in range(len(metadata)):
        if deleted[i]:
            parsed_list.append(py_none)
        elif metadata[i]:
            if use_orjson:
                parsed_list.append(orjson_mod.loads(metadata[i].value()))
            else:
                parsed_list.append(json_mod.loads(metadata[i].value()))
        else:
            parsed_list.append(py_none)

    # Run filter in Python — single boundary crossing
    var result = Python.list()
    for i in range(len(ids)):
        var parsed = parsed_list[i]
        if parsed is py_none:
            continue

        var matches = True
        for key_py in filter_obj:
            var expected = filter_obj[key_py]
            var expected_type = String(py=expected.__class__.__name__)

            if expected_type == "dict":
                var exists = Bool(py=parsed.__contains__(key_py))
                var actual = parsed[key_py] if exists else py_none

                for op_py in expected:
                    var op = String(py=op_py)
                    var val = expected[op_py]

                    if op == "$exists":
                        if exists != Bool(py=val):
                            matches = False
                            break
                    elif op == "$in":
                        if not exists or not Bool(py=val.__contains__(actual)):
                            matches = False
                            break
                    elif op == "$gt":
                        if not exists or not (actual > val):
                            matches = False
                            break
                    elif op == "$lt":
                        if not exists or not (actual < val):
                            matches = False
                            break
                    elif op == "$gte":
                        if not exists or not (actual >= val):
                            matches = False
                            break
                    elif op == "$lte":
                        if not exists or not (actual <= val):
                            matches = False
                            break
            else:
                var exists = Bool(py=parsed.__contains__(key_py))
                if not exists or parsed[key_py] != expected:
                    matches = False

            if not matches:
                break

        if matches:
            result.append(ids[i])
    return result


def filter_ids_batch(
    external_ids: PythonObject,
    metadata_strings: PythonObject,
    deleted_flags: PythonObject,
    filter_obj: PythonObject,
) raises -> PythonObject:
    """Evaluate filter with pre-built Python lists (no Mojo list building)."""
    var py_none = Python.none()
    var parsed_list = Python.list()

    # Use orjson for 4x faster JSON parsing, fall back to json
    var use_orjson = False
    var json_mod = py_none
    var orjson_mod = py_none
    try:
        orjson_mod = Python.import_module("orjson")
        use_orjson = True
    except:
        json_mod = Python.import_module("json")

    # Build Python list of parsed metadata in one pass
    var n = len(metadata_strings)
    for i in range(n):
        if Bool(py=deleted_flags[i]):
            parsed_list.append(py_none)
        elif metadata_strings[i] is not py_none:
            if use_orjson:
                parsed_list.append(orjson_mod.loads(metadata_strings[i]))
            else:
                parsed_list.append(json_mod.loads(metadata_strings[i]))
        else:
            parsed_list.append(py_none)

    # Run filter in Python — single boundary crossing
    var result = Python.list()
    for i in range(n):
        var parsed = parsed_list[i]
        if parsed is py_none:
            continue

        var matches = True
        for key_py in filter_obj:
            var expected = filter_obj[key_py]
            var expected_type = String(py=expected.__class__.__name__)

            if expected_type == "dict":
                var exists = Bool(py=parsed.__contains__(key_py))
                var actual = parsed[key_py] if exists else py_none

                for op_py in expected:
                    var op = String(py=op_py)
                    var val = expected[op_py]

                    if op == "$exists":
                        if exists != Bool(py=val):
                            matches = False
                            break
                    elif op == "$in":
                        if not exists or not Bool(py=val.__contains__(actual)):
                            matches = False
                            break
                    elif op == "$gt":
                        if not exists or not (actual > val):
                            matches = False
                            break
                    elif op == "$lt":
                        if not exists or not (actual < val):
                            matches = False
                            break
                    elif op == "$gte":
                        if not exists or not (actual >= val):
                            matches = False
                            break
                    elif op == "$lte":
                        if not exists or not (actual <= val):
                            matches = False
                            break
            else:
                var exists = Bool(py=parsed.__contains__(key_py))
                if not exists or parsed[key_py] != expected:
                    matches = False

            if not matches:
                break

        if matches:
            result.append(external_ids[i])
    return result
