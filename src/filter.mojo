"""Filter expression parser and bitmap evaluator.

Port of Rust filter.rs. Parses Python filter dicts into Filter structs,
evaluates them against MetadataIndex to produce Bitmaps.

Supports: $eq, $ne, $gt, $gte, $lt, $lte, $in, $and, $or, $not, $exists.
"""

from bitmap import Bitmap
from metadata_index import MetadataIndex
from std.python import Python, PythonObject


# === Filter expression tree ===

struct FilterExpr(Movable, Copyable):
    """Filter expression tree node."""
    var op: Int  # 0=eq, 1=ne, 2=gt, 3=gte, 4=lt, 5=lte, 6=in, 7=exists, 8=and, 9=or, 10=not
    var field: String
    var value: PythonObject  # for scalar predicates
    var values: List[PythonObject]  # for $in
    var children: List[FilterExpr]  # for $and/$or/$not

    def __init__(out self):
        self.op = 0
        self.field = ""
        self.value = Python.none()
        self.values = List[PythonObject]()
        self.children = List[FilterExpr]()

    def __init__(out self, *, deinit take: Self):
        self.op = take.op
        self.field = take.field^
        self.value = take.value
        self.values = take.values^
        self.children = take.children^


# === Parser ===

def _parse_filter(filter_dict: PythonObject) raises -> FilterExpr:
    """Parse a Python filter dict into a FilterExpr tree.

    Supported syntax:
        {"field": value}                  -> Eq
        {"field": {"$eq": value}}         -> Eq
        {"field": {"$ne": value}}         -> Ne
        {"field": {"$gt": number}}        -> Gt
        {"field": {"$gte": number}}       -> Gte
        {"field": {"$lt": number}}        -> Lt
        {"field": {"$lte": number}}       -> Lte
        {"field": {"$in": [values]}}      -> In
        {"field": {"$exists": bool}}      -> Exists
        {"$and": [filters]}               -> And
        {"$or": [filters]}                -> Or
        {"$not": filter}                  -> Not
    """
    var py = Python.import_module("builtins")
    var expr = FilterExpr()

    # Handle $and/$or/$not at top level
    if py.len(filter_dict) == 1:
        var first_key = py.list(filter_dict)[0]
        var key_str = String(first_key)

        if key_str == "$and":
            expr.op = 8
            var items = filter_dict[first_key]
            var n = len(items)
            for i in range(n):
                expr.children.append(_parse_filter(items[i]))
            return expr^

        if key_str == "$or":
            expr.op = 9
            var items = filter_dict[first_key]
            var n = len(items)
            for i in range(n):
                expr.children.append(_parse_filter(items[i]))
            return expr^

        if key_str == "$not":
            expr.op = 10
            expr.children.append(_parse_filter(filter_dict[first_key]))
            return expr^

    # Handle field-level predicates: implicit AND for multiple fields
    var keys = py.list(filter_dict)
    var n = len(keys)

    # If multiple fields, create implicit AND
    if n > 1:
        expr.op = 8  # And
        for i in range(n):
            var field_key = keys[i]
            var single = FilterExpr()
            single.op = 0
            var field_str = String(field_key)
            var val = filter_dict[field_key]
            if not py.isinstance(val, py.dict):
                single.field = field_str
                single.value = val
                expr.children.append(single^)
            else:
                var op_keys = py.list(val)
                var op_n = len(op_keys)
                for j in range(op_n):
                    var op_key = op_keys[j]
                    var op_str = String(op_key)
                    var op_val = val[op_key]
                    var child = FilterExpr()
                    child.field = field_str
                    if op_str == "$eq":
                        child.op = 0
                    elif op_str == "$ne":
                        child.op = 1
                    elif op_str == "$gt":
                        child.op = 2
                    elif op_str == "$gte":
                        child.op = 3
                    elif op_str == "$lt":
                        child.op = 4
                    elif op_str == "$lte":
                        child.op = 5
                    elif op_str == "$in":
                        child.op = 6
                    elif op_str == "$exists":
                        child.op = 7
                    child.value = op_val
                    expr.children.append(child^)
        return expr^

    # Single field case
    for i in range(n):
        var field_key = keys[i]
        var field_str = String(field_key)
        var val = filter_dict[field_key]

        # Simple equality: {"field": value}
        if not py.isinstance(val, py.dict):
            expr.op = 0
            expr.field = field_str
            expr.value = val
            return expr^

        # Operator dict: {"field": {"$op": value}}
        var op_keys = py.list(val)
        var op_n = len(op_keys)

        # Single op — return direct
        if op_n == 1:
            var op_key = op_keys[0]
            var op_str = String(op_key)
            var op_val = val[op_key]
            if op_str == "$eq":
                expr.op = 0
            elif op_str == "$ne":
                expr.op = 1
            elif op_str == "$gt":
                expr.op = 2
            elif op_str == "$gte":
                expr.op = 3
            elif op_str == "$lt":
                expr.op = 4
            elif op_str == "$lte":
                expr.op = 5
            elif op_str == "$in":
                expr.op = 6
                var items = op_val
                var m = len(items)
                for k in range(m):
                    expr.values.append(items[k])
                expr.field = field_str
                return expr^
            elif op_str == "$exists":
                expr.op = 7
            expr.field = field_str
            expr.value = op_val
            return expr^

        # Multiple ops on same field — implicit AND
        expr.op = 8  # And
        for j in range(op_n):
            var op_key = op_keys[j]
            var op_str = String(op_key)
            var op_val = val[op_key]
            var child = FilterExpr()
            child.field = field_str
            if op_str == "$eq":
                child.op = 0
            elif op_str == "$ne":
                child.op = 1
            elif op_str == "$gt":
                child.op = 2
            elif op_str == "$gte":
                child.op = 3
            elif op_str == "$lt":
                child.op = 4
            elif op_str == "$lte":
                child.op = 5
            elif op_str == "$in":
                child.op = 6
                var items = op_val
                var m = len(items)
                for k in range(m):
                    child.values.append(items[k])
                expr.children.append(child^)
                continue
            elif op_str == "$exists":
                child.op = 7
            child.value = op_val
            expr.children.append(child^)
        return expr^

    return expr^


# === Bitmap evaluation ===

def _evaluate_bitmap(
    ref expr: FilterExpr,
    ref index: MetadataIndex,
    num_docs: Int,
) raises -> Optional[Bitmap]:
    """Evaluate a filter expression against a MetadataIndex. Returns bitmap or None."""
    var py = Python.import_module("builtins")

    if expr.op == 0:  # $eq
        return index.evaluate_eq(expr.field, expr.value)

    elif expr.op == 1:  # $ne
        var eq_bm = index.evaluate_eq(expr.field, expr.value)
        if Bool(eq_bm):
            var universe = Bitmap(num_docs)
            for i in range(num_docs):
                universe.set(i)
            return universe.and_not(eq_bm.value())
        return None

    elif expr.op == 2:  # $gt
        var threshold = Float64(py=py.float(expr.value))
        return index.evaluate_range(expr.field, threshold + 1e-15, 1e308)

    elif expr.op == 3:  # $gte
        var threshold = Float64(py=py.float(expr.value))
        return index.evaluate_range(expr.field, threshold, 1e308)

    elif expr.op == 4:  # $lt
        var threshold = Float64(py=py.float(expr.value))
        return index.evaluate_range(expr.field, -1e308, threshold - 1e-15)

    elif expr.op == 5:  # $lte
        var threshold = Float64(py=py.float(expr.value))
        return index.evaluate_range(expr.field, -1e308, threshold)

    elif expr.op == 6:  # $in
        var result = Bitmap(num_docs)
        var any_found = False
        for k in range(len(expr.values)):
            var val = expr.values[k]
            var bm = index.evaluate_eq(expr.field, val)
            if Bool(bm):
                any_found = True
                var merged = result.union(bm.value())
                result = merged^
        if any_found:
            return result.copy()
        return None

    elif expr.op == 7:  # $exists
        # If field is indexed, all docs in that index "exist"
        var exists = Bool(py=py.bool(expr.value))
        if expr.field in index.keyword_fields:
            var bm = Bitmap(num_docs)
            var kw = index.keyword_fields[expr.field].copy()
            var term_keys = List[String]()
            for key in kw.terms:
                term_keys.append(key)
            for key in term_keys:
                var bm_idx = kw.terms[key]
                var term_bm = kw.bitmaps[bm_idx].copy()
                for i in range(term_bm.size):
                    if term_bm.test(i):
                        bm.set(i)
            if not exists:
                return bm.complement()
            return bm.copy()
        elif expr.field in index.boolean_fields:
            var bi = index.boolean_fields[expr.field].copy()
            var bm = bi.get_true().union(bi.get_false())
            if not exists:
                return bm.complement()
            return bm.copy()
        elif expr.field in index.numeric_fields:
            var bm = Bitmap(num_docs)
            var ni = index.numeric_fields[expr.field].copy()
            for i in range(len(ni.bitmaps)):
                var entry_bm = ni.bitmaps[i].copy()
                for j in range(entry_bm.size):
                    if entry_bm.test(j):
                        bm.set(j)
            if not exists:
                return bm.complement()
            return bm.copy()
        # Field not indexed — if exists=True, return None (can't determine)
        if exists:
            return None
        # exists=False on unindexed field: all docs match (they don't have it)
        var all_bm = Bitmap(num_docs)
        for i in range(num_docs):
            all_bm.set(i)
        return all_bm.copy()

    elif expr.op == 8:  # $and
        var result: Optional[Bitmap] = None
        for i in range(len(expr.children)):
            var child_bm = _evaluate_bitmap(expr.children[i], index, num_docs)
            if not Bool(child_bm):
                return None
            if not Bool(result):
                result = child_bm.copy()
            else:
                var merged = result.value().intersect(child_bm.value())
                result = merged.copy()
        return result.copy()

    elif expr.op == 9:  # $or
        var result = Bitmap(num_docs)
        for i in range(len(expr.children)):
            var child_bm = _evaluate_bitmap(expr.children[i], index, num_docs)
            if not Bool(child_bm):
                return None
            var merged = result.union(child_bm.value())
            result = merged^
        return result.copy()

    elif expr.op == 10:  # $not
        if len(expr.children) == 0:
            return None
        var child_bm = _evaluate_bitmap(expr.children[0], index, num_docs)
        if not Bool(child_bm):
            return None
        var universe = Bitmap(num_docs)
        for i in range(num_docs):
            universe.set(i)
        return universe.and_not(child_bm.value())

    return None


# === Public API ===

def evaluate_filter_to_bitmap(
    filter_dict: PythonObject,
    ref index: MetadataIndex,
    num_docs: Int,
) raises -> Optional[Bitmap]:
    """Parse a Python filter dict and evaluate to a bitmap using the MetadataIndex.
    
    Returns None if the filter can't be evaluated via index (caller should fall back).
    Returns Bitmap of matching doc IDs if successful.
    """
    var expr = _parse_filter(filter_dict)
    return _evaluate_bitmap(expr, index, num_docs)


# === Tests ===

def test_filter() raises:
    print("=== Filter tests ===")
    var py = Python.import_module("builtins")

    # Build index
    var idx = MetadataIndex()
    var d0 = py.dict()
    d0["category"] = "A"
    d0["score"] = 10
    d0["active"] = True
    idx.index_json(0, d0)

    var d1 = py.dict()
    d1["category"] = "B"
    d1["score"] = 20
    d1["active"] = False
    idx.index_json(1, d1)

    var d2 = py.dict()
    d2["category"] = "A"
    d2["score"] = 30
    d2["active"] = True
    idx.index_json(2, d2)

    var d3 = py.dict()
    d3["category"] = "C"
    d3["score"] = 15
    d3["active"] = True
    idx.index_json(3, d3)

    var d4 = py.dict()
    d4["category"] = "A"
    d4["score"] = 25
    d4["active"] = False
    idx.index_json(4, d4)

    # Test $eq
    var f1 = py.dict()
    f1["category"] = "A"
    var bm1 = evaluate_filter_to_bitmap(f1, idx, 5)
    assert bm1.value().popcount() == 3  # docs 0, 2, 4

    # Test $gt
    var f2 = py.dict()
    f2["score"] = py.dict()
    f2["score"]["$gt"] = 15
    var bm2 = evaluate_filter_to_bitmap(f2, idx, 5)
    assert bm2.value().popcount() == 3  # docs 1(20), 2(30), 4(25)

    # Test $gte
    var f3 = py.dict()
    f3["score"] = py.dict()
    f3["score"]["$gte"] = 20
    var bm3 = evaluate_filter_to_bitmap(f3, idx, 5)
    assert bm3.value().popcount() == 3  # docs 1(20), 2(30), 4(25)

    # Test $lt
    var f4 = py.dict()
    f4["score"] = py.dict()
    f4["score"]["$lt"] = 20
    var bm4 = evaluate_filter_to_bitmap(f4, idx, 5)
    assert bm4.value().popcount() == 2  # docs 0(10), 3(15)

    # Test $lte
    var f5 = py.dict()
    f5["score"] = py.dict()
    f5["score"]["$lte"] = 15
    var bm5 = evaluate_filter_to_bitmap(f5, idx, 5)
    assert bm5.value().popcount() == 2  # docs 0(10), 3(15)

    # Test $in
    var f6 = py.dict()
    f6["category"] = py.dict()
    var in_list = py.list()
    in_list.append("A")
    in_list.append("C")
    f6["category"]["$in"] = in_list
    var bm6 = evaluate_filter_to_bitmap(f6, idx, 5)
    assert bm6.value().popcount() == 4  # docs 0, 2, 3, 4

    # Test $and
    var f7_and1 = py.dict()
    f7_and1["category"] = "A"
    var f7_and2 = py.dict()
    var f7_score_op = py.dict()
    f7_score_op["$gt"] = 20
    f7_and2["score"] = f7_score_op
    var f7_list = py.list()
    f7_list.append(f7_and1)
    f7_list.append(f7_and2)
    var f7 = py.dict()
    f7["$and"] = f7_list
    var bm7 = evaluate_filter_to_bitmap(f7, idx, 5)
    assert bm7.value().popcount() == 1  # doc 4 (category=A, score=25)

    # Test $or
    var f8_or1 = py.dict()
    f8_or1["category"] = "B"
    var f8_or2 = py.dict()
    f8_or2["category"] = "C"
    var f8_list = py.list()
    f8_list.append(f8_or1)
    f8_list.append(f8_or2)
    var f8 = py.dict()
    f8["$or"] = f8_list
    var bm8 = evaluate_filter_to_bitmap(f8, idx, 5)
    assert bm8.value().popcount() == 2  # docs 1(B), 3(C)

    print("All Filter tests passed!")


def main() raises:
    test_filter()
