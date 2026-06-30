"""Metadata index for filtered vector search.

Port of Rust metadata.rs. Three index types:
- KeywordIndex: inverted index for string equality (term -> bitmap)
- BooleanIndex: two bitmaps for true/false
- NumericIndex: sorted (value, bitmap) pairs for range queries

All bitmaps are plain List[UInt64] (no Roaring). Sufficient to ~10M docs.
"""

from bitmap import Bitmap
from std.python import Python, PythonObject


# === KeywordIndex ===

struct KeywordIndex(Movable, Copyable):
    """Inverted index for string equality: term -> bitmap of doc IDs.
    
    Stores bitmaps in a List for O(1) in-place mutation. Dict maps term -> index.
    """
    var bitmaps: List[Bitmap]
    var terms: Dict[String, Int]  # term -> index into bitmaps list
    var num_docs: Int

    def __init__(out self):
        self.bitmaps = List[Bitmap]()
        self.terms = Dict[String, Int]()
        self.num_docs = 0

    def __init__(out self, *, deinit take: Self):
        self.bitmaps = take.bitmaps^
        self.terms = take.terms^
        self.num_docs = take.num_docs

    def insert(mut self, doc_id: Int, term: String) raises:
        if term in self.terms:
            var idx = self.terms[term]
            self.bitmaps[idx].set(doc_id)
        else:
            var bm = Bitmap(max(doc_id + 1, self.num_docs))
            bm.set(doc_id)
            var idx = len(self.bitmaps)
            self.bitmaps.append(bm^)
            self.terms[term] = idx

    def remove(mut self, doc_id: Int) raises:
        var empty_keys = List[String]()
        for key in self.terms:
            empty_keys.append(key)
        for key in empty_keys:
            var idx = self.terms[key]
            self.bitmaps[idx].clear(doc_id)
            if self.bitmaps[idx].is_empty():
                _ = self.terms.pop(key)

    def get(ref self, term: String) raises -> Optional[Bitmap]:
        if term in self.terms:
            return self.bitmaps[self.terms[term]].copy()
        return None

    def contains(ref self, doc_id: Int, term: String) raises -> Bool:
        if term in self.terms:
            return self.bitmaps[self.terms[term]].test(doc_id)
        return False

    def ensure_size(mut self, size: Int) raises:
        if size <= self.num_docs:
            return
        # Exponential growth: double until >= size
        var new_cap = max(self.num_docs, 1)
        while new_cap < size:
            new_cap *= 2
        # Resize all bitmaps in-place (List access is mutable)
        for i in range(len(self.bitmaps)):
            self.bitmaps[i] = self.bitmaps[i].resized(new_cap)^
        self.num_docs = new_cap


# === BooleanIndex ===

struct BooleanIndex(Movable, Copyable):
    """Two bitmaps for true/false values."""
    var true_docs: Bitmap
    var false_docs: Bitmap
    var num_docs: Int

    def __init__(out self):
        self.true_docs = Bitmap(0)
        self.false_docs = Bitmap(0)
        self.num_docs = 0

    def __init__(out self, *, deinit take: Self):
        self.true_docs = take.true_docs^
        self.false_docs = take.false_docs^
        self.num_docs = take.num_docs

    def insert(mut self, doc_id: Int, value: Bool):
        var needed = doc_id + 1
        if needed > self.num_docs:
            self.ensure_size(needed)
        if value:
            self.true_docs.set(doc_id)
        else:
            self.false_docs.set(doc_id)

    def remove(mut self, doc_id: Int):
        self.true_docs.clear(doc_id)
        self.false_docs.clear(doc_id)

    def get_true(ref self) -> Bitmap:
        return self.true_docs.copy()

    def get_false(ref self) -> Bitmap:
        return self.false_docs.copy()

    def matches(ref self, doc_id: Int, value: Bool) -> Bool:
        if value:
            return self.true_docs.test(doc_id)
        return self.false_docs.test(doc_id)

    def ensure_size(mut self, size: Int):
        if size <= self.num_docs:
            return
        # Exponential growth: double until >= size
        var new_cap = max(self.num_docs, 1)
        while new_cap < size:
            new_cap *= 2
        self.true_docs = self.true_docs.resized(new_cap)^
        self.false_docs = self.false_docs.resized(new_cap)^
        self.num_docs = new_cap


# === NumericIndex ===

struct NumericIndex(Movable, Copyable):
    """Sorted (value, doc_id) pairs for range queries.
    
    For low-cardinality fields (<=256 unique values), uses bitmap-per-value.
    For high-cardinality fields, uses a flat sorted array of (value, doc_id) pairs.
    """
    var values: List[Float64]
    var bitmaps: List[Bitmap]
    # Flat sorted array for high-cardinality: (value, doc_id) pairs
    var flat_values: List[Float64]
    var flat_doc_ids: List[Int]
    var num_docs: Int
    var use_flat: Bool  # True when >256 unique values

    def __init__(out self):
        self.values = List[Float64]()
        self.bitmaps = List[Bitmap]()
        self.flat_values = List[Float64]()
        self.flat_doc_ids = List[Int]()
        self.num_docs = 0
        self.use_flat = False

    def __init__(out self, *, deinit take: Self):
        self.values = take.values^
        self.bitmaps = take.bitmaps^
        self.flat_values = take.flat_values^
        self.flat_doc_ids = take.flat_doc_ids^
        self.num_docs = take.num_docs
        self.use_flat = take.use_flat

    def _find_index(ref self, value: Float64) -> Int:
        var lo = 0
        var hi = len(self.values)
        while lo < hi:
            var mid = (lo + hi) // 2
            if self.values[mid] < value:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def _flat_find(ref self, value: Float64) -> Int:
        """Binary search in flat sorted array."""
        var lo = 0
        var hi = len(self.flat_values)
        while lo < hi:
            var mid = (lo + hi) // 2
            if self.flat_values[mid] < value:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def _convert_to_flat(mut self):
        """Convert from bitmap mode to flat mode when cardinality exceeds threshold."""
        for i in range(len(self.values)):
            var val = self.values[i]
            var bm = self.bitmaps[i].copy()
            for j in range(bm.size):
                if bm.test(j):
                    self.flat_values.append(val)
                    self.flat_doc_ids.append(j)
        self.values = List[Float64]()
        self.bitmaps = List[Bitmap]()
        self.use_flat = True

    def insert(mut self, doc_id: Int, value: Float64) raises:
        if value != value:  # NaN
            return
        if self.use_flat:
            # Insert into flat sorted array
            var idx = self._flat_find(value)
            self.flat_values.insert(idx, value)
            self.flat_doc_ids.insert(idx, doc_id)
            return
        var idx = self._find_index(value)
        if idx < len(self.values) and self.values[idx] == value:
            self.bitmaps[idx].set(doc_id)
        else:
            var bm = Bitmap(max(doc_id + 1, self.num_docs))
            bm.set(doc_id)
            self.values.insert(idx, value)
            self.bitmaps.insert(idx, bm^)
            # Switch to flat mode if too many unique values
            if len(self.values) > 256:
                self._convert_to_flat()

    def remove(mut self, doc_id: Int):
        if self.use_flat:
            # Linear scan to remove — acceptable for rare deletes
            var new_vals = List[Float64]()
            var new_ids = List[Int]()
            for i in range(len(self.flat_doc_ids)):
                if self.flat_doc_ids[i] != doc_id:
                    new_vals.append(self.flat_values[i])
                    new_ids.append(self.flat_doc_ids[i])
            self.flat_values = new_vals^
            self.flat_doc_ids = new_ids^
            return
        for i in range(len(self.bitmaps)):
            self.bitmaps[i].clear(doc_id)

    def get_eq(ref self, value: Float64) raises -> Optional[Bitmap]:
        if self.use_flat:
            var result = Bitmap(self.num_docs)
            var lo = self._flat_find(value)
            var i = lo
            while i < len(self.flat_values) and self.flat_values[i] == value:
                result.set(self.flat_doc_ids[i])
                i += 1
            return result^
        var idx = self._find_index(value)
        if idx < len(self.values) and self.values[idx] == value:
            return self.bitmaps[idx].copy()
        return None

    def get_range(ref self, min_val: Float64, max_val: Float64) raises -> Bitmap:
        if self.use_flat:
            var result = Bitmap(self.num_docs)
            var lo = self._flat_find(min_val)
            var i = lo
            while i < len(self.flat_values) and self.flat_values[i] <= max_val:
                result.set(self.flat_doc_ids[i])
                i += 1
            return result^
        var result = Bitmap(self.num_docs)
        var lo = self._find_index(min_val)
        for i in range(lo, len(self.values)):
            if self.values[i] > max_val:
                break
            for j in range(self.bitmaps[i].size):
                if self.bitmaps[i].test(j):
                    result.set(j)
        return result^

    def matches_eq(ref self, doc_id: Int, value: Float64) raises -> Bool:
        if self.use_flat:
            var lo = self._flat_find(value)
            var i = lo
            while i < len(self.flat_values) and self.flat_values[i] == value:
                if self.flat_doc_ids[i] == doc_id:
                    return True
                i += 1
            return False
        var idx = self._find_index(value)
        if idx < len(self.values) and self.values[idx] == value:
            return self.bitmaps[idx].test(doc_id)
        return False

    def matches_gt(ref self, doc_id: Int, threshold: Float64) -> Bool:
        if self.use_flat:
            var lo = self._flat_find(threshold)
            var i = lo
            while i < len(self.flat_values) and self.flat_values[i] <= threshold:
                i += 1
            while i < len(self.flat_values):
                if self.flat_doc_ids[i] == doc_id:
                    return True
                i += 1
            return False
        var idx = self._find_index(threshold)
        if idx < len(self.values) and self.values[idx] == threshold:
            idx += 1
        for i in range(idx, len(self.values)):
            if self.bitmaps[i].test(doc_id):
                return True
        return False

    def matches_lt(ref self, doc_id: Int, threshold: Float64) -> Bool:
        if self.use_flat:
            var lo = self._flat_find(threshold)
            var i = 0
            while i < lo:
                if self.flat_doc_ids[i] == doc_id:
                    return True
                i += 1
            return False
        var idx = self._find_index(threshold)
        for i in range(idx):
            if self.bitmaps[i].test(doc_id):
                return True
        return False

    def matches_range(ref self, doc_id: Int, min_val: Float64, max_val: Float64) -> Bool:
        if self.use_flat:
            var lo = self._flat_find(min_val)
            var i = lo
            while i < len(self.flat_values) and self.flat_values[i] <= max_val:
                if self.flat_doc_ids[i] == doc_id:
                    return True
                i += 1
            return False
        var lo = self._find_index(min_val)
        for i in range(lo, len(self.values)):
            if self.values[i] > max_val:
                break
            if self.bitmaps[i].test(doc_id):
                return True
        return False

    def ensure_size(mut self, size: Int) raises:
        if size <= self.num_docs:
            return
        var new_cap = max(self.num_docs, 1)
        while new_cap < size:
            new_cap *= 2
        if not self.use_flat:
            for i in range(len(self.bitmaps)):
                if self.bitmaps[i].size < new_cap:
                    self.bitmaps[i] = self.bitmaps[i].resized(new_cap)^
        self.num_docs = new_cap


# === MetadataIndex ===

struct MetadataIndex(Movable, Copyable):
    """Collection of field indexes. Auto-indexes all fields on set().
    
    Uses per-field keyword/boolean/numeric indexes. Each field type is
    auto-detected from the Python value type on first insert.
    """
    var keyword_fields: Dict[String, KeywordIndex]
    var boolean_fields: Dict[String, BooleanIndex]
    var numeric_fields: Dict[String, NumericIndex]
    var num_docs: Int

    def __init__(out self):
        self.keyword_fields = Dict[String, KeywordIndex]()
        self.boolean_fields = Dict[String, BooleanIndex]()
        self.numeric_fields = Dict[String, NumericIndex]()
        self.num_docs = 0

    def __init__(out self, *, deinit take: Self):
        self.keyword_fields = take.keyword_fields^
        self.boolean_fields = take.boolean_fields^
        self.numeric_fields = take.numeric_fields^
        self.num_docs = take.num_docs

    def index_json(mut self, doc_id: Int, metadata: PythonObject) raises:
        """Index a Python dict metadata object. Auto-indexes all fields."""
        if metadata is Python.none():
            return
        var py = Python.import_module("builtins")
        var needed = doc_id + 1
        if needed > self.num_docs:
            var kw_keys = List[String]()
            for key in self.keyword_fields:
                kw_keys.append(key)
            for key in kw_keys:
                var kw = self.keyword_fields[key].copy()
                kw.ensure_size(needed)
                self.keyword_fields[key] = kw^
            var bi_keys = List[String]()
            for key in self.boolean_fields:
                bi_keys.append(key)
            for key in bi_keys:
                var bi = self.boolean_fields[key].copy()
                bi.ensure_size(needed)
                self.boolean_fields[key] = bi^
            var ni_keys = List[String]()
            for key in self.numeric_fields:
                ni_keys.append(key)
            for key in ni_keys:
                var ni = self.numeric_fields[key].copy()
                ni.ensure_size(needed)
                self.numeric_fields[key] = ni^
            self.num_docs = needed

        var keys = py.list(metadata)
        var n = len(keys)
        for i in range(n):
            var key = keys[i]
            var key_str = String(key)
            var value = metadata[key]

            if py.isinstance(value, py.str):
                var kw: KeywordIndex
                if key_str in self.keyword_fields:
                    kw = self.keyword_fields[key_str].copy()
                else:
                    kw = KeywordIndex()
                kw.insert(doc_id, String(value))
                self.keyword_fields[key_str] = kw^
            elif py.isinstance(value, py.bool):
                var bi: BooleanIndex
                if key_str in self.boolean_fields:
                    bi = self.boolean_fields[key_str].copy()
                else:
                    bi = BooleanIndex()
                bi.insert(doc_id, Bool(py=value))
                self.boolean_fields[key_str] = bi^
            elif py.isinstance(value, py.int) or py.isinstance(value, py.float):
                var ni: NumericIndex
                if key_str in self.numeric_fields:
                    ni = self.numeric_fields[key_str].copy()
                else:
                    ni = NumericIndex()
                ni.insert(doc_id, Float64(py=value))
                self.numeric_fields[key_str] = ni^

    def remove(mut self, doc_id: Int) raises:
        """Remove a document from all field indexes."""
        var kw_keys = List[String]()
        for key in self.keyword_fields:
            kw_keys.append(key)
        for key in kw_keys:
            var kw = self.keyword_fields[key].copy()
            kw.remove(doc_id)
            self.keyword_fields[key] = kw^
        var bi_keys = List[String]()
        for key in self.boolean_fields:
            bi_keys.append(key)
        for key in bi_keys:
            var bi = self.boolean_fields[key].copy()
            bi.remove(doc_id)
            self.boolean_fields[key] = bi^
        var ni_keys = List[String]()
        for key in self.numeric_fields:
            ni_keys.append(key)
        for key in ni_keys:
            var ni = self.numeric_fields[key].copy()
            ni.remove(doc_id)
            self.numeric_fields[key] = ni^

    def evaluate_eq(ref self, field: String, value: PythonObject) raises -> Optional[Bitmap]:
        """Evaluate equality predicate against index. Returns bitmap or None if not indexed."""
        var py = Python.import_module("builtins")
        if py.isinstance(value, py.str):
            if field in self.keyword_fields:
                return self.keyword_fields[field].get(String(value))
        elif py.isinstance(value, py.bool):
            if field in self.boolean_fields:
                var bi = self.boolean_fields[field].copy()
                if Bool(py=value):
                    return bi.get_true()
                return bi.get_false()
        elif py.isinstance(value, py.int) or py.isinstance(value, py.float):
            if field in self.numeric_fields:
                return self.numeric_fields[field].get_eq(Float64(py=value))
        return None

    def evaluate_range(ref self, field: String, min_val: Float64, max_val: Float64) raises -> Optional[Bitmap]:
        """Evaluate range predicate. Returns bitmap or None if not indexed."""
        if field in self.numeric_fields:
            return self.numeric_fields[field].get_range(min_val, max_val)
        return None


# === Tests ===

def test_metadata_index() raises:
    print("=== MetadataIndex tests ===")
    var py = Python.import_module("builtins")

    # KeywordIndex
    var kw = KeywordIndex()
    kw.num_docs = 4
    kw.insert(0, "cat")
    kw.insert(1, "dog")
    kw.insert(2, "cat")
    kw.insert(3, "bird")
    assert kw.contains(0, "cat")
    assert not kw.contains(1, "cat")
    assert kw.contains(2, "cat")
    var cat_bm = kw.get("cat")
    assert cat_bm.value().popcount() == 2
    var dog_bm = kw.get("dog")
    assert dog_bm.value().popcount() == 1
    assert kw.get("bird").value().popcount() == 1

    # BooleanIndex
    var bi = BooleanIndex()
    bi.num_docs = 4
    bi.insert(0, True)
    bi.insert(1, False)
    bi.insert(2, True)
    bi.insert(3, False)
    assert bi.matches(0, True)
    assert not bi.matches(0, False)
    assert bi.matches(1, False)
    assert bi.get_true().popcount() == 2
    assert bi.get_false().popcount() == 2

    # NumericIndex
    var ni = NumericIndex()
    ni.num_docs = 5
    ni.insert(0, 10.0)
    ni.insert(1, 20.0)
    ni.insert(2, 30.0)
    ni.insert(3, 15.0)
    ni.insert(4, 25.0)
    assert ni.matches_eq(0, 10.0)
    assert not ni.matches_eq(0, 20.0)
    assert ni.matches_gt(1, 15.0)
    assert not ni.matches_gt(0, 15.0)
    assert ni.matches_lt(0, 15.0)
    assert not ni.matches_lt(1, 15.0)
    assert ni.matches_range(1, 15.0, 25.0)
    assert not ni.matches_range(0, 15.0, 25.0)
    var range_bm = ni.get_range(15.0, 25.0)
    assert range_bm.popcount() == 3

    # MetadataIndex with Python dicts
    var idx = MetadataIndex()
    var d0 = py.dict()
    d0["name"] = "Alice"
    d0["age"] = 30
    d0["active"] = True
    idx.index_json(0, d0)

    var d1 = py.dict()
    d1["name"] = "Bob"
    d1["age"] = 25
    d1["active"] = False
    idx.index_json(1, d1)

    var d2 = py.dict()
    d2["name"] = "Alice"
    d2["age"] = 35
    d2["active"] = True
    idx.index_json(2, d2)

    # Keyword check
    assert idx.keyword_fields["name"].contains(0, "Alice")
    assert idx.keyword_fields["name"].contains(1, "Bob")
    assert idx.keyword_fields["name"].contains(2, "Alice")
    assert idx.keyword_fields["name"].get("Alice").value().popcount() == 2

    # Numeric check
    assert idx.numeric_fields["age"].matches_eq(0, 30.0)
    assert idx.numeric_fields["age"].matches_gt(2, 31.0)

    # Boolean check
    assert idx.boolean_fields["active"].matches(0, True)
    assert idx.boolean_fields["active"].matches(1, False)

    # Remove
    idx.remove(1)
    assert not idx.keyword_fields["name"].contains(1, "Bob")

    # evaluate_eq
    var bm = idx.evaluate_eq("name", "Alice")
    assert bm.value().popcount() == 2
    assert bm.value().test(0)
    assert bm.value().test(2)

    var bm2 = idx.evaluate_range("age", 28.0, 100.0)
    assert bm2.value().popcount() == 2  # age 30 and 35

    print("All MetadataIndex tests passed!")


def main() raises:
    test_metadata_index()
