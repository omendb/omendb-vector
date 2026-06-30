from hnsw import HNSWIndex
from store_types import Metric
from std.memory import Span
from std.sys import size_of


def save_list_f32(path: String, data: List[Float32]) raises:
    var n = len(data)
    with open(path, "w") as f:
        if n > 0:
            f.write_all(
                Span(
                    ptr=data.unsafe_ptr().bitcast[Byte](),
                    length=n * size_of[Float32](),
                )
            )


def load_list_f32(path: String) raises -> List[Float32]:
    var b: List[UInt8]
    try:
        with open(path, "r") as f:
            b = f.read_bytes()
    except:
        raise Error("missing Float32 list file")
    if len(b) == 0:
        return List[Float32]()

    var n_bytes = len(b)
    if n_bytes % size_of[Float32]() != 0:
        raise Error("corrupt Float32 list file")
    var n_elements = n_bytes // size_of[Float32]()
    var ptr = b.unsafe_ptr().bitcast[Float32]()
    var result = List[Float32]()
    for i in range(n_elements):
        result.append(ptr[i])
    return result^


def save_list_u32(path: String, data: List[UInt32]) raises:
    var n = len(data)
    with open(path, "w") as f:
        if n > 0:
            f.write_all(
                Span(
                    ptr=data.unsafe_ptr().bitcast[Byte](),
                    length=n * size_of[UInt32](),
                )
            )


def load_list_u32(path: String) raises -> List[UInt32]:
    var b: List[UInt8]
    try:
        with open(path, "r") as f:
            b = f.read_bytes()
    except:
        raise Error("missing UInt32 list file")
    if len(b) == 0:
        return List[UInt32]()
    var n_bytes = len(b)
    if n_bytes % size_of[UInt32]() != 0:
        raise Error("corrupt UInt32 list file")
    var n_elements = n_bytes // size_of[UInt32]()
    var ptr = b.unsafe_ptr().bitcast[UInt32]()
    var result = List[UInt32]()
    for i in range(n_elements):
        result.append(ptr[i])
    return result^


def save_list_u8(path: String, data: List[UInt8]) raises:
    var n = len(data)
    with open(path, "w") as f:
        if n > 0:
            f.write_all(Span(ptr=data.unsafe_ptr().bitcast[Byte](), length=n))


def load_list_u8(path: String) raises -> List[UInt8]:
    try:
        with open(path, "r") as f:
            return f.read_bytes()
    except:
        raise Error("missing UInt8 list file")


def save_list_int(path: String, data: List[Int]) raises:
    var n = len(data)
    with open(path, "w") as f:
        if n > 0:
            f.write_all(
                Span(
                    ptr=data.unsafe_ptr().bitcast[Byte](),
                    length=n * size_of[Int](),
                )
            )


def load_list_int(path: String) raises -> List[Int]:
    var b: List[UInt8]
    try:
        with open(path, "r") as f:
            b = f.read_bytes()
    except:
        raise Error("missing Int list file")
    if len(b) == 0:
        return List[Int]()
    var n_bytes = len(b)
    if n_bytes % size_of[Int]() != 0:
        raise Error("corrupt Int list file")
    var n_elements = n_bytes // size_of[Int]()
    var ptr = b.unsafe_ptr().bitcast[Int]()
    var result = List[Int]()
    for i in range(n_elements):
        result.append(ptr[i])
    return result^


def save_hnsw[dim: Int](path_prefix: String, ref idx: HNSWIndex[dim]) raises:
    save_list_f32(path_prefix + ".data", idx.data)
    save_list_u8(path_prefix + ".codes", idx.codes)

    var meta = List[Float32]()
    meta.append(Float32(idx.ep))
    meta.append(Float32(idx.max_level))
    meta.append(Float32(idx.num_elements))
    meta.append(Float32(idx.M))
    meta.append(Float32(idx.M0))
    meta.append(Float32(idx.ef_construction))
    meta.append(idx.alpha)
    meta.append(idx.quantizer.min_val)
    meta.append(idx.quantizer.max_val)
    meta.append(Float32(idx.metric._value))
    save_list_f32(path_prefix + ".meta", meta)

    save_list_int(path_prefix + ".levels", idx.node_levels)

    for lc in range(idx.graph.layer_count()):
        save_list_u32(
            path_prefix + ".layer" + String(lc) + ".nbrs",
            idx.graph.layer_neighbors_copy(lc),
        )
        save_list_u32(
            path_prefix + ".layer" + String(lc) + ".counts",
            idx.graph.layer_counts_copy(lc),
        )


def load_hnsw[dim: Int](path_prefix: String) raises -> HNSWIndex[dim]:
    var meta = load_list_f32(path_prefix + ".meta")
    if len(meta) < 9:
        raise Error("Invalid metadata")

    var ep = UInt32(Int(meta[0]))
    var max_level = Int(meta[1])
    var num_elements = Int(meta[2])
    var M = Int(meta[3])
    var ef_construction = Int(meta[5])
    var alpha = meta[6]
    var min_val = meta[7]
    var max_val = meta[8]
    var metric = Metric.L2
    if len(meta) > 9:
        metric = Metric(_value=Int(meta[9]))

    var idx = HNSWIndex[dim](
        M=M,
        ef_construction=ef_construction,
        min_val=min_val,
        max_val=max_val,
        alpha=alpha,
        metric=metric,
    )

    var data = load_list_f32(path_prefix + ".data")
    var codes = load_list_u8(path_prefix + ".codes")
    var levels = load_list_int(path_prefix + ".levels")

    if len(data) != num_elements * dim:
        raise Error("Invalid HNSW data length")
    if len(codes) != num_elements * dim:
        raise Error("Invalid HNSW code length")
    if len(levels) != num_elements:
        raise Error("Invalid HNSW level count")

    var layer_neighbors = List[List[UInt32]]()
    var layer_counts = List[List[UInt32]]()
    for lc in range(max_level + 1):
        layer_neighbors.append(
            load_list_u32(path_prefix + ".layer" + String(lc) + ".nbrs")
        )
        layer_counts.append(
            load_list_u32(path_prefix + ".layer" + String(lc) + ".counts")
        )

        var max_conn = M * 2 if lc == 0 else M
        if len(layer_counts[lc]) > num_elements:
            raise Error("Invalid HNSW layer count length")
        if len(layer_neighbors[lc]) != len(layer_counts[lc]) * max_conn:
            raise Error("Invalid HNSW neighbor layer length")

    idx.load_from_data(
        data^,
        codes^,
        layer_neighbors^,
        layer_counts^,
        levels^,
        ep,
        max_level,
        num_elements,
    )
    return idx^


# === MetadataIndex persistence ===
# Fully inlined — Mojo 1.0 @parameter fn closures can't take List by value.

from metadata_index import MetadataIndex, KeywordIndex, BooleanIndex, NumericIndex, Bitmap
from std.python import Python, PythonObject


def save_metadata_index(path: String, ref idx: MetadataIndex) raises:
    """Save MetadataIndex via Python JSON (avoids Mojo List parameter issues)."""
    var py = Python.import_module("builtins")
    var json = Python.import_module("json")
    var data = py.dict()
    data["num_docs"] = idx.num_docs

    # keyword fields: {name: {num_docs: N, terms: {term: [u64_words...]}}}
    var kw = py.dict()
    var kw_keys = List[String]()
    for k in idx.keyword_fields:
        kw_keys.append(k)
    for ki in range(len(kw_keys)):
        var name = kw_keys[ki]
        var kw_idx = idx.keyword_fields[name].copy()
        var kw_entry = py.dict()
        kw_entry["num_docs"] = kw_idx.num_docs
        var terms = py.dict()
        var term_keys = List[String]()
        for t in kw_idx.terms:
            term_keys.append(t)
        for ti in range(len(term_keys)):
            var term = term_keys[ti]
            var bm_idx = kw_idx.terms[term]
            var bm = kw_idx.bitmaps[bm_idx].copy()
            var words = py.list()
            for i in range(len(bm.bits)):
                _ = words.append(Int(bm.bits[i]))
            terms[term] = words
        kw_entry["terms"] = terms
        kw[name] = kw_entry
    data["keyword_fields"] = kw

    # boolean fields: {name: {num_docs: N, true_words: [...], false_words: [...]}}
    var bi = py.dict()
    var bi_keys = List[String]()
    for k in idx.boolean_fields:
        bi_keys.append(k)
    for ki in range(len(bi_keys)):
        var name = bi_keys[ki]
        var bi_idx = idx.boolean_fields[name].copy()
        var bi_entry = py.dict()
        bi_entry["num_docs"] = bi_idx.num_docs
        var true_words = py.list()
        for i in range(len(bi_idx.true_docs.bits)):
            _ = true_words.append(Int(bi_idx.true_docs.bits[i]))
        bi_entry["true_words"] = true_words
        var false_words = py.list()
        for i in range(len(bi_idx.false_docs.bits)):
            _ = false_words.append(Int(bi_idx.false_docs.bits[i]))
        bi_entry["false_words"] = false_words
        bi[name] = bi_entry
    data["boolean_fields"] = bi

    # numeric fields: {name: {num_docs: N, use_flat: bool, entries: [...], flat_values: [...], flat_doc_ids: [...]}}
    var ni = py.dict()
    var ni_keys = List[String]()
    for k in idx.numeric_fields:
        ni_keys.append(k)
    for ki in range(len(ni_keys)):
        var name = ni_keys[ki]
        var ni_idx = idx.numeric_fields[name].copy()
        var ni_entry = py.dict()
        ni_entry["num_docs"] = ni_idx.num_docs
        ni_entry["use_flat"] = ni_idx.use_flat
        if ni_idx.use_flat:
            var flat_vals = py.list()
            var flat_ids = py.list()
            for fi in range(len(ni_idx.flat_values)):
                _ = flat_vals.append(ni_idx.flat_values[fi])
                _ = flat_ids.append(ni_idx.flat_doc_ids[fi])
            ni_entry["flat_values"] = flat_vals
            ni_entry["flat_doc_ids"] = flat_ids
        else:
            var entries = py.list()
            for ei in range(len(ni_idx.values)):
                var entry = py.dict()
                entry["value"] = ni_idx.values[ei]
                var words = py.list()
                for i in range(len(ni_idx.bitmaps[ei].bits)):
                    _ = words.append(Int(ni_idx.bitmaps[ei].bits[i]))
                entry["words"] = words
                _ = entries.append(entry)
            ni_entry["entries"] = entries
        ni[name] = ni_entry
    data["numeric_fields"] = ni

    var json_str = json.dumps(data)
    var builtins = Python.import_module("builtins")
    var f = builtins.open(path, "w")
    _ = f.write(json_str)
    _ = f.close()


def load_metadata_index(path: String) raises -> MetadataIndex:
    """Load MetadataIndex via Python JSON. Returns empty index if file missing."""
    var os = Python.import_module("os")
    if not Bool(os.path.exists(path)):
        return MetadataIndex()
    var builtins = Python.import_module("builtins")
    var json = Python.import_module("json")
    var f = builtins.open(path, "r")
    var json_str = f.read()
    _ = f.close()
    var data = json.loads(json_str)

    var idx = MetadataIndex()
    idx.num_docs = Int(py=data["num_docs"])

    # keyword fields
    var kw = data["keyword_fields"]
    for name in kw:
        var name_str = String(name)
        var kw_entry = kw[name]
        var kw_idx = KeywordIndex()
        kw_idx.num_docs = Int(py=kw_entry["num_docs"])
        var terms = kw_entry["terms"]
        for term in terms:
            var term_str = String(term)
            var words = terms[term]
            var n_words = len(words)
            var bm = Bitmap(n_words * 64)
            for i in range(n_words):
                bm.bits[i] = UInt64(Int(py=words[i]))
            var bm_idx = len(kw_idx.bitmaps)
            kw_idx.bitmaps.append(bm^)
            kw_idx.terms[term_str] = bm_idx
        idx.keyword_fields[name_str] = kw_idx^

    # boolean fields
    var bi = data["boolean_fields"]
    for name in bi:
        var name_str = String(name)
        var bi_entry = bi[name]
        var bi_idx = BooleanIndex()
        bi_idx.num_docs = Int(py=bi_entry["num_docs"])
        var true_words = bi_entry["true_words"]
        var n_true = len(true_words)
        var true_bm = Bitmap(n_true * 64)
        for i in range(n_true):
            true_bm.bits[i] = UInt64(Int(py=true_words[i]))
        bi_idx.true_docs = true_bm^
        var false_words = bi_entry["false_words"]
        var n_false = len(false_words)
        var false_bm = Bitmap(n_false * 64)
        for i in range(n_false):
            false_bm.bits[i] = UInt64(Int(py=false_words[i]))
        bi_idx.false_docs = false_bm^
        idx.boolean_fields[name_str] = bi_idx^

    # numeric fields
    var ni = data["numeric_fields"]
    for name in ni:
        var name_str = String(name)
        var ni_entry = ni[name]
        var ni_idx = NumericIndex()
        ni_idx.num_docs = Int(py=ni_entry["num_docs"])
        var use_flat = Bool(Int(py=ni_entry.get("use_flat", False)))
        ni_idx.use_flat = use_flat
        if use_flat:
            var flat_vals = ni_entry["flat_values"]
            var flat_ids = ni_entry["flat_doc_ids"]
            for fi in range(len(flat_vals)):
                ni_idx.flat_values.append(Float64(py=flat_vals[fi]))
                ni_idx.flat_doc_ids.append(Int(py=flat_ids[fi]))
        else:
            var entries = ni_entry["entries"]
            for entry in entries:
                var val = Float64(py=entry["value"])
                var words = entry["words"]
                var n_words = len(words)
                var bm = Bitmap(n_words * 64)
                for i in range(n_words):
                    bm.bits[i] = UInt64(Int(py=words[i]))
                ni_idx.values.append(val)
                ni_idx.bitmaps.append(bm^)
        idx.numeric_fields[name_str] = ni_idx^

    return idx^
