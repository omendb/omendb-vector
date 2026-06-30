from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from store_types import Metric
from dynamic_flat_store import DynamicFlatVectorStore, DynamicFlatSearchResult


from graph import GraphDirection
from hnsw import Candidate
from multivector import (
    MultiVectorExactStore,
    MultiVectorMuveraStore,
    MultiVectorResult,
)
from mojo_filter import filter_ids_native, filter_ids_batch
from sparse_index import SparseVector
from filter import evaluate_filter_to_bitmap
from store import (
    HNSWParams,
    SearchOptions,
    SearchResult,
    VectorStore,
    VectorStoreOptions,
)





@export
def PyInit__omendb_engine() -> PythonObject:
    try:
        var module = PythonModuleBuilder("_omendb_engine")
        module.def_function[engine_version](
            "engine_version",
            docstring="Return the packaged Mojo engine version.",
        )
        module.def_function[dense_collection2_memory](
            "_dense_collection2_memory",
            docstring="Create an in-memory dim=2 collection.",
        )
        module.def_function[dense_collection2_create](
            "_dense_collection2_create",
            docstring="Create a persistent dim=2 collection.",
        )
        module.def_function[dense_collection2_open](
            "_dense_collection2_open",
            docstring="Open a persistent dim=2 collection.",
        )
        module.def_function[dense_collection128_memory](
            "_dense_collection128_memory",
            docstring="Create an in-memory dim=128 collection.",
        )
        module.def_function[dense_collection128_create](
            "_dense_collection128_create",
            docstring="Create a persistent dim=128 collection.",
        )
        module.def_function[dense_collection128_open](
            "_dense_collection128_open",
            docstring="Open a persistent dim=128 collection.",
        )
        module.def_function[dense_collection384_memory](
            "_dense_collection384_memory",
            docstring="Create an in-memory dim=384 collection.",
        )
        module.def_function[dense_collection384_create](
            "_dense_collection384_create",
            docstring="Create a persistent dim=384 collection.",
        )
        module.def_function[dense_collection384_open](
            "_dense_collection384_open",
            docstring="Open a persistent dim=384 collection.",
        )
        module.def_function[dense_collection256_memory](
            "_dense_collection256_memory",
            docstring="Create an in-memory dim=256 collection.",
        )
        module.def_function[dense_collection256_create](
            "_dense_collection256_create",
            docstring="Create a persistent dim=256 collection.",
        )
        module.def_function[dense_collection256_open](
            "_dense_collection256_open",
            docstring="Open a persistent dim=256 collection.",
        )
        module.def_function[dense_collection512_memory](
            "_dense_collection512_memory",
            docstring="Create an in-memory dim=512 collection.",
        )
        module.def_function[dense_collection512_create](
            "_dense_collection512_create",
            docstring="Create a persistent dim=512 collection.",
        )
        module.def_function[dense_collection512_open](
            "_dense_collection512_open",
            docstring="Open a persistent dim=512 collection.",
        )
        module.def_function[dense_collection768_memory](
            "_dense_collection768_memory",
            docstring="Create an in-memory dim=768 collection.",
        )
        module.def_function[dense_collection768_create](
            "_dense_collection768_create",
            docstring="Create a persistent dim=768 collection.",
        )
        module.def_function[dense_collection768_open](
            "_dense_collection768_open",
            docstring="Open a persistent dim=768 collection.",
        )
        module.def_function[dense_collection1024_memory](
            "_dense_collection1024_memory",
            docstring="Create an in-memory dim=1024 collection.",
        )
        module.def_function[dense_collection1024_create](
            "_dense_collection1024_create",
            docstring="Create a persistent dim=1024 collection.",
        )
        module.def_function[dense_collection1024_open](
            "_dense_collection1024_open",
            docstring="Open a persistent dim=1024 collection.",
        )
        module.def_function[dense_collection1536_memory](
            "_dense_collection1536_memory",
            docstring="Create an in-memory dim=1536 collection.",
        )
        module.def_function[dense_collection1536_create](
            "_dense_collection1536_create",
            docstring="Create a persistent dim=1536 collection.",
        )
        module.def_function[dense_collection1536_open](
            "_dense_collection1536_open",
            docstring="Open a persistent dim=1536 collection.",
        )
        module.def_function[dense_collection3072_memory](
            "_dense_collection3072_memory",
            docstring="Create an in-memory dim=3072 collection.",
        )
        module.def_function[dense_collection3072_create](
            "_dense_collection3072_create",
            docstring="Create a persistent dim=3072 collection.",
        )
        module.def_function[dense_collection3072_open](
            "_dense_collection3072_open",
            docstring="Open a persistent dim=3072 collection.",
        )
        module.def_function[dynamic_flat_collection_memory](
            "_dynamic_flat_collection_memory",
            docstring="Create an in-memory dynamic flat collection.",
        )
        module.def_function[dynamic_flat_collection_create](
            "_dynamic_flat_collection_create",
            docstring="Create a persistent dynamic flat collection.",
        )
        module.def_function[dynamic_flat_collection_open](
            "_dynamic_flat_collection_open",
            docstring="Open a persistent dynamic flat collection.",
        )
        module.def_function[multivector_collection2_memory](
            "_multivector_collection2_memory",
            docstring="Create an in-memory dim=2 multi-vector collection.",
        )
        module.def_function[multivector_collection2_create](
            "_multivector_collection2_create",
            docstring="Create a persistent dim=2 multi-vector collection.",
        )
        module.def_function[multivector_collection2_open](
            "_multivector_collection2_open",
            docstring="Open a persistent dim=2 multi-vector collection.",
        )
        module.def_function[multivector_collection2_muvera_memory](
            "_multivector_collection2_muvera_memory",
            docstring=(
                "Create an in-memory dim=2 MuVERA multi-vector collection."
            ),
        )
        module.def_function[multivector_collection2_muvera_create](
            "_multivector_collection2_muvera_create",
            docstring=(
                "Create a persistent dim=2 MuVERA multi-vector collection."
            ),
        )
        module.def_function[multivector_collection2_muvera_open](
            "_multivector_collection2_muvera_open",
            docstring="Open a persistent dim=2 MuVERA multi-vector collection.",
        )
        module.def_function[multivector_collection48_memory](
            "_multivector_collection48_memory",
            docstring="Create an in-memory dim=48 multi-vector collection.",
        )
        module.def_function[multivector_collection48_create](
            "_multivector_collection48_create",
            docstring="Create a persistent dim=48 multi-vector collection.",
        )
        module.def_function[multivector_collection48_open](
            "_multivector_collection48_open",
            docstring="Open a persistent dim=48 multi-vector collection.",
        )
        module.def_function[multivector_collection48_muvera_memory](
            "_multivector_collection48_muvera_memory",
            docstring=(
                "Create an in-memory dim=48 MuVERA multi-vector collection."
            ),
        )
        module.def_function[multivector_collection48_muvera_create](
            "_multivector_collection48_muvera_create",
            docstring=(
                "Create a persistent dim=48 MuVERA multi-vector collection."
            ),
        )
        module.def_function[multivector_collection48_muvera_open](
            "_multivector_collection48_muvera_open",
            docstring=(
                "Open a persistent dim=48 MuVERA multi-vector collection."
            ),
        )
        module.def_function[multivector_collection128_memory](
            "_multivector_collection128_memory",
            docstring="Create an in-memory dim=128 multi-vector collection.",
        )
        module.def_function[multivector_collection128_create](
            "_multivector_collection128_create",
            docstring="Create a persistent dim=128 multi-vector collection.",
        )
        module.def_function[multivector_collection128_open](
            "_multivector_collection128_open",
            docstring="Open a persistent dim=128 multi-vector collection.",
        )
        module.def_function[multivector_collection128_muvera_memory](
            "_multivector_collection128_muvera_memory",
            docstring=(
                "Create an in-memory dim=128 MuVERA multi-vector collection."
            ),
        )
        module.def_function[multivector_collection128_muvera_create](
            "_multivector_collection128_muvera_create",
            docstring=(
                "Create a persistent dim=128 MuVERA multi-vector collection."
            ),
        )
        module.def_function[multivector_collection128_muvera_open](
            "_multivector_collection128_muvera_open",
            docstring=(
                "Open a persistent dim=128 MuVERA multi-vector collection."
            ),
        )
        _ = (
            module.add_type[DenseCollection[2]]("DenseCollection2")
            .def_py_init[DenseCollection[2].py_init]()
            .def_method[DenseCollection[2].set]("set")
            .def_method[DenseCollection[2].set_batch]("set_batch")
            .def_method[DenseCollection[2].set_text]("set_text")
            .def_method[DenseCollection[2].set_text_only]("set_text_only")
            .def_method[DenseCollection[2].enable_graph]("enable_graph")
            .def_method[DenseCollection[2].add_edge]("add_edge")
            .def_method[DenseCollection[2].remove_edge]("remove_edge")
            .def_method[DenseCollection[2].neighbors]("neighbors")
            .def_method[DenseCollection[2].traverse]("traverse")
            .def_method[DenseCollection[2].has_path]("has_path")
            .def_method[DenseCollection[2].shortest_path]("shortest_path")
            .def_method[DenseCollection[2].get]("get")
            .def_method[DenseCollection[2].get_text]("get_text")
            .def_method[DenseCollection[2].records]("records")
            .def_method[DenseCollection[2].live_ids]("live_ids")
            .def_method[DenseCollection[2].batch_metadata]("batch_metadata")
            .def_method[DenseCollection[2].filter_matching_ids]("filter_matching_ids")
            .def_method[DenseCollection[2].metadata_for_id]("metadata_for_id")
            .def_method[DenseCollection[2].delete]("delete")
            .def_method[DenseCollection[2].supersede]("supersede")
            .def_method[DenseCollection[2].search]("search")
            .def_method[DenseCollection[2].search_exact_ids]("search_exact_ids")
            .def_method[DenseCollection[2].search_filtered_ids](
                "search_filtered_ids"
            )
            .def_method[DenseCollection[2].search_filtered_by_bitmap]("search_filtered_by_bitmap")
            .def_method[DenseCollection[2].index_metadata]("index_metadata")
            .def_method[DenseCollection[2].index_metadata_batch]("index_metadata_batch")
            .def_method[DenseCollection[2].set_sparse]("set_sparse")
            .def_method[DenseCollection[2].search_text]("search_text")
                        .def_method[DenseCollection[2].search_hybrid]("search_hybrid")
            .def_method[DenseCollection[2].flush]("flush")
            .def_method[DenseCollection[2].needs_compaction]("needs_compaction")
            .def_method[DenseCollection[2].len]("len")
        )
        _ = (
            module.add_type[DenseCollection[128]]("DenseCollection128")
            .def_py_init[DenseCollection[128].py_init]()
            .def_method[DenseCollection[128].set]("set")
            .def_method[DenseCollection[128].set_batch]("set_batch")
            .def_method[DenseCollection[128].set_text]("set_text")
            .def_method[DenseCollection[128].set_text_only]("set_text_only")
            .def_method[DenseCollection[128].enable_graph]("enable_graph")
            .def_method[DenseCollection[128].add_edge]("add_edge")
            .def_method[DenseCollection[128].remove_edge]("remove_edge")
            .def_method[DenseCollection[128].neighbors]("neighbors")
            .def_method[DenseCollection[128].traverse]("traverse")
            .def_method[DenseCollection[128].has_path]("has_path")
            .def_method[DenseCollection[128].shortest_path]("shortest_path")
            .def_method[DenseCollection[128].get]("get")
            .def_method[DenseCollection[128].get_text]("get_text")
            .def_method[DenseCollection[128].records]("records")
            .def_method[DenseCollection[128].live_ids]("live_ids")
            .def_method[DenseCollection[128].batch_metadata]("batch_metadata")
            .def_method[DenseCollection[128].filter_matching_ids]("filter_matching_ids")
            .def_method[DenseCollection[128].metadata_for_id]("metadata_for_id")
            .def_method[DenseCollection[128].delete]("delete")
            .def_method[DenseCollection[128].supersede]("supersede")
            .def_method[DenseCollection[128].search]("search")
            .def_method[DenseCollection[128].search_exact_ids](
                "search_exact_ids"
            )
            .def_method[DenseCollection[128].search_filtered_ids](
                "search_filtered_ids"
            )
            .def_method[DenseCollection[128].search_filtered_by_bitmap]("search_filtered_by_bitmap")
            .def_method[DenseCollection[128].index_metadata]("index_metadata")
            .def_method[DenseCollection[128].index_metadata_batch]("index_metadata_batch")
            .def_method[DenseCollection[128].set_sparse]("set_sparse")
            .def_method[DenseCollection[128].search_text]("search_text")
                        .def_method[DenseCollection[128].search_hybrid]("search_hybrid")
            .def_method[DenseCollection[128].flush]("flush")
            .def_method[DenseCollection[128].needs_compaction]("needs_compaction")
            .def_method[DenseCollection[128].len]("len")
        )
        _ = (
            module.add_type[DenseCollection[384]]("DenseCollection384")
            .def_py_init[DenseCollection[384].py_init]()
            .def_method[DenseCollection[384].set]("set")
            .def_method[DenseCollection[384].set_batch]("set_batch")
            .def_method[DenseCollection[384].set_text]("set_text")
            .def_method[DenseCollection[384].set_text_only]("set_text_only")
            .def_method[DenseCollection[384].enable_graph]("enable_graph")
            .def_method[DenseCollection[384].add_edge]("add_edge")
            .def_method[DenseCollection[384].remove_edge]("remove_edge")
            .def_method[DenseCollection[384].neighbors]("neighbors")
            .def_method[DenseCollection[384].traverse]("traverse")
            .def_method[DenseCollection[384].has_path]("has_path")
            .def_method[DenseCollection[384].shortest_path]("shortest_path")
            .def_method[DenseCollection[384].get]("get")
            .def_method[DenseCollection[384].get_text]("get_text")
            .def_method[DenseCollection[384].records]("records")
            .def_method[DenseCollection[384].live_ids]("live_ids")
            .def_method[DenseCollection[384].batch_metadata]("batch_metadata")
            .def_method[DenseCollection[384].filter_matching_ids]("filter_matching_ids")
            .def_method[DenseCollection[384].metadata_for_id]("metadata_for_id")
            .def_method[DenseCollection[384].delete]("delete")
            .def_method[DenseCollection[384].supersede]("supersede")
            .def_method[DenseCollection[384].search]("search")
            .def_method[DenseCollection[384].search_exact_ids](
                "search_exact_ids"
            )
            .def_method[DenseCollection[384].search_filtered_ids](
                "search_filtered_ids"
            )
            .def_method[DenseCollection[384].search_filtered_by_bitmap]("search_filtered_by_bitmap")
            .def_method[DenseCollection[384].index_metadata]("index_metadata")
            .def_method[DenseCollection[384].index_metadata_batch]("index_metadata_batch")
            .def_method[DenseCollection[384].set_sparse]("set_sparse")
            .def_method[DenseCollection[384].search_text]("search_text")
                        .def_method[DenseCollection[384].search_hybrid]("search_hybrid")
            .def_method[DenseCollection[384].flush]("flush")
            .def_method[DenseCollection[384].needs_compaction]("needs_compaction")
            .def_method[DenseCollection[384].len]("len")
        )
        _ = (
            module.add_type[DenseCollection[256]]("DenseCollection256")
            .def_py_init[DenseCollection[256].py_init]()
            .def_method[DenseCollection[256].set]("set")
            .def_method[DenseCollection[256].set_batch]("set_batch")
            .def_method[DenseCollection[256].set_text]("set_text")
            .def_method[DenseCollection[256].set_text_only]("set_text_only")
            .def_method[DenseCollection[256].enable_graph]("enable_graph")
            .def_method[DenseCollection[256].add_edge]("add_edge")
            .def_method[DenseCollection[256].remove_edge]("remove_edge")
            .def_method[DenseCollection[256].neighbors]("neighbors")
            .def_method[DenseCollection[256].traverse]("traverse")
            .def_method[DenseCollection[256].has_path]("has_path")
            .def_method[DenseCollection[256].shortest_path]("shortest_path")
            .def_method[DenseCollection[256].get]("get")
            .def_method[DenseCollection[256].get_text]("get_text")
            .def_method[DenseCollection[256].records]("records")
            .def_method[DenseCollection[256].live_ids]("live_ids")
            .def_method[DenseCollection[256].batch_metadata]("batch_metadata")
            .def_method[DenseCollection[256].filter_matching_ids]("filter_matching_ids")
            .def_method[DenseCollection[256].metadata_for_id]("metadata_for_id")
            .def_method[DenseCollection[256].delete]("delete")
            .def_method[DenseCollection[256].supersede]("supersede")
            .def_method[DenseCollection[256].search]("search")
            .def_method[DenseCollection[256].search_exact_ids](
                "search_exact_ids"
            )
            .def_method[DenseCollection[256].search_filtered_ids](
                "search_filtered_ids"
            )
            .def_method[DenseCollection[256].search_filtered_by_bitmap]("search_filtered_by_bitmap")
            .def_method[DenseCollection[256].index_metadata]("index_metadata")
            .def_method[DenseCollection[256].index_metadata_batch]("index_metadata_batch")
            .def_method[DenseCollection[256].set_sparse]("set_sparse")
            .def_method[DenseCollection[256].search_text]("search_text")
                        .def_method[DenseCollection[256].search_hybrid]("search_hybrid")
            .def_method[DenseCollection[256].flush]("flush")
            .def_method[DenseCollection[256].needs_compaction]("needs_compaction")
            .def_method[DenseCollection[256].len]("len")
        )
        _ = (
            module.add_type[DenseCollection[512]]("DenseCollection512")
            .def_py_init[DenseCollection[512].py_init]()
            .def_method[DenseCollection[512].set]("set")
            .def_method[DenseCollection[512].set_batch]("set_batch")
            .def_method[DenseCollection[512].set_text]("set_text")
            .def_method[DenseCollection[512].set_text_only]("set_text_only")
            .def_method[DenseCollection[512].enable_graph]("enable_graph")
            .def_method[DenseCollection[512].add_edge]("add_edge")
            .def_method[DenseCollection[512].remove_edge]("remove_edge")
            .def_method[DenseCollection[512].neighbors]("neighbors")
            .def_method[DenseCollection[512].traverse]("traverse")
            .def_method[DenseCollection[512].has_path]("has_path")
            .def_method[DenseCollection[512].shortest_path]("shortest_path")
            .def_method[DenseCollection[512].get]("get")
            .def_method[DenseCollection[512].get_text]("get_text")
            .def_method[DenseCollection[512].records]("records")
            .def_method[DenseCollection[512].live_ids]("live_ids")
            .def_method[DenseCollection[512].batch_metadata]("batch_metadata")
            .def_method[DenseCollection[512].filter_matching_ids]("filter_matching_ids")
            .def_method[DenseCollection[512].metadata_for_id]("metadata_for_id")
            .def_method[DenseCollection[512].delete]("delete")
            .def_method[DenseCollection[512].supersede]("supersede")
            .def_method[DenseCollection[512].search]("search")
            .def_method[DenseCollection[512].search_exact_ids](
                "search_exact_ids"
            )
            .def_method[DenseCollection[512].search_filtered_ids](
                "search_filtered_ids"
            )
            .def_method[DenseCollection[512].search_filtered_by_bitmap]("search_filtered_by_bitmap")
            .def_method[DenseCollection[512].index_metadata]("index_metadata")
            .def_method[DenseCollection[512].index_metadata_batch]("index_metadata_batch")
            .def_method[DenseCollection[512].set_sparse]("set_sparse")
            .def_method[DenseCollection[512].search_text]("search_text")
                        .def_method[DenseCollection[512].search_hybrid]("search_hybrid")
            .def_method[DenseCollection[512].flush]("flush")
            .def_method[DenseCollection[512].needs_compaction]("needs_compaction")
            .def_method[DenseCollection[512].len]("len")
        )
        _ = (
            module.add_type[DenseCollection[768]]("DenseCollection768")
            .def_py_init[DenseCollection[768].py_init]()
            .def_method[DenseCollection[768].set]("set")
            .def_method[DenseCollection[768].set_batch]("set_batch")
            .def_method[DenseCollection[768].set_text]("set_text")
            .def_method[DenseCollection[768].set_text_only]("set_text_only")
            .def_method[DenseCollection[768].enable_graph]("enable_graph")
            .def_method[DenseCollection[768].add_edge]("add_edge")
            .def_method[DenseCollection[768].remove_edge]("remove_edge")
            .def_method[DenseCollection[768].neighbors]("neighbors")
            .def_method[DenseCollection[768].traverse]("traverse")
            .def_method[DenseCollection[768].has_path]("has_path")
            .def_method[DenseCollection[768].shortest_path]("shortest_path")
            .def_method[DenseCollection[768].get]("get")
            .def_method[DenseCollection[768].get_text]("get_text")
            .def_method[DenseCollection[768].records]("records")
            .def_method[DenseCollection[768].live_ids]("live_ids")
            .def_method[DenseCollection[768].batch_metadata]("batch_metadata")
            .def_method[DenseCollection[768].filter_matching_ids]("filter_matching_ids")
            .def_method[DenseCollection[768].metadata_for_id]("metadata_for_id")
            .def_method[DenseCollection[768].delete]("delete")
            .def_method[DenseCollection[768].supersede]("supersede")
            .def_method[DenseCollection[768].search]("search")
            .def_method[DenseCollection[768].search_exact_ids](
                "search_exact_ids"
            )
            .def_method[DenseCollection[768].search_filtered_ids](
                "search_filtered_ids"
            )
            .def_method[DenseCollection[768].search_filtered_by_bitmap]("search_filtered_by_bitmap")
            .def_method[DenseCollection[768].index_metadata]("index_metadata")
            .def_method[DenseCollection[768].index_metadata_batch]("index_metadata_batch")
            .def_method[DenseCollection[768].set_sparse]("set_sparse")
            .def_method[DenseCollection[768].search_text]("search_text")
                        .def_method[DenseCollection[768].search_hybrid]("search_hybrid")
            .def_method[DenseCollection[768].flush]("flush")
            .def_method[DenseCollection[768].needs_compaction]("needs_compaction")
            .def_method[DenseCollection[768].len]("len")
        )
        _ = (
            module.add_type[DenseCollection[1024]]("DenseCollection1024")
            .def_py_init[DenseCollection[1024].py_init]()
            .def_method[DenseCollection[1024].set]("set")
            .def_method[DenseCollection[1024].set_batch]("set_batch")
            .def_method[DenseCollection[1024].set_text]("set_text")
            .def_method[DenseCollection[1024].set_text_only]("set_text_only")
            .def_method[DenseCollection[1024].enable_graph]("enable_graph")
            .def_method[DenseCollection[1024].add_edge]("add_edge")
            .def_method[DenseCollection[1024].remove_edge]("remove_edge")
            .def_method[DenseCollection[1024].neighbors]("neighbors")
            .def_method[DenseCollection[1024].traverse]("traverse")
            .def_method[DenseCollection[1024].has_path]("has_path")
            .def_method[DenseCollection[1024].shortest_path]("shortest_path")
            .def_method[DenseCollection[1024].get]("get")
            .def_method[DenseCollection[1024].get_text]("get_text")
            .def_method[DenseCollection[1024].records]("records")
            .def_method[DenseCollection[1024].live_ids]("live_ids")
            .def_method[DenseCollection[1024].batch_metadata]("batch_metadata")
            .def_method[DenseCollection[1024].filter_matching_ids]("filter_matching_ids")
            .def_method[DenseCollection[1024].metadata_for_id]("metadata_for_id")
            .def_method[DenseCollection[1024].delete]("delete")
            .def_method[DenseCollection[1024].supersede]("supersede")
            .def_method[DenseCollection[1024].search]("search")
            .def_method[DenseCollection[1024].search_exact_ids](
                "search_exact_ids"
            )
            .def_method[DenseCollection[1024].search_filtered_ids](
                "search_filtered_ids"
            )
            .def_method[DenseCollection[1024].search_filtered_by_bitmap]("search_filtered_by_bitmap")
            .def_method[DenseCollection[1024].index_metadata]("index_metadata")
            .def_method[DenseCollection[1024].index_metadata_batch]("index_metadata_batch")
            .def_method[DenseCollection[1024].set_sparse]("set_sparse")
            .def_method[DenseCollection[1024].search_text]("search_text")
                        .def_method[DenseCollection[1024].search_hybrid]("search_hybrid")
            .def_method[DenseCollection[1024].flush]("flush")
            .def_method[DenseCollection[1024].needs_compaction]("needs_compaction")
            .def_method[DenseCollection[1024].len]("len")
        )
        _ = (
            module.add_type[DenseCollection[1536]]("DenseCollection1536")
            .def_py_init[DenseCollection[1536].py_init]()
            .def_method[DenseCollection[1536].set]("set")
            .def_method[DenseCollection[1536].set_batch]("set_batch")
            .def_method[DenseCollection[1536].set_text]("set_text")
            .def_method[DenseCollection[1536].set_text_only]("set_text_only")
            .def_method[DenseCollection[1536].enable_graph]("enable_graph")
            .def_method[DenseCollection[1536].add_edge]("add_edge")
            .def_method[DenseCollection[1536].remove_edge]("remove_edge")
            .def_method[DenseCollection[1536].neighbors]("neighbors")
            .def_method[DenseCollection[1536].traverse]("traverse")
            .def_method[DenseCollection[1536].has_path]("has_path")
            .def_method[DenseCollection[1536].shortest_path]("shortest_path")
            .def_method[DenseCollection[1536].get]("get")
            .def_method[DenseCollection[1536].get_text]("get_text")
            .def_method[DenseCollection[1536].records]("records")
            .def_method[DenseCollection[1536].live_ids]("live_ids")
            .def_method[DenseCollection[1536].batch_metadata]("batch_metadata")
            .def_method[DenseCollection[1536].filter_matching_ids]("filter_matching_ids")
            .def_method[DenseCollection[1536].metadata_for_id]("metadata_for_id")
            .def_method[DenseCollection[1536].delete]("delete")
            .def_method[DenseCollection[1536].supersede]("supersede")
            .def_method[DenseCollection[1536].search]("search")
            .def_method[DenseCollection[1536].search_exact_ids](
                "search_exact_ids"
            )
            .def_method[DenseCollection[1536].search_filtered_ids](
                "search_filtered_ids"
            )
            .def_method[DenseCollection[1536].search_filtered_by_bitmap]("search_filtered_by_bitmap")
            .def_method[DenseCollection[1536].index_metadata]("index_metadata")
            .def_method[DenseCollection[1536].index_metadata_batch]("index_metadata_batch")
            .def_method[DenseCollection[1536].set_sparse]("set_sparse")
            .def_method[DenseCollection[1536].search_text]("search_text")
                        .def_method[DenseCollection[1536].search_hybrid]("search_hybrid")
            .def_method[DenseCollection[1536].flush]("flush")
            .def_method[DenseCollection[1536].needs_compaction]("needs_compaction")
            .def_method[DenseCollection[1536].len]("len")
        )
        _ = (
            module.add_type[DenseCollection[3072]]("DenseCollection3072")
            .def_py_init[DenseCollection[3072].py_init]()
            .def_method[DenseCollection[3072].set]("set")
            .def_method[DenseCollection[3072].set_batch]("set_batch")
            .def_method[DenseCollection[3072].set_text]("set_text")
            .def_method[DenseCollection[3072].set_text_only]("set_text_only")
            .def_method[DenseCollection[3072].enable_graph]("enable_graph")
            .def_method[DenseCollection[3072].add_edge]("add_edge")
            .def_method[DenseCollection[3072].remove_edge]("remove_edge")
            .def_method[DenseCollection[3072].neighbors]("neighbors")
            .def_method[DenseCollection[3072].traverse]("traverse")
            .def_method[DenseCollection[3072].has_path]("has_path")
            .def_method[DenseCollection[3072].shortest_path]("shortest_path")
            .def_method[DenseCollection[3072].get]("get")
            .def_method[DenseCollection[3072].get_text]("get_text")
            .def_method[DenseCollection[3072].records]("records")
            .def_method[DenseCollection[3072].live_ids]("live_ids")
            .def_method[DenseCollection[3072].batch_metadata]("batch_metadata")
            .def_method[DenseCollection[3072].filter_matching_ids]("filter_matching_ids")
            .def_method[DenseCollection[3072].metadata_for_id]("metadata_for_id")
            .def_method[DenseCollection[3072].delete]("delete")
            .def_method[DenseCollection[3072].supersede]("supersede")
            .def_method[DenseCollection[3072].search]("search")
            .def_method[DenseCollection[3072].search_exact_ids](
                "search_exact_ids"
            )
            .def_method[DenseCollection[3072].search_filtered_ids](
                "search_filtered_ids"
            )
            .def_method[DenseCollection[3072].search_filtered_by_bitmap]("search_filtered_by_bitmap")
            .def_method[DenseCollection[3072].index_metadata]("index_metadata")
            .def_method[DenseCollection[3072].index_metadata_batch]("index_metadata_batch")
            .def_method[DenseCollection[3072].set_sparse]("set_sparse")
            .def_method[DenseCollection[3072].search_text]("search_text")
                        .def_method[DenseCollection[3072].search_hybrid]("search_hybrid")
            .def_method[DenseCollection[3072].flush]("flush")
            .def_method[DenseCollection[3072].needs_compaction]("needs_compaction")
            .def_method[DenseCollection[3072].len]("len")
        )
        _ = (
            module.add_type[DynamicFlatCollection]("DynamicFlatCollection")
            .def_py_init[DynamicFlatCollection.py_init]()
            .def_method[DynamicFlatCollection.set]("set")
            .def_method[DynamicFlatCollection.get]("get")
            .def_method[DynamicFlatCollection.get_text]("get_text")
            .def_method[DynamicFlatCollection.records]("records")
            .def_method[DynamicFlatCollection.live_ids]("live_ids")
            .def_method[DynamicFlatCollection.batch_metadata]("batch_metadata")
            .def_method[DynamicFlatCollection.filter_matching_ids]("filter_matching_ids")
            .def_method[DynamicFlatCollection.metadata_for_id]("metadata_for_id")
            .def_method[DynamicFlatCollection.delete]("delete")
            .def_method[DynamicFlatCollection.search]("search")
            .def_method[DynamicFlatCollection.search_exact_ids](
                "search_exact_ids"
            )
            .def_method[DynamicFlatCollection.search_filtered_ids](
                "search_filtered_ids"
            )
            .def_method[DynamicFlatCollection.search_text]("search_text")
            .def_method[DynamicFlatCollection.flush]("flush")
            .def_method[DynamicFlatCollection.len]("len")
        )
        _ = (
            module.add_type[MultiVectorCollection[2]]("MultiVectorCollection2")
            .def_py_init[MultiVectorCollection[2].py_init]()
            .def_method[MultiVectorCollection[2].set]("set")
            .def_method[MultiVectorCollection[2].set_text]("set_text")
            .def_method[MultiVectorCollection[2].get]("get")
            .def_method[MultiVectorCollection[2].get_text]("get_text")
            .def_method[MultiVectorCollection[2].records]("records")
            .def_method[MultiVectorCollection[2].live_ids]("live_ids")
            .def_method[MultiVectorCollection[2].batch_metadata]("batch_metadata")
            .def_method[MultiVectorCollection[2].filter_matching_ids]("filter_matching_ids")
            .def_method[MultiVectorCollection[2].metadata_for_id]("metadata_for_id")
            .def_method[MultiVectorCollection[2].delete]("delete")
            .def_method[MultiVectorCollection[2].search]("search")
            .def_method[MultiVectorCollection[2].search_text]("search_text")
            .def_method[MultiVectorCollection[2].flush]("flush")
            .def_method[MultiVectorCollection[2].len]("len")
        )
        _ = (
            module.add_type[MultiVectorCollection[48]](
                "MultiVectorCollection48"
            )
            .def_py_init[MultiVectorCollection[48].py_init]()
            .def_method[MultiVectorCollection[48].set]("set")
            .def_method[MultiVectorCollection[48].set_text]("set_text")
            .def_method[MultiVectorCollection[48].get]("get")
            .def_method[MultiVectorCollection[48].get_text]("get_text")
            .def_method[MultiVectorCollection[48].records]("records")
            .def_method[MultiVectorCollection[48].live_ids]("live_ids")
            .def_method[MultiVectorCollection[48].batch_metadata]("batch_metadata")
            .def_method[MultiVectorCollection[48].filter_matching_ids]("filter_matching_ids")
            .def_method[MultiVectorCollection[48].metadata_for_id]("metadata_for_id")
            .def_method[MultiVectorCollection[48].delete]("delete")
            .def_method[MultiVectorCollection[48].search]("search")
            .def_method[MultiVectorCollection[48].search_text]("search_text")
            .def_method[MultiVectorCollection[48].flush]("flush")
            .def_method[MultiVectorCollection[48].len]("len")
        )
        _ = (
            module.add_type[MultiVectorMuveraCollection[2]](
                "MultiVectorMuveraCollection2"
            )
            .def_py_init[MultiVectorMuveraCollection[2].py_init]()
            .def_method[MultiVectorMuveraCollection[2].set]("set")
            .def_method[MultiVectorMuveraCollection[2].set_text]("set_text")
            .def_method[MultiVectorMuveraCollection[2].get]("get")
            .def_method[MultiVectorMuveraCollection[2].get_text]("get_text")
            .def_method[MultiVectorMuveraCollection[2].records]("records")
            .def_method[MultiVectorMuveraCollection[2].live_ids]("live_ids")
            .def_method[MultiVectorMuveraCollection[2].batch_metadata]("batch_metadata")
            .def_method[MultiVectorMuveraCollection[2].filter_matching_ids]("filter_matching_ids")
            .def_method[MultiVectorMuveraCollection[2].metadata_for_id]("metadata_for_id")
            .def_method[MultiVectorMuveraCollection[2].delete]("delete")
            .def_method[MultiVectorMuveraCollection[2].search]("search")
            .def_method[MultiVectorMuveraCollection[2].search_text](
                "search_text"
            )
            .def_method[MultiVectorMuveraCollection[2].flush]("flush")
            .def_method[MultiVectorMuveraCollection[2].len]("len")
        )
        _ = (
            module.add_type[MultiVectorMuveraCollection[48]](
                "MultiVectorMuveraCollection48"
            )
            .def_py_init[MultiVectorMuveraCollection[48].py_init]()
            .def_method[MultiVectorMuveraCollection[48].set]("set")
            .def_method[MultiVectorMuveraCollection[48].set_text]("set_text")
            .def_method[MultiVectorMuveraCollection[48].get]("get")
            .def_method[MultiVectorMuveraCollection[48].get_text]("get_text")
            .def_method[MultiVectorMuveraCollection[48].records]("records")
            .def_method[MultiVectorMuveraCollection[48].live_ids]("live_ids")
            .def_method[MultiVectorMuveraCollection[48].batch_metadata]("batch_metadata")
            .def_method[MultiVectorMuveraCollection[48].filter_matching_ids]("filter_matching_ids")
            .def_method[MultiVectorMuveraCollection[48].metadata_for_id]("metadata_for_id")
            .def_method[MultiVectorMuveraCollection[48].delete]("delete")
            .def_method[MultiVectorMuveraCollection[48].search]("search")
            .def_method[MultiVectorMuveraCollection[48].search_text](
                "search_text"
            )
            .def_method[MultiVectorMuveraCollection[48].flush]("flush")
            .def_method[MultiVectorMuveraCollection[48].len]("len")
        )
        _ = (
            module.add_type[MultiVectorCollection[128]](
                "MultiVectorCollection128"
            )
            .def_py_init[MultiVectorCollection[128].py_init]()
            .def_method[MultiVectorCollection[128].set]("set")
            .def_method[MultiVectorCollection[128].set_text]("set_text")
            .def_method[MultiVectorCollection[128].get]("get")
            .def_method[MultiVectorCollection[128].get_text]("get_text")
            .def_method[MultiVectorCollection[128].records]("records")
            .def_method[MultiVectorCollection[128].live_ids]("live_ids")
            .def_method[MultiVectorCollection[128].batch_metadata]("batch_metadata")
            .def_method[MultiVectorCollection[128].filter_matching_ids]("filter_matching_ids")
            .def_method[MultiVectorCollection[128].metadata_for_id]("metadata_for_id")
            .def_method[MultiVectorCollection[128].delete]("delete")
            .def_method[MultiVectorCollection[128].search]("search")
            .def_method[MultiVectorCollection[128].search_text]("search_text")
            .def_method[MultiVectorCollection[128].flush]("flush")
            .def_method[MultiVectorCollection[128].len]("len")
        )
        _ = (
            module.add_type[MultiVectorMuveraCollection[128]](
                "MultiVectorMuveraCollection128"
            )
            .def_py_init[MultiVectorMuveraCollection[128].py_init]()
            .def_method[MultiVectorMuveraCollection[128].set]("set")
            .def_method[MultiVectorMuveraCollection[128].set_text]("set_text")
            .def_method[MultiVectorMuveraCollection[128].get]("get")
            .def_method[MultiVectorMuveraCollection[128].get_text]("get_text")
            .def_method[MultiVectorMuveraCollection[128].records]("records")
            .def_method[MultiVectorMuveraCollection[128].live_ids]("live_ids")
            .def_method[MultiVectorMuveraCollection[128].batch_metadata]("batch_metadata")
            .def_method[MultiVectorMuveraCollection[128].filter_matching_ids]("filter_matching_ids")
            .def_method[MultiVectorMuveraCollection[128].metadata_for_id]("metadata_for_id")
            .def_method[MultiVectorMuveraCollection[128].delete]("delete")
            .def_method[MultiVectorMuveraCollection[128].search]("search")
            .def_method[MultiVectorMuveraCollection[128].search_text](
                "search_text"
            )
            .def_method[MultiVectorMuveraCollection[128].flush]("flush")
            .def_method[MultiVectorMuveraCollection[128].len]("len")
        )
        return module.finalize()
    except e:
        abort(String("error creating OmenDB Mojo module: ", e))


def engine_version() raises -> PythonObject:
    return PythonObject("0.1.0")


def dense_collection2_memory(
    m_obj: PythonObject,
    ef_construction_obj: PythonObject,
    ef_search_obj: PythonObject,
    alpha_obj: PythonObject,
    metric_obj: PythonObject,
) raises -> PythonObject:
    return PythonObject(
        alloc=DenseCollection[2].memory(
            _options_from_python(
                m_obj, ef_construction_obj, ef_search_obj, alpha_obj, metric_obj
            )
        )
    )


def dense_collection2_create(
    path_obj: PythonObject,
    m_obj: PythonObject,
    ef_construction_obj: PythonObject,
    ef_search_obj: PythonObject,
    alpha_obj: PythonObject,
    metric_obj: PythonObject,
) raises -> PythonObject:
    return PythonObject(
        alloc=DenseCollection[2].create(
            String(py=path_obj),
            _options_from_python(
                m_obj, ef_construction_obj, ef_search_obj, alpha_obj, metric_obj
            ),
        )
    )


def dense_collection2_open(path_obj: PythonObject) raises -> PythonObject:
    return PythonObject(alloc=DenseCollection[2].open(String(py=path_obj)))


def dense_collection128_memory(
    m_obj: PythonObject,
    ef_construction_obj: PythonObject,
    ef_search_obj: PythonObject,
    alpha_obj: PythonObject,
    metric_obj: PythonObject,
) raises -> PythonObject:
    return PythonObject(
        alloc=DenseCollection[128].memory(
            _options_from_python(
                m_obj, ef_construction_obj, ef_search_obj, alpha_obj, metric_obj
            )
        )
    )


def dense_collection128_create(
    path_obj: PythonObject,
    m_obj: PythonObject,
    ef_construction_obj: PythonObject,
    ef_search_obj: PythonObject,
    alpha_obj: PythonObject,
    metric_obj: PythonObject,
) raises -> PythonObject:
    return PythonObject(
        alloc=DenseCollection[128].create(
            String(py=path_obj),
            _options_from_python(
                m_obj, ef_construction_obj, ef_search_obj, alpha_obj, metric_obj
            ),
        )
    )


def dense_collection128_open(path_obj: PythonObject) raises -> PythonObject:
    return PythonObject(alloc=DenseCollection[128].open(String(py=path_obj)))


def dense_collection384_memory(
    m_obj: PythonObject,
    ef_construction_obj: PythonObject,
    ef_search_obj: PythonObject,
    alpha_obj: PythonObject,
    metric_obj: PythonObject,
) raises -> PythonObject:
    return PythonObject(
        alloc=DenseCollection[384].memory(
            _options_from_python(
                m_obj, ef_construction_obj, ef_search_obj, alpha_obj, metric_obj
            )
        )
    )


def dense_collection384_create(
    path_obj: PythonObject,
    m_obj: PythonObject,
    ef_construction_obj: PythonObject,
    ef_search_obj: PythonObject,
    alpha_obj: PythonObject,
    metric_obj: PythonObject,
) raises -> PythonObject:
    return PythonObject(
        alloc=DenseCollection[384].create(
            String(py=path_obj),
            _options_from_python(
                m_obj, ef_construction_obj, ef_search_obj, alpha_obj, metric_obj
            ),
        )
    )


def _options_from_python(
    m_obj: PythonObject,
    ef_construction_obj: PythonObject,
    ef_search_obj: PythonObject,
    alpha_obj: PythonObject,
    metric_obj: PythonObject,
) raises -> VectorStoreOptions:
    var metric = Metric.L2
    var metric_str = String(py=metric_obj)
    if metric_str == "cosine":
        metric = Metric.COSINE
    elif metric_str == "dot":
        metric = Metric.DOT
    return VectorStoreOptions(
        hnsw=HNSWParams(
            M=Int(py=m_obj),
            ef_construction=Int(py=ef_construction_obj),
            ef_search=Int(py=ef_search_obj),
            alpha=Float32(py=alpha_obj),
        ),
        metric=metric,
    )


def dense_collection384_open(path_obj: PythonObject) raises -> PythonObject:
    return PythonObject(alloc=DenseCollection[384].open(String(py=path_obj)))


def dense_collection256_memory(
    m_obj: PythonObject,
    ef_construction_obj: PythonObject,
    ef_search_obj: PythonObject,
    alpha_obj: PythonObject,
    metric_obj: PythonObject,
) raises -> PythonObject:
    return PythonObject(
        alloc=DenseCollection[256].memory(
            _options_from_python(
                m_obj, ef_construction_obj, ef_search_obj, alpha_obj, metric_obj
            )
        )
    )


def dense_collection256_create(
    path_obj: PythonObject,
    m_obj: PythonObject,
    ef_construction_obj: PythonObject,
    ef_search_obj: PythonObject,
    alpha_obj: PythonObject,
    metric_obj: PythonObject,
) raises -> PythonObject:
    return PythonObject(
        alloc=DenseCollection[256].create(
            String(py=path_obj),
            _options_from_python(
                m_obj, ef_construction_obj, ef_search_obj, alpha_obj, metric_obj
            ),
        )
    )


def dense_collection256_open(path_obj: PythonObject) raises -> PythonObject:
    return PythonObject(alloc=DenseCollection[256].open(String(py=path_obj)))


def dense_collection512_memory(
    m_obj: PythonObject,
    ef_construction_obj: PythonObject,
    ef_search_obj: PythonObject,
    alpha_obj: PythonObject,
    metric_obj: PythonObject,
) raises -> PythonObject:
    return PythonObject(
        alloc=DenseCollection[512].memory(
            _options_from_python(
                m_obj, ef_construction_obj, ef_search_obj, alpha_obj, metric_obj
            )
        )
    )


def dense_collection512_create(
    path_obj: PythonObject,
    m_obj: PythonObject,
    ef_construction_obj: PythonObject,
    ef_search_obj: PythonObject,
    alpha_obj: PythonObject,
    metric_obj: PythonObject,
) raises -> PythonObject:
    return PythonObject(
        alloc=DenseCollection[512].create(
            String(py=path_obj),
            _options_from_python(
                m_obj, ef_construction_obj, ef_search_obj, alpha_obj, metric_obj
            ),
        )
    )


def dense_collection512_open(path_obj: PythonObject) raises -> PythonObject:
    return PythonObject(alloc=DenseCollection[512].open(String(py=path_obj)))


def dense_collection768_memory(
    m_obj: PythonObject,
    ef_construction_obj: PythonObject,
    ef_search_obj: PythonObject,
    alpha_obj: PythonObject,
    metric_obj: PythonObject,
) raises -> PythonObject:
    return PythonObject(
        alloc=DenseCollection[768].memory(
            _options_from_python(
                m_obj, ef_construction_obj, ef_search_obj, alpha_obj, metric_obj
            )
        )
    )


def dense_collection768_create(
    path_obj: PythonObject,
    m_obj: PythonObject,
    ef_construction_obj: PythonObject,
    ef_search_obj: PythonObject,
    alpha_obj: PythonObject,
    metric_obj: PythonObject,
) raises -> PythonObject:
    return PythonObject(
        alloc=DenseCollection[768].create(
            String(py=path_obj),
            _options_from_python(
                m_obj, ef_construction_obj, ef_search_obj, alpha_obj, metric_obj
            ),
        )
    )


def dense_collection768_open(path_obj: PythonObject) raises -> PythonObject:
    return PythonObject(alloc=DenseCollection[768].open(String(py=path_obj)))


def dense_collection1024_memory(
    m_obj: PythonObject,
    ef_construction_obj: PythonObject,
    ef_search_obj: PythonObject,
    alpha_obj: PythonObject,
    metric_obj: PythonObject,
) raises -> PythonObject:
    return PythonObject(
        alloc=DenseCollection[1024].memory(
            _options_from_python(
                m_obj, ef_construction_obj, ef_search_obj, alpha_obj, metric_obj
            )
        )
    )


def dense_collection1024_create(
    path_obj: PythonObject,
    m_obj: PythonObject,
    ef_construction_obj: PythonObject,
    ef_search_obj: PythonObject,
    alpha_obj: PythonObject,
    metric_obj: PythonObject,
) raises -> PythonObject:
    return PythonObject(
        alloc=DenseCollection[1024].create(
            String(py=path_obj),
            _options_from_python(
                m_obj, ef_construction_obj, ef_search_obj, alpha_obj, metric_obj
            ),
        )
    )


def dense_collection1024_open(path_obj: PythonObject) raises -> PythonObject:
    return PythonObject(alloc=DenseCollection[1024].open(String(py=path_obj)))


def dense_collection1536_memory(
    m_obj: PythonObject,
    ef_construction_obj: PythonObject,
    ef_search_obj: PythonObject,
    alpha_obj: PythonObject,
    metric_obj: PythonObject,
) raises -> PythonObject:
    return PythonObject(
        alloc=DenseCollection[1536].memory(
            _options_from_python(
                m_obj, ef_construction_obj, ef_search_obj, alpha_obj, metric_obj
            )
        )
    )


def dense_collection1536_create(
    path_obj: PythonObject,
    m_obj: PythonObject,
    ef_construction_obj: PythonObject,
    ef_search_obj: PythonObject,
    alpha_obj: PythonObject,
    metric_obj: PythonObject,
) raises -> PythonObject:
    return PythonObject(
        alloc=DenseCollection[1536].create(
            String(py=path_obj),
            _options_from_python(
                m_obj, ef_construction_obj, ef_search_obj, alpha_obj, metric_obj
            ),
        )
    )


def dense_collection1536_open(path_obj: PythonObject) raises -> PythonObject:
    return PythonObject(alloc=DenseCollection[1536].open(String(py=path_obj)))


def dense_collection3072_memory(
    m_obj: PythonObject,
    ef_construction_obj: PythonObject,
    ef_search_obj: PythonObject,
    alpha_obj: PythonObject,
    metric_obj: PythonObject,
) raises -> PythonObject:
    return PythonObject(
        alloc=DenseCollection[3072].memory(
            _options_from_python(
                m_obj, ef_construction_obj, ef_search_obj, alpha_obj, metric_obj
            )
        )
    )


def dense_collection3072_create(
    path_obj: PythonObject,
    m_obj: PythonObject,
    ef_construction_obj: PythonObject,
    ef_search_obj: PythonObject,
    alpha_obj: PythonObject,
    metric_obj: PythonObject,
) raises -> PythonObject:
    return PythonObject(
        alloc=DenseCollection[3072].create(
            String(py=path_obj),
            _options_from_python(
                m_obj, ef_construction_obj, ef_search_obj, alpha_obj, metric_obj
            ),
        )
    )


def dense_collection3072_open(path_obj: PythonObject) raises -> PythonObject:
    return PythonObject(alloc=DenseCollection[3072].open(String(py=path_obj)))


def dynamic_flat_collection_memory(
    dim_obj: PythonObject,
) raises -> PythonObject:
    return PythonObject(alloc=DynamicFlatCollection.memory(Int(py=dim_obj)))


def dynamic_flat_collection_create(
    path_obj: PythonObject,
    dim_obj: PythonObject,
) raises -> PythonObject:
    return PythonObject(
        alloc=DynamicFlatCollection.create(String(py=path_obj), Int(py=dim_obj))
    )


def dynamic_flat_collection_open(
    path_obj: PythonObject,
) raises -> PythonObject:
    return PythonObject(alloc=DynamicFlatCollection.open(String(py=path_obj)))


def multivector_collection2_memory() raises -> PythonObject:
    return PythonObject(alloc=MultiVectorCollection[2].memory())


def multivector_collection2_create(
    path_obj: PythonObject,
) raises -> PythonObject:
    return PythonObject(
        alloc=MultiVectorCollection[2].create(String(py=path_obj))
    )


def multivector_collection2_open(path_obj: PythonObject) raises -> PythonObject:
    return PythonObject(
        alloc=MultiVectorCollection[2].open(String(py=path_obj))
    )


def multivector_collection2_muvera_memory() raises -> PythonObject:
    return PythonObject(alloc=MultiVectorMuveraCollection[2].memory())


def multivector_collection2_muvera_create(
    path_obj: PythonObject,
) raises -> PythonObject:
    return PythonObject(
        alloc=MultiVectorMuveraCollection[2].create(String(py=path_obj))
    )


def multivector_collection2_muvera_open(
    path_obj: PythonObject,
) raises -> PythonObject:
    return PythonObject(
        alloc=MultiVectorMuveraCollection[2].open(String(py=path_obj))
    )


def multivector_collection48_memory() raises -> PythonObject:
    return PythonObject(alloc=MultiVectorCollection[48].memory())


def multivector_collection48_create(
    path_obj: PythonObject,
) raises -> PythonObject:
    return PythonObject(
        alloc=MultiVectorCollection[48].create(String(py=path_obj))
    )


def multivector_collection48_open(
    path_obj: PythonObject,
) raises -> PythonObject:
    return PythonObject(
        alloc=MultiVectorCollection[48].open(String(py=path_obj))
    )


def multivector_collection48_muvera_memory() raises -> PythonObject:
    return PythonObject(alloc=MultiVectorMuveraCollection[48].memory())


def multivector_collection48_muvera_create(
    path_obj: PythonObject,
) raises -> PythonObject:
    return PythonObject(
        alloc=MultiVectorMuveraCollection[48].create(String(py=path_obj))
    )


def multivector_collection48_muvera_open(
    path_obj: PythonObject,
) raises -> PythonObject:
    return PythonObject(
        alloc=MultiVectorMuveraCollection[48].open(String(py=path_obj))
    )


def multivector_collection128_memory() raises -> PythonObject:
    return PythonObject(alloc=MultiVectorCollection[128].memory())


def multivector_collection128_create(
    path_obj: PythonObject,
) raises -> PythonObject:
    return PythonObject(
        alloc=MultiVectorCollection[128].create(String(py=path_obj))
    )


def multivector_collection128_open(
    path_obj: PythonObject,
) raises -> PythonObject:
    return PythonObject(
        alloc=MultiVectorCollection[128].open(String(py=path_obj))
    )


def multivector_collection128_muvera_memory() raises -> PythonObject:
    return PythonObject(alloc=MultiVectorMuveraCollection[128].memory())


def multivector_collection128_muvera_create(
    path_obj: PythonObject,
) raises -> PythonObject:
    return PythonObject(
        alloc=MultiVectorMuveraCollection[128].create(String(py=path_obj))
    )


def multivector_collection128_muvera_open(
    path_obj: PythonObject,
) raises -> PythonObject:
    return PythonObject(
        alloc=MultiVectorMuveraCollection[128].open(String(py=path_obj))
    )


struct DenseCollection[dim: Int](Movable, Writable):
    var store: VectorStore[Self.dim]

    def __init__(out self) raises:
        self = Self.memory()

    @staticmethod
    def memory(
        options: VectorStoreOptions = VectorStoreOptions(),
    ) raises -> Self:
        return Self(VectorStore[Self.dim].create_in_memory(options))

    @staticmethod
    def create(
        path: String, options: VectorStoreOptions = VectorStoreOptions()
    ) raises -> Self:
        return Self(VectorStore[Self.dim].create(path, options))

    @staticmethod
    def open(path: String) raises -> Self:
        return Self(VectorStore[Self.dim].open(path))

    def __init__(out self, var store: VectorStore[Self.dim]):
        self.store = store^

    def write_to(self, mut writer: Some[Writer]):
        writer.write("DenseCollection2(len=", self.store.len(), ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    @staticmethod
    def py_init(
        out self: Self, args: PythonObject, kwargs: PythonObject
    ) raises:
        if len(args) != 0:
            raise Error("DenseCollection2() takes no positional arguments")
        self = Self()

    @staticmethod
    def set(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        id_obj: PythonObject,
        vector_obj: PythonObject,
        metadata_obj: PythonObject,
        source_obj: PythonObject,
    ) raises -> PythonObject:
        var vector = _vector_from_python(vector_obj, Self.dim)
        var metadata = _optional_string_from_python(metadata_obj)
        var source_span = _optional_string_from_python(source_obj)
        var vector_id = self_ptr[].store.set(
            String(py=id_obj),
            Span(ptr=vector.unsafe_ptr(), length=Self.dim),
            metadata=metadata,
            source_span=source_span,
        )
        return PythonObject(vector_id)

    @staticmethod
    def set_batch(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        ids_obj: PythonObject,
        vectors_obj: PythonObject,
        metadata_obj: PythonObject,
    ) raises -> PythonObject:
        """set_batch(ids, flat_vectors, metadata_list) -> list of vector_ids"""
        var count = len(ids_obj)
        if count == 0:
            return Python.list()
        # Build IDs list
        var ids = List[String]()
        ids.reserve(count)
        for i in range(count):
            ids.append(String(py=ids_obj[i]))
        # Build flat vector array
        var expected_len = count * Self.dim
        if len(vectors_obj) != expected_len:
            raise Error(
                "vector data length mismatch: expected "
                + String(expected_len)
                + ", got "
                + String(len(vectors_obj))
            )
        var vectors = List[Float32]()
        vectors.reserve(expected_len)
        for i in range(expected_len):
            vectors.append(Float32(py=vectors_obj[i]))
        # Build metadata list if provided
        var meta_list: Optional[List[Optional[String]]] = None
        if metadata_obj is not Python.none():
            var mlist = List[Optional[String]]()
            mlist.reserve(count)
            for i in range(count):
                if metadata_obj[i] is Python.none():
                    mlist.append(Optional[String](None))
                else:
                    mlist.append(
                        Optional[String](
                            String(py=Python.import_module("json").dumps(metadata_obj[i]))
                        )
                    )
            meta_list = Optional(mlist^)
        var result_ids = self_ptr[].store.set_batch(
            ids^, Span(ptr=vectors.unsafe_ptr(), length=expected_len), meta_list^
        )
        # Convert to Python list
        var py_result = Python.list()
        for i in range(len(result_ids)):
            py_result.append(PythonObject(Int(result_ids[i])))
        return py_result

    @staticmethod
    def set_text(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        id_obj: PythonObject,
        vector_obj: PythonObject,
        text_obj: PythonObject,
        metadata_obj: PythonObject,
        source_obj: PythonObject,
    ) raises -> PythonObject:
        var vector = _vector_from_python(vector_obj, Self.dim)
        var metadata = _optional_string_from_python(metadata_obj)
        var source_span = _optional_string_from_python(source_obj)
        _ = self_ptr[].store.set_text(
            String(py=id_obj),
            Span(ptr=vector.unsafe_ptr(), length=Self.dim),
            String(py=text_obj),
            metadata=metadata,
            source_span=source_span,
        )
        return Python.none()

    @staticmethod
    def set_text_only(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        id_obj: PythonObject,
        text_obj: PythonObject,
        metadata_obj: PythonObject,
        source_obj: PythonObject,
    ) raises -> PythonObject:
        var metadata = _optional_string_from_python(metadata_obj)
        var source_span = _optional_string_from_python(source_obj)
        _ = self_ptr[].store.set_text_only(
            String(py=id_obj),
            String(py=text_obj),
            metadata=metadata,
            source_span=source_span,
        )
        return Python.none()

    @staticmethod
    def search(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        vector_obj: PythonObject,
        k_obj: PythonObject,
        ef_obj: PythonObject,
    ) raises -> PythonObject:
        var vector = _vector_from_python(vector_obj, Self.dim)
        var results = self_ptr[].store.search(
            Span(ptr=vector.unsafe_ptr(), length=Self.dim),
            SearchOptions(k=Int(py=k_obj), ef_search=Int(py=ef_obj)),
        )

        return _search_results_to_python(results)

    @staticmethod
    def enable_graph(
        self_ptr: UnsafePointer[Self, MutAnyOrigin]
    ) raises -> PythonObject:
        self_ptr[].store.enable_graph()
        return Python.none()

    @staticmethod
    def add_edge(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        from_obj: PythonObject,
        to_obj: PythonObject,
        type_obj: PythonObject,
        weight_obj: PythonObject,
    ) raises -> PythonObject:
        var weight = Optional[Float32](None)
        if not (weight_obj is Python.none()):
            weight = Optional[Float32](Float32(py=weight_obj))
        var edge_id = self_ptr[].store.add_edge(
            String(py=from_obj),
            String(py=to_obj),
            String(py=type_obj),
            weight,
        )
        return PythonObject(Int(edge_id))

    @staticmethod
    def remove_edge(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        from_obj: PythonObject,
        to_obj: PythonObject,
        type_obj: PythonObject,
    ) raises -> PythonObject:
        return PythonObject(
            self_ptr[].store.remove_edge(
                String(py=from_obj), String(py=to_obj), String(py=type_obj)
            )
        )

    @staticmethod
    def neighbors(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        id_obj: PythonObject,
        direction_obj: PythonObject,
        type_obj: PythonObject,
    ) raises -> PythonObject:
        var edge_type = Optional[String](None)
        if not (type_obj is Python.none()):
            edge_type = Optional[String](String(py=type_obj))
        var ids = self_ptr[].store.neighbors(
            String(py=id_obj),
            _direction_from_python(direction_obj),
            edge_type,
        )
        return _string_list_to_python(ids)

    @staticmethod
    def traverse(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        id_obj: PythonObject,
        direction_obj: PythonObject,
        max_depth_obj: PythonObject,
        type_obj: PythonObject,
    ) raises -> PythonObject:
        var edge_type = Optional[String](None)
        if not (type_obj is Python.none()):
            edge_type = Optional[String](String(py=type_obj))
        var ids = self_ptr[].store.traverse(
            String(py=id_obj),
            _direction_from_python(direction_obj),
            Int(py=max_depth_obj),
            edge_type,
        )
        return _string_list_to_python(ids)

    @staticmethod
    def has_path(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        from_obj: PythonObject,
        to_obj: PythonObject,
        direction_obj: PythonObject,
        max_depth_obj: PythonObject,
        type_obj: PythonObject,
    ) raises -> PythonObject:
        var edge_type = Optional[String](None)
        if not (type_obj is Python.none()):
            edge_type = Optional[String](String(py=type_obj))
        return PythonObject(
            self_ptr[].store.has_path(
                String(py=from_obj),
                String(py=to_obj),
                _direction_from_python(direction_obj),
                Int(py=max_depth_obj),
                edge_type,
            )
        )

    @staticmethod
    def shortest_path(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        from_obj: PythonObject,
        to_obj: PythonObject,
        direction_obj: PythonObject,
        max_depth_obj: PythonObject,
        type_obj: PythonObject,
    ) raises -> PythonObject:
        var edge_type = Optional[String](None)
        if not (type_obj is Python.none()):
            edge_type = Optional[String](String(py=type_obj))
        var ids = self_ptr[].store.shortest_path(
            String(py=from_obj),
            String(py=to_obj),
            _direction_from_python(direction_obj),
            Int(py=max_depth_obj),
            edge_type,
        )
        return _string_list_to_python(ids)

    @staticmethod
    def search_exact_ids(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        vector_obj: PythonObject,
        ids_obj: PythonObject,
    ) raises -> PythonObject:
        var query = _vector_from_python(vector_obj, Self.dim)
        var results = Python.list()
        for i in range(len(ids_obj)):
            var id = String(py=ids_obj[i])
            if id not in self_ptr[].store.external_to_vector:
                continue
            var vector_id = self_ptr[].store.external_to_vector[id]
            var vector = self_ptr[].store.get(id)
            if len(vector) != Self.dim:
                raise Error("stored vector dimension mismatch")
            var distance: Float32 = 0.0
            for j in range(Self.dim):
                var delta = query[j] - vector[j]
                distance += delta * delta
            var item = Python.dict()
            item["id"] = id
            item["score"] = -distance
            item["distance"] = distance
            item["vector_id"] = Int(vector_id)
            if self_ptr[].store.metadata[Int(vector_id)]:
                item["metadata"] = (
                    self_ptr[].store.metadata[Int(vector_id)].value()
                )
            else:
                item["metadata"] = Python.none()
            if self_ptr[].store.source_spans[Int(vector_id)]:
                item["source"] = (
                    self_ptr[].store.source_spans[Int(vector_id)].value()
                )
            else:
                item["source"] = Python.none()
            results.append(item)
        # Sort by distance ascending using Python's sorted + operator.itemgetter
        var op = Python.import_module("operator")
        var getter = op.itemgetter("distance")
        var builtins = Python.import_module("builtins")
        return builtins.sorted(results, key=getter)

    @staticmethod
    def search_filtered_ids(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        vector_obj: PythonObject,
        ids_obj: PythonObject,
        k_obj: PythonObject,
        ef_obj: PythonObject,
    ) raises -> PythonObject:
        var vector = _vector_from_python(vector_obj, Self.dim)
        var total_elements = self_ptr[].store.index.num_elements()
        var allowed = List[UInt8]()
        allowed.reserve(total_elements)
        for _ in range(total_elements):
            allowed.append(0)
        for i in range(len(ids_obj)):
            var id = String(py=ids_obj[i])
            if id not in self_ptr[].store.external_to_vector:
                continue
            var vector_id = Int(self_ptr[].store.external_to_vector[id])
            if vector_id < len(allowed):
                allowed[vector_id] = 1

        var candidates = self_ptr[].store.index.search_filtered(
            Span(ptr=vector.unsafe_ptr(), length=Self.dim),
            allowed,
            k=Int(py=k_obj),
            ef=Int(py=ef_obj),
        )
        return _candidates_to_python(self_ptr, candidates)

    @staticmethod
    def search_filtered_by_bitmap(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        vector_obj: PythonObject,
        filter_dict: PythonObject,
        k_obj: PythonObject,
        ef_obj: PythonObject,
    ) raises -> PythonObject:
        """Filtered search using MetadataIndex bitmap evaluation."""
        var vector = _vector_from_python(vector_obj, Self.dim)
        var total_elements = self_ptr[].store.index.num_elements()

        # Evaluate filter via metadata index
        var bm = evaluate_filter_to_bitmap(
            filter_dict,
            self_ptr[].store.metadata_index,
            total_elements,
        )

        if not Bool(bm):
            # Can't evaluate via index — return empty list
            return Python.list()

        var allowed = bm.value().to_bytes()
        var candidates = self_ptr[].store.index.search_filtered(
            Span(ptr=vector.unsafe_ptr(), length=Self.dim),
            allowed,
            k=Int(py=k_obj),
            ef=Int(py=ef_obj),
        )
        return _candidates_to_python(self_ptr, candidates)

    @staticmethod
    def index_metadata(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        vector_id_obj: PythonObject,
        metadata_obj: PythonObject,
    ) raises -> PythonObject:
        """Index metadata dict for filtered search. Call after set()."""
        if metadata_obj is Python.none():
            return Python.none()
        self_ptr[].store.metadata_index.index_json(
            Int(py=vector_id_obj), metadata_obj
        )
        return Python.none()

    @staticmethod
    def index_metadata_batch(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        ids_obj: PythonObject,
        metadata_list_obj: PythonObject,
    ) raises -> PythonObject:
        """Index metadata for a batch of documents. Single round-trip."""
        var count = len(ids_obj)
        for i in range(count):
            var meta = metadata_list_obj[i]
            if meta is not Python.none():
                self_ptr[].store.metadata_index.index_json(
                    Int(py=ids_obj[i]), meta
                )
        return Python.none()

    @staticmethod
    def get(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        id_obj: PythonObject,
        include_vector_obj: PythonObject,
    ) raises -> PythonObject:
        var id = String(py=id_obj)
        if id not in self_ptr[].store.external_to_vector:
            return Python.none()

        var vector_id = self_ptr[].store.external_to_vector[id]
        var idx = Int(vector_id)
        # Check if item is deleted or superseded
        if idx < len(self_ptr[].store.deleted) and self_ptr[].store.deleted[idx]:
            return Python.none()

        var record = Python.dict()
        record["id"] = id
        record["vector_id"] = Int(vector_id)
        if self_ptr[].store.metadata[Int(vector_id)]:
            record["metadata"] = (
                self_ptr[].store.metadata[Int(vector_id)].value()
            )
        else:
            record["metadata"] = Python.none()
        if self_ptr[].store.source_spans[Int(vector_id)]:
            record["source"] = (
                self_ptr[].store.source_spans[Int(vector_id)].value()
            )
        else:
            record["source"] = Python.none()
        # Timestamps
        var ts_idx = Int(vector_id)
        if ts_idx < len(self_ptr[].store.created_at):
            record["created_at"] = self_ptr[].store.created_at[ts_idx]
            record["updated_at"] = self_ptr[].store.updated_at[ts_idx]
            if self_ptr[].store.superseded_at[ts_idx] > 0.0:
                record["superseded_at"] = self_ptr[].store.superseded_at[ts_idx]
        if Bool(include_vector_obj):
            var vector = self_ptr[].store.get(id)
            var py_vector = Python.list()
            for i in range(len(vector)):
                py_vector.append(vector[i])
            record["vector"] = py_vector
        return record

    @staticmethod
    def get_text(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        id_obj: PythonObject,
    ) raises -> PythonObject:
        var id = String(py=id_obj)
        if id not in self_ptr[].store.external_to_vector:
            return Python.none()

        var vector_id = self_ptr[].store.external_to_vector[id]
        for i in range(len(self_ptr[].store.text_doc_vector_ids)):
            if self_ptr[].store.text_doc_vector_ids[i] == vector_id:
                return PythonObject(self_ptr[].store.text_doc_texts[i])
        return Python.none()

    @staticmethod
    def records(
        self_ptr: UnsafePointer[Self, MutAnyOrigin]
    ) raises -> PythonObject:
        var records = Python.list()
        for i in range(len(self_ptr[].store.vector_external_ids)):
            if self_ptr[].store.deleted[i]:
                continue
            var item = Python.dict()
            item["id"] = self_ptr[].store.vector_external_ids[i]
            item["vector_id"] = i
            if self_ptr[].store.metadata[i]:
                item["metadata"] = self_ptr[].store.metadata[i].value()
            else:
                item["metadata"] = Python.none()
            if self_ptr[].store.source_spans[i]:
                item["source"] = self_ptr[].store.source_spans[i].value()
            else:
                item["source"] = Python.none()
            records.append(item)
        return records

    @staticmethod
    def live_ids(
        self_ptr: UnsafePointer[Self, MutAnyOrigin]
    ) raises -> PythonObject:
        var ids = Python.list()
        for i in range(len(self_ptr[].store.vector_external_ids)):
            if self_ptr[].store.deleted[i]:
                continue
            ids.append(self_ptr[].store.vector_external_ids[i])
        return ids

    @staticmethod
    def batch_metadata(
        self_ptr: UnsafePointer[Self, MutAnyOrigin]
    ) raises -> PythonObject:
        """Return (ids_list, metadata_list, deleted_list) for all items."""
        var ids = Python.list()
        var metas = Python.list()
        var dels = Python.list()
        for i in range(len(self_ptr[].store.vector_external_ids)):
            ids.append(self_ptr[].store.vector_external_ids[i])
            if self_ptr[].store.metadata[i]:
                metas.append(PythonObject(self_ptr[].store.metadata[i].value()))
            else:
                metas.append(Python.none())
            dels.append(self_ptr[].store.deleted[i])
        return Python.tuple(ids, metas, dels)

    @staticmethod
    def filter_matching_ids(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        filter_obj: PythonObject,
    ) raises -> PythonObject:
        """Evaluate filter natively, return matching IDs."""
        # Build all Python lists in one pass
        var py_ids = Python.list()
        var py_metas = Python.list()
        var py_dels = Python.list()
        var py_none = Python.none()
        for i in range(len(self_ptr[].store.vector_external_ids)):
            py_ids.append(self_ptr[].store.vector_external_ids[i])
            if self_ptr[].store.metadata[i]:
                py_metas.append(PythonObject(self_ptr[].store.metadata[i].value()))
            else:
                py_metas.append(py_none)
            py_dels.append(self_ptr[].store.deleted[i])
        return filter_ids_batch(
            py_ids,
            py_metas,
            py_dels,
            filter_obj,
        )

    @staticmethod
    def metadata_for_id(
        self_ptr: UnsafePointer[Self, MutAnyOrigin], id_obj: PythonObject
    ) raises -> PythonObject:
        var id = String(py=id_obj)
        if id not in self_ptr[].store.external_to_vector:
            return Python.none()
        var doc_id = self_ptr[].store.external_to_vector[id]
        var idx = Int(doc_id)
        if idx < len(self_ptr[].store.deleted) and self_ptr[].store.deleted[idx]:
            return Python.none()
        if self_ptr[].store.metadata[idx]:
            return PythonObject(self_ptr[].store.metadata[idx].value())
        return Python.none()

    @staticmethod
    def delete(
        self_ptr: UnsafePointer[Self, MutAnyOrigin], id_obj: PythonObject
    ) raises -> PythonObject:
        return PythonObject(self_ptr[].store.delete(String(py=id_obj)))

    @staticmethod
    def supersede(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        old_id_obj: PythonObject,
        new_id_obj: PythonObject,
    ) raises -> PythonObject:
        return PythonObject(self_ptr[].store.supersede(
            String(py=old_id_obj), String(py=new_id_obj)
        ))

    @staticmethod
    def search_text(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        query_obj: PythonObject,
        k_obj: PythonObject,
    ) raises -> PythonObject:
        var results = self_ptr[].store.search_text(
            String(py=query_obj), k=Int(py=k_obj)
        )
        return _search_results_to_python(results)

    @staticmethod
    def search_hybrid(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        query_obj: PythonObject,
        text_obj: PythonObject,
        opts_obj: PythonObject,
    ) raises -> PythonObject:
        """search_hybrid(query_vector, text_query, opts_dict)"""
        var query = _vector_from_python(query_obj, Self.dim)
        var text_query = String(py=text_obj)
        var k = 10
        var alpha: Float32 = 0.5
        var rrf_k = 60
        var ef = 200
        if opts_obj is not Python.none():
            if "k" in opts_obj:
                k = Int(py=opts_obj["k"])
            if "alpha" in opts_obj:
                alpha = Float32(py=opts_obj["alpha"])
            if "rrf_k" in opts_obj:
                rrf_k = Int(py=opts_obj["rrf_k"])
            if "ef_search" in opts_obj:
                ef = Int(py=opts_obj["ef_search"])
        var results = self_ptr[].store.search_hybrid(
            Span(ptr=query.unsafe_ptr(), length=Self.dim),
            text_query,
            k=k,
            alpha=alpha,
            rrf_k=rrf_k,
            ef_search=ef,
        )
        return _search_results_to_python(results)

    @staticmethod
    def set_sparse(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        id_obj: PythonObject,
        sparse_obj: PythonObject,
    ) raises -> PythonObject:
        if sparse_obj is Python.none():
            return Python.none()
        var sparse = _sparse_from_python(sparse_obj)
        self_ptr[].store.set_sparse(String(py=id_obj), sparse)
        return Python.none()

    @staticmethod
    def needs_compaction(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
    ) raises -> PythonObject:
        return PythonObject(self_ptr[].store.needs_compaction())

    @staticmethod
    def len(self_ptr: UnsafePointer[Self, MutAnyOrigin]) raises -> PythonObject:
        return PythonObject(self_ptr[].store.len())

    @staticmethod
    def flush(
        self_ptr: UnsafePointer[Self, MutAnyOrigin]
    ) raises -> PythonObject:
        self_ptr[].store.flush()
        return Python.none()


struct MultiVectorCollection[dim: Int](Movable, Writable):
    var store: MultiVectorExactStore[Self.dim]

    def __init__(out self) raises:
        self = Self.memory()

    @staticmethod
    def memory() raises -> Self:
        return Self(MultiVectorExactStore[Self.dim]())

    @staticmethod
    def create(path: String) raises -> Self:
        return Self(MultiVectorExactStore[Self.dim].create(path))

    @staticmethod
    def open(path: String) raises -> Self:
        return Self(MultiVectorExactStore[Self.dim].open(path))

    def __init__(out self, var store: MultiVectorExactStore[Self.dim]):
        self.store = store^

    def write_to(self, mut writer: Some[Writer]):
        writer.write("MultiVectorCollection(len=", self.store.len(), ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    @staticmethod
    def py_init(
        out self: Self, args: PythonObject, kwargs: PythonObject
    ) raises:
        if len(args) != 0:
            raise Error("MultiVectorCollection() takes no positional arguments")
        self = Self()

    @staticmethod
    def set(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        id_obj: PythonObject,
        vectors_obj: PythonObject,
        metadata_obj: PythonObject,
        source_obj: PythonObject,
    ) raises -> PythonObject:
        var vector_count = len(vectors_obj)
        var vectors = _vectors_from_python(vectors_obj, Self.dim)
        var metadata = _optional_string_from_python(metadata_obj)
        var source_span = _optional_string_from_python(source_obj)
        _ = self_ptr[].store.set_vectors(
            String(py=id_obj),
            Span(ptr=vectors.unsafe_ptr(), length=len(vectors)),
            vector_count,
            metadata=metadata,
            source_span=source_span,
        )
        return Python.none()

    @staticmethod
    def set_text(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        id_obj: PythonObject,
        vectors_obj: PythonObject,
        text_obj: PythonObject,
        metadata_obj: PythonObject,
        source_obj: PythonObject,
    ) raises -> PythonObject:
        var vector_count = len(vectors_obj)
        var vectors = _vectors_from_python(vectors_obj, Self.dim)
        var metadata = _optional_string_from_python(metadata_obj)
        var source_span = _optional_string_from_python(source_obj)
        _ = self_ptr[].store.set_vectors_text(
            String(py=id_obj),
            Span(ptr=vectors.unsafe_ptr(), length=len(vectors)),
            vector_count,
            String(py=text_obj),
            metadata=metadata,
            source_span=source_span,
        )
        return Python.none()

    @staticmethod
    def search(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        vectors_obj: PythonObject,
        k_obj: PythonObject,
    ) raises -> PythonObject:
        var vector_count = len(vectors_obj)
        var vectors = _vectors_from_python(vectors_obj, Self.dim)
        var results = self_ptr[].store.search_vectors_with_options(
            Span(ptr=vectors.unsafe_ptr(), length=len(vectors)),
            vector_count,
            k=Int(py=k_obj),
        )
        return _multivector_results_to_python(results)

    @staticmethod
    def search_text(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        query_obj: PythonObject,
        vectors_obj: PythonObject,
        k_obj: PythonObject,
    ) raises -> PythonObject:
        var vector_count = len(vectors_obj)
        var vectors = _vectors_from_python(vectors_obj, Self.dim)
        var results = self_ptr[].store.search_hybrid_vectors(
            String(py=query_obj),
            Span(ptr=vectors.unsafe_ptr(), length=len(vectors)),
            vector_count,
            k=Int(py=k_obj),
        )
        return _multivector_results_to_python(results)

    @staticmethod
    def get(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        id_obj: PythonObject,
        include_vectors_obj: PythonObject,
    ) raises -> PythonObject:
        var id = String(py=id_obj)
        var metadata = self_ptr[].store.get_metadata(id)
        if not metadata and self_ptr[].store.get_vector_count(id) == 0:
            return Python.none()

        var record = Python.dict()
        record["id"] = id
        if metadata:
            record["metadata"] = metadata.value()
        else:
            record["metadata"] = Python.none()
        var source_span = self_ptr[].store.get_source_span(id)
        if source_span:
            record["source"] = source_span.value()
        else:
            record["source"] = Python.none()
        if Bool(include_vectors_obj):
            var flat = self_ptr[].store.get_vectors(id)
            var vector_count = self_ptr[].store.get_vector_count(id)
            record["vectors"] = _nested_vectors_to_python(
                flat, vector_count, Self.dim
            )
        return record

    @staticmethod
    def get_text(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        id_obj: PythonObject,
    ) raises -> PythonObject:
        var id = String(py=id_obj)
        if id not in self_ptr[].store.external_to_doc:
            return Python.none()

        var doc_id = self_ptr[].store.external_to_doc[id]
        for i in range(len(self_ptr[].store.text_doc_ids)):
            if self_ptr[].store.text_doc_ids[i] == doc_id:
                return PythonObject(self_ptr[].store.text_doc_texts[i])
        return Python.none()

    @staticmethod
    def records(
        self_ptr: UnsafePointer[Self, MutAnyOrigin]
    ) raises -> PythonObject:
        var records = Python.list()
        for i in range(len(self_ptr[].store.external_ids)):
            if self_ptr[].store.deleted[i]:
                continue
            var item = Python.dict()
            item["id"] = self_ptr[].store.external_ids[i]
            item["doc_id"] = i
            if self_ptr[].store.metadata[i]:
                item["metadata"] = self_ptr[].store.metadata[i].value()
            else:
                item["metadata"] = Python.none()
            if self_ptr[].store.source_spans[i]:
                item["source"] = self_ptr[].store.source_spans[i].value()
            else:
                item["source"] = Python.none()
            records.append(item)
        return records

    @staticmethod
    def live_ids(
        self_ptr: UnsafePointer[Self, MutAnyOrigin]
    ) raises -> PythonObject:
        var ids = Python.list()
        for i in range(len(self_ptr[].store.external_ids)):
            if self_ptr[].store.deleted[i]:
                continue
            ids.append(self_ptr[].store.external_ids[i])
        return ids

    @staticmethod
    def batch_metadata(
        self_ptr: UnsafePointer[Self, MutAnyOrigin]
    ) raises -> PythonObject:
        """Return (ids_list, metadata_list, deleted_list) for all items."""
        var ids = Python.list()
        var metas = Python.list()
        var dels = Python.list()
        for i in range(len(self_ptr[].store.external_ids)):
            ids.append(self_ptr[].store.external_ids[i])
            if self_ptr[].store.metadata[i]:
                metas.append(PythonObject(self_ptr[].store.metadata[i].value()))
            else:
                metas.append(Python.none())
            dels.append(self_ptr[].store.deleted[i])
        return Python.tuple(ids, metas, dels)

    @staticmethod
    def filter_matching_ids(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        filter_obj: PythonObject,
    ) raises -> PythonObject:
        """Evaluate filter natively, return matching IDs."""
        var py_ids = Python.list()
        var py_metas = Python.list()
        var py_dels = Python.list()
        var py_none = Python.none()
        for i in range(len(self_ptr[].store.external_ids)):
            py_ids.append(self_ptr[].store.external_ids[i])
            if self_ptr[].store.metadata[i]:
                py_metas.append(PythonObject(self_ptr[].store.metadata[i].value()))
            else:
                py_metas.append(py_none)
            py_dels.append(self_ptr[].store.deleted[i])
        return filter_ids_batch(
            py_ids,
            py_metas,
            py_dels,
            filter_obj,
        )

    @staticmethod
    def metadata_for_id(
        self_ptr: UnsafePointer[Self, MutAnyOrigin], id_obj: PythonObject
    ) raises -> PythonObject:
        var id = String(py=id_obj)
        if id not in self_ptr[].store.external_to_doc:
            return Python.none()
        var doc_id = self_ptr[].store.external_to_doc[id]
        var idx = Int(doc_id)
        if idx < len(self_ptr[].store.deleted) and self_ptr[].store.deleted[idx]:
            return Python.none()
        if self_ptr[].store.metadata[idx]:
            return PythonObject(self_ptr[].store.metadata[idx].value())
        return Python.none()

    @staticmethod
    def delete(
        self_ptr: UnsafePointer[Self, MutAnyOrigin], id_obj: PythonObject
    ) raises -> PythonObject:
        return PythonObject(self_ptr[].store.delete(String(py=id_obj)))

    @staticmethod
    def supersede(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        old_id_obj: PythonObject,
        new_id_obj: PythonObject,
    ) raises -> PythonObject:
        return PythonObject(self_ptr[].store.supersede(
            String(py=old_id_obj), String(py=new_id_obj)
        ))

    @staticmethod
    def len(self_ptr: UnsafePointer[Self, MutAnyOrigin]) raises -> PythonObject:
        return PythonObject(self_ptr[].store.len())

    @staticmethod
    def flush(
        self_ptr: UnsafePointer[Self, MutAnyOrigin]
    ) raises -> PythonObject:
        self_ptr[].store.flush()
        return Python.none()


struct MultiVectorMuveraCollection[dim: Int](Movable, Writable):
    var store: MultiVectorMuveraStore[Self.dim]

    def __init__(out self) raises:
        self = Self.memory()

    @staticmethod
    def memory() raises -> Self:
        return Self(MultiVectorMuveraStore[Self.dim]())

    @staticmethod
    def create(path: String) raises -> Self:
        return Self(MultiVectorMuveraStore[Self.dim].create(path))

    @staticmethod
    def open(path: String) raises -> Self:
        return Self(MultiVectorMuveraStore[Self.dim].open(path))

    def __init__(out self, var store: MultiVectorMuveraStore[Self.dim]):
        self.store = store^

    def write_to(self, mut writer: Some[Writer]):
        writer.write("MultiVectorMuveraCollection(len=", self.store.len(), ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    @staticmethod
    def py_init(
        out self: Self, args: PythonObject, kwargs: PythonObject
    ) raises:
        if len(args) != 0:
            raise Error(
                "MultiVectorMuveraCollection() takes no positional arguments"
            )
        self = Self()

    @staticmethod
    def set(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        id_obj: PythonObject,
        vectors_obj: PythonObject,
        metadata_obj: PythonObject,
        source_obj: PythonObject,
    ) raises -> PythonObject:
        var vector_count = len(vectors_obj)
        var vectors = _vectors_from_python(vectors_obj, Self.dim)
        var metadata = _optional_string_from_python(metadata_obj)
        var source_span = _optional_string_from_python(source_obj)
        _ = self_ptr[].store.set_vectors(
            String(py=id_obj),
            Span(ptr=vectors.unsafe_ptr(), length=len(vectors)),
            vector_count,
            metadata=metadata,
            source_span=source_span,
        )
        return Python.none()

    @staticmethod
    def set_text(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        id_obj: PythonObject,
        vectors_obj: PythonObject,
        text_obj: PythonObject,
        metadata_obj: PythonObject,
        source_obj: PythonObject,
    ) raises -> PythonObject:
        var vector_count = len(vectors_obj)
        var vectors = _vectors_from_python(vectors_obj, Self.dim)
        var metadata = _optional_string_from_python(metadata_obj)
        var source_span = _optional_string_from_python(source_obj)
        _ = self_ptr[].store.set_vectors_text(
            String(py=id_obj),
            Span(ptr=vectors.unsafe_ptr(), length=len(vectors)),
            vector_count,
            String(py=text_obj),
            metadata=metadata,
            source_span=source_span,
        )
        return Python.none()

    @staticmethod
    def search(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        vectors_obj: PythonObject,
        k_obj: PythonObject,
    ) raises -> PythonObject:
        var vector_count = len(vectors_obj)
        var vectors = _vectors_from_python(vectors_obj, Self.dim)
        var results = self_ptr[].store.search_vectors_with_options(
            Span(ptr=vectors.unsafe_ptr(), length=len(vectors)),
            vector_count,
            k=Int(py=k_obj),
        )
        return _multivector_results_to_python(results)

    @staticmethod
    def search_text(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        query_obj: PythonObject,
        vectors_obj: PythonObject,
        k_obj: PythonObject,
    ) raises -> PythonObject:
        var vector_count = len(vectors_obj)
        var vectors = _vectors_from_python(vectors_obj, Self.dim)
        var results = self_ptr[].store.search_hybrid_vectors(
            String(py=query_obj),
            Span(ptr=vectors.unsafe_ptr(), length=len(vectors)),
            vector_count,
            k=Int(py=k_obj),
        )
        return _multivector_results_to_python(results)

    @staticmethod
    def get(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        id_obj: PythonObject,
        include_vectors_obj: PythonObject,
    ) raises -> PythonObject:
        var id = String(py=id_obj)
        var metadata = self_ptr[].store.get_metadata(id)
        if not metadata and self_ptr[].store.get_vector_count(id) == 0:
            return Python.none()

        var record = Python.dict()
        record["id"] = id
        if metadata:
            record["metadata"] = metadata.value()
        else:
            record["metadata"] = Python.none()
        var source_span = self_ptr[].store.get_source_span(id)
        if source_span:
            record["source"] = source_span.value()
        else:
            record["source"] = Python.none()
        if Bool(include_vectors_obj):
            var flat = self_ptr[].store.get_vectors(id)
            var vector_count = self_ptr[].store.get_vector_count(id)
            record["vectors"] = _nested_vectors_to_python(
                flat, vector_count, Self.dim
            )
        return record

    @staticmethod
    def get_text(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        id_obj: PythonObject,
    ) raises -> PythonObject:
        var id = String(py=id_obj)
        if id not in self_ptr[].store.exact.external_to_doc:
            return Python.none()

        var doc_id = self_ptr[].store.exact.external_to_doc[id]
        for i in range(len(self_ptr[].store.exact.text_doc_ids)):
            if self_ptr[].store.exact.text_doc_ids[i] == doc_id:
                return PythonObject(self_ptr[].store.exact.text_doc_texts[i])
        return Python.none()

    @staticmethod
    def records(
        self_ptr: UnsafePointer[Self, MutAnyOrigin]
    ) raises -> PythonObject:
        var records = Python.list()
        for i in range(len(self_ptr[].store.exact.external_ids)):
            if self_ptr[].store.exact.deleted[i]:
                continue
            var item = Python.dict()
            item["id"] = self_ptr[].store.exact.external_ids[i]
            item["doc_id"] = i
            if self_ptr[].store.exact.metadata[i]:
                item["metadata"] = self_ptr[].store.exact.metadata[i].value()
            else:
                item["metadata"] = Python.none()
            if self_ptr[].store.exact.source_spans[i]:
                item["source"] = self_ptr[].store.exact.source_spans[i].value()
            else:
                item["source"] = Python.none()
            records.append(item)
        return records

    @staticmethod
    def live_ids(
        self_ptr: UnsafePointer[Self, MutAnyOrigin]
    ) raises -> PythonObject:
        var ids = Python.list()
        for i in range(len(self_ptr[].store.exact.external_ids)):
            if self_ptr[].store.exact.deleted[i]:
                continue
            ids.append(self_ptr[].store.exact.external_ids[i])
        return ids

    @staticmethod
    def batch_metadata(
        self_ptr: UnsafePointer[Self, MutAnyOrigin]
    ) raises -> PythonObject:
        """Return (ids_list, metadata_list, deleted_list) for all items."""
        var ids = Python.list()
        var metas = Python.list()
        var dels = Python.list()
        for i in range(len(self_ptr[].store.exact.external_ids)):
            ids.append(self_ptr[].store.exact.external_ids[i])
            if self_ptr[].store.exact.metadata[i]:
                metas.append(PythonObject(self_ptr[].store.exact.metadata[i].value()))
            else:
                metas.append(Python.none())
            dels.append(self_ptr[].store.exact.deleted[i])
        return Python.tuple(ids, metas, dels)

    @staticmethod
    def filter_matching_ids(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        filter_obj: PythonObject,
    ) raises -> PythonObject:
        """Evaluate filter natively, return matching IDs."""
        var py_ids = Python.list()
        var py_metas = Python.list()
        var py_dels = Python.list()
        var py_none = Python.none()
        for i in range(len(self_ptr[].store.exact.external_ids)):
            py_ids.append(self_ptr[].store.exact.external_ids[i])
            if self_ptr[].store.exact.metadata[i]:
                py_metas.append(PythonObject(self_ptr[].store.exact.metadata[i].value()))
            else:
                py_metas.append(py_none)
            py_dels.append(self_ptr[].store.exact.deleted[i])
        return filter_ids_batch(
            py_ids,
            py_metas,
            py_dels,
            filter_obj,
        )

    @staticmethod
    def metadata_for_id(
        self_ptr: UnsafePointer[Self, MutAnyOrigin], id_obj: PythonObject
    ) raises -> PythonObject:
        var id = String(py=id_obj)
        if id not in self_ptr[].store.exact.external_to_doc:
            return Python.none()
        var doc_id = self_ptr[].store.exact.external_to_doc[id]
        var idx = Int(doc_id)
        if idx < len(self_ptr[].store.exact.deleted) and self_ptr[].store.exact.deleted[idx]:
            return Python.none()
        if self_ptr[].store.exact.metadata[idx]:
            return PythonObject(self_ptr[].store.exact.metadata[idx].value())
        return Python.none()

    @staticmethod
    def delete(
        self_ptr: UnsafePointer[Self, MutAnyOrigin], id_obj: PythonObject
    ) raises -> PythonObject:
        return PythonObject(self_ptr[].store.delete(String(py=id_obj)))

    @staticmethod
    def supersede(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        old_id_obj: PythonObject,
        new_id_obj: PythonObject,
    ) raises -> PythonObject:
        return PythonObject(self_ptr[].store.supersede(
            String(py=old_id_obj), String(py=new_id_obj)
        ))

    @staticmethod
    def len(self_ptr: UnsafePointer[Self, MutAnyOrigin]) raises -> PythonObject:
        return PythonObject(self_ptr[].store.len())

    @staticmethod
    def flush(
        self_ptr: UnsafePointer[Self, MutAnyOrigin]
    ) raises -> PythonObject:
        self_ptr[].store.flush()
        return Python.none()


struct DynamicFlatCollection(Movable, Writable):
    """A collection using dynamic flat index for arbitrary dimensions."""

    var store: DynamicFlatVectorStore

    def __init__(out self, dim: Int) raises:
        self = Self.memory(dim)

    @staticmethod
    def memory(dim: Int) raises -> Self:
        return Self(DynamicFlatVectorStore(dim))

    @staticmethod
    def create(path: String, dim: Int) raises -> Self:
        return Self(DynamicFlatVectorStore.create(path, dim))

    @staticmethod
    def open(path: String) raises -> Self:
        return Self(DynamicFlatVectorStore.open(path))

    def __init__(out self, var store: DynamicFlatVectorStore):
        self.store = store^

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "DynamicFlatCollection(dim=",
            self.store.dim,
            ", len=",
            self.store.len(),
            ")",
        )

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    @staticmethod
    def py_init(
        out self: Self, args: PythonObject, kwargs: PythonObject
    ) raises:
        if len(args) != 1:
            raise Error(
                "DynamicFlatCollection(dim) requires one positional argument"
            )
        var dim = Int(py=args[0])
        self = Self(dim)

    @staticmethod
    def set(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        id_obj: PythonObject,
        vector_obj: PythonObject,
        metadata_obj: PythonObject,
        source_obj: PythonObject,
    ) raises -> PythonObject:
        var dim = self_ptr[].store.dim
        var vector = _vector_from_python(vector_obj, dim)
        var metadata = _optional_string_from_python(metadata_obj)
        var source_span = _optional_string_from_python(source_obj)
        _ = self_ptr[].store.set(
            String(py=id_obj),
            Span(ptr=vector.unsafe_ptr(), length=dim),
            metadata=metadata,
            source_span=source_span,
        )
        return Python.none()

    @staticmethod
    def search(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        vector_obj: PythonObject,
        k_obj: PythonObject,
        ef_obj: PythonObject,
    ) raises -> PythonObject:
        var dim = self_ptr[].store.dim
        var vector = _vector_from_python(vector_obj, dim)
        var results = self_ptr[].store.search(
            Span(ptr=vector.unsafe_ptr(), length=dim),
            Int(py=k_obj),
        )
        return _dynamic_flat_results_to_python(results)

    @staticmethod
    def search_exact_ids(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        vector_obj: PythonObject,
        ids_obj: PythonObject,
    ) raises -> PythonObject:
        var dim = self_ptr[].store.dim
        var query = _vector_from_python(vector_obj, dim)
        var ids = List[String]()
        for i in range(len(ids_obj)):
            ids.append(String(py=ids_obj[i]))
        var results = self_ptr[].store.search_exact_ids(
            Span(ptr=query.unsafe_ptr(), length=dim),
            ids,
        )
        return _dynamic_flat_results_to_python(results)

    @staticmethod
    def search_filtered_ids(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        vector_obj: PythonObject,
        ids_obj: PythonObject,
        k_obj: PythonObject,
        ef_obj: PythonObject,
    ) raises -> PythonObject:
        var dim = self_ptr[].store.dim
        var vector = _vector_from_python(vector_obj, dim)
        var ids = List[String]()
        for i in range(len(ids_obj)):
            ids.append(String(py=ids_obj[i]))
        var results = self_ptr[].store.search_exact_ids(
            Span(ptr=vector.unsafe_ptr(), length=dim),
            ids,
        )
        # Limit to k results
        var limited = List[DynamicFlatSearchResult]()
        var k = Int(py=k_obj)
        for i in range(min(k, len(results))):
            limited.append(results[i].copy())
        return _dynamic_flat_results_to_python(limited)

    @staticmethod
    def get(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        id_obj: PythonObject,
        include_vector_obj: PythonObject,
    ) raises -> PythonObject:
        var id = String(py=id_obj)
        if not (id in self_ptr[].store.external_to_vector):
            return Python.none()

        var record = Python.dict()
        record["id"] = id
        var vector_id = Int(self_ptr[].store.external_to_vector[id])
        if (
            vector_id < len(self_ptr[].store.metadata)
            and self_ptr[].store.metadata[vector_id]
        ):
            record["metadata"] = self_ptr[].store.metadata[vector_id].value()
        else:
            record["metadata"] = Python.none()
        if (
            vector_id < len(self_ptr[].store.source_spans)
            and self_ptr[].store.source_spans[vector_id]
        ):
            record["source"] = self_ptr[].store.source_spans[vector_id].value()
        else:
            record["source"] = Python.none()
        if Bool(include_vector_obj):
            var vector = self_ptr[].store.get(id)
            var py_vector = Python.list()
            for i in range(len(vector)):
                py_vector.append(vector[i])
            record["vector"] = py_vector
        return record

    @staticmethod
    def get_text(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        id_obj: PythonObject,
    ) raises -> PythonObject:
        # Text not supported in dynamic flat store
        return Python.none()

    @staticmethod
    def records(
        self_ptr: UnsafePointer[Self, MutAnyOrigin]
    ) raises -> PythonObject:
        var records = Python.list()
        for i in range(len(self_ptr[].store.vector_external_ids)):
            if self_ptr[].store.deleted[i]:
                continue
            var item = Python.dict()
            item["id"] = self_ptr[].store.vector_external_ids[i]
            item["vector_id"] = i
            if self_ptr[].store.metadata[i]:
                item["metadata"] = self_ptr[].store.metadata[i].value()
            else:
                item["metadata"] = Python.none()
            if self_ptr[].store.source_spans[i]:
                item["source"] = self_ptr[].store.source_spans[i].value()
            else:
                item["source"] = Python.none()
            records.append(item)
        return records

    @staticmethod
    def live_ids(
        self_ptr: UnsafePointer[Self, MutAnyOrigin]
    ) raises -> PythonObject:
        var ids = Python.list()
        for i in range(len(self_ptr[].store.vector_external_ids)):
            if self_ptr[].store.deleted[i]:
                continue
            ids.append(self_ptr[].store.vector_external_ids[i])
        return ids

    @staticmethod
    def batch_metadata(
        self_ptr: UnsafePointer[Self, MutAnyOrigin]
    ) raises -> PythonObject:
        """Return (ids_list, metadata_list, deleted_list) for all items."""
        var ids = Python.list()
        var metas = Python.list()
        var dels = Python.list()
        for i in range(len(self_ptr[].store.vector_external_ids)):
            ids.append(self_ptr[].store.vector_external_ids[i])
            if self_ptr[].store.metadata[i]:
                metas.append(PythonObject(self_ptr[].store.metadata[i].value()))
            else:
                metas.append(Python.none())
            dels.append(self_ptr[].store.deleted[i])
        return Python.tuple(ids, metas, dels)

    @staticmethod
    def filter_matching_ids(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        filter_obj: PythonObject,
    ) raises -> PythonObject:
        """Evaluate filter natively, return matching IDs."""
        # Build all Python lists in one pass
        var py_ids = Python.list()
        var py_metas = Python.list()
        var py_dels = Python.list()
        var py_none = Python.none()
        for i in range(len(self_ptr[].store.vector_external_ids)):
            py_ids.append(self_ptr[].store.vector_external_ids[i])
            if self_ptr[].store.metadata[i]:
                py_metas.append(PythonObject(self_ptr[].store.metadata[i].value()))
            else:
                py_metas.append(py_none)
            py_dels.append(self_ptr[].store.deleted[i])
        return filter_ids_batch(
            py_ids,
            py_metas,
            py_dels,
            filter_obj,
        )

    @staticmethod
    def metadata_for_id(
        self_ptr: UnsafePointer[Self, MutAnyOrigin], id_obj: PythonObject
    ) raises -> PythonObject:
        var id = String(py=id_obj)
        if id not in self_ptr[].store.external_to_vector:
            return Python.none()
        var doc_id = self_ptr[].store.external_to_vector[id]
        var idx = Int(doc_id)
        if idx < len(self_ptr[].store.deleted) and self_ptr[].store.deleted[idx]:
            return Python.none()
        if self_ptr[].store.metadata[idx]:
            return PythonObject(self_ptr[].store.metadata[idx].value())
        return Python.none()

    @staticmethod
    def delete(
        self_ptr: UnsafePointer[Self, MutAnyOrigin], id_obj: PythonObject
    ) raises -> PythonObject:
        return PythonObject(self_ptr[].store.delete(String(py=id_obj)))

    @staticmethod
    def supersede(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        old_id_obj: PythonObject,
        new_id_obj: PythonObject,
    ) raises -> PythonObject:
        return PythonObject(self_ptr[].store.supersede(
            String(py=old_id_obj), String(py=new_id_obj)
        ))

    @staticmethod
    def search_text(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        query_obj: PythonObject,
        k_obj: PythonObject,
    ) raises -> PythonObject:
        # Text search not supported in dynamic flat store
        raise Error("text search not supported in dynamic flat store")

    @staticmethod
    def len(self_ptr: UnsafePointer[Self, MutAnyOrigin]) raises -> PythonObject:
        return PythonObject(self_ptr[].store.len())

    @staticmethod
    def flush(
        self_ptr: UnsafePointer[Self, MutAnyOrigin]
    ) raises -> PythonObject:
        self_ptr[].store.flush()
        return Python.none()


def _vector_from_python(
    vector_obj: PythonObject, dim: Int
) raises -> List[Float32]:
    if len(vector_obj) != dim:
        raise Error("vector dimension mismatch")
    var vector = List[Float32]()
    vector.reserve(dim)
    for i in range(dim):
        vector.append(Float32(py=vector_obj[i]))
    return vector^


def _vectors_from_python(
    vectors_obj: PythonObject, dim: Int
) raises -> List[Float32]:
    var vector_count = len(vectors_obj)
    if vector_count <= 0:
        raise Error("vectors must contain at least one vector")
    var flat = List[Float32]()
    flat.reserve(vector_count * dim)
    for i in range(vector_count):
        if len(vectors_obj[i]) != dim:
            raise Error("vector dimension mismatch")
        for j in range(dim):
            flat.append(Float32(py=vectors_obj[i][j]))
    return flat^


def _optional_string_from_python(
    value_obj: PythonObject,
) raises -> Optional[String]:
    if value_obj is Python.none():
        return Optional[String](None)
    return Optional[String](String(py=value_obj))


def _direction_from_python(
    direction_obj: PythonObject,
) raises -> GraphDirection:
    var direction = String(py=direction_obj)
    if direction == "out":
        return GraphDirection.OUTGOING
    if direction == "in":
        return GraphDirection.INCOMING
    if direction == "both":
        return GraphDirection.BOTH
    raise Error("direction must be 'out', 'in', or 'both'")


def _string_list_to_python(ids: List[String]) raises -> PythonObject:
    var result = Python.list()
    for i in range(len(ids)):
        result.append(ids[i])
    return result


def _search_results_to_python(
    results: List[SearchResult],
) raises -> PythonObject:
    var py_results = Python.list()
    for i in range(len(results)):
        var item = Python.dict()
        item["id"] = results[i].id
        # Use explicit score if set (hybrid search), else -distance (vector/text)
        if results[i].has_rrf_score:
            item["score"] = results[i].rrf_score
        else:
            item["score"] = -results[i].distance
        item["distance"] = results[i].distance
        item["vector_id"] = Int(results[i].vector_id)
        if results[i].metadata:
            item["metadata"] = results[i].metadata.value()
        else:
            item["metadata"] = Python.none()
        if results[i].source_span:
            item["source"] = results[i].source_span.value()
        else:
            item["source"] = Python.none()
        py_results.append(item)
    return py_results


def _dynamic_flat_results_to_python(
    results: List[DynamicFlatSearchResult],
) raises -> PythonObject:
    var py_results = Python.list()
    for i in range(len(results)):
        var item = Python.dict()
        item["id"] = results[i].id
        item["score"] = -results[i].distance
        item["distance"] = results[i].distance
        item["vector_id"] = Int(results[i].vector_id)
        if results[i].metadata:
            item["metadata"] = results[i].metadata.value()
        else:
            item["metadata"] = Python.none()
        if results[i].source_span:
            item["source"] = results[i].source_span.value()
        else:
            item["source"] = Python.none()
        py_results.append(item)
    return py_results


def _multivector_results_to_python(
    results: List[MultiVectorResult],
) raises -> PythonObject:
    var py_results = Python.list()
    for i in range(len(results)):
        var maxsim = -results[i].distance
        var item = Python.dict()
        item["id"] = results[i].id
        item["score"] = maxsim
        item["distance"] = Python.none()
        item["maxsim"] = maxsim
        item["doc_id"] = Int(results[i].doc_id)
        if results[i].metadata:
            item["metadata"] = results[i].metadata.value()
        else:
            item["metadata"] = Python.none()
        if results[i].source_span:
            item["source"] = results[i].source_span.value()
        else:
            item["source"] = Python.none()
        py_results.append(item)
    return py_results


def _nested_vectors_to_python(
    flat: List[Float32], vector_count: Int, dim: Int
) raises -> PythonObject:
    var py_vectors = Python.list()
    for i in range(vector_count):
        var py_vector = Python.list()
        for j in range(dim):
            py_vector.append(flat[i * dim + j])
        py_vectors.append(py_vector)
    return py_vectors


def _candidates_to_python[
    dim: Int
](
    self_ptr: UnsafePointer[DenseCollection[dim], MutAnyOrigin],
    candidates: List[Candidate],
) raises -> PythonObject:
    var py_results = Python.list()
    for i in range(len(candidates)):
        var vector_id = UInt64(candidates[i].id)
        var item = Python.dict()
        item["id"] = self_ptr[].store.vector_external_ids[Int(vector_id)]
        item["score"] = -candidates[i].distance
        item["distance"] = candidates[i].distance
        item["vector_id"] = Int(vector_id)
        if self_ptr[].store.metadata[Int(vector_id)]:
            item["metadata"] = self_ptr[].store.metadata[Int(vector_id)].value()
        else:
            item["metadata"] = Python.none()
        if self_ptr[].store.source_spans[Int(vector_id)]:
            item["source"] = (
                self_ptr[].store.source_spans[Int(vector_id)].value()
            )
        else:
            item["source"] = Python.none()
        py_results.append(item)
    return py_results


def _sparse_from_python(sparse_obj: PythonObject) raises -> SparseVector:
    """Convert Python {int: float} dict to SparseVector."""
    var result = SparseVector()
    var builtins = Python.import_module("builtins")
    var items = builtins.list(sparse_obj.items())
    for i in range(len(items)):
        var kv = items[i]
        var k = UInt32(Int(py=kv[0]))
        var v = Float32(py=kv[1])
        if v != 0.0:
            result.dims.append(k)
            result.weights.append(v)
    return result^