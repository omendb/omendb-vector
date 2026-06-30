"""
OmenDB-Mojo Quickstart — end-to-end usage example.

Demonstrates:
  1. Creating an in-memory vector store
  2. Inserting vectors with metadata
  3. Searching by vector similarity
  4. Text search (BM25)
  5. Persistence (create persistent store, flush, reopen)
  6. Memory budget inspection
"""
from store import VectorStore, VectorStoreOptions, SearchOptions
from store_types import Metric, HNSWParams


def main() raises:
    print("=== OmenDB-Mojo Quickstart ===\n")

    # --- 1. Create an in-memory store ---
    var store = VectorStore[128]()
    print("Created in-memory store with dim=128")

    # --- 2. Insert vectors (without text) ---
    var ids = List[String]()
    ids.append("doc:alpha")
    ids.append("doc:beta")
    ids.append("doc:gamma")

    for i in range(3):
        var vec = List[Float32]()
        for d in range(128):
            vec.append(Float32(i * 100 + d))
        var vid = store.set(ids[i], vec)
        print("Inserted " + ids[i] + " -> vector_id " + String(vid))

    print("Store count: " + String(store.count()))

    # --- 3. Search by vector similarity ---
    var query = List[Float32]()
    for d in range(128):
        query.append(Float32(d))

    var results = store.search(query, SearchOptions(k=3))
    print("\nVector search results (k=3):")
    for i in range(len(results)):
        var r = results[i].copy()
        print("  " + r.id + "  distance=" + String(r.distance))

    # --- 4. Text search (BM25) ---
    # Use a separate store for text demo since set_text requires unique IDs
    var text_store = VectorStore[128]()
    text_store.enable_text_search()

    var alpha_vec = List[Float32]()
    var beta_vec = List[Float32]()
    var gamma_vec = List[Float32]()
    for d in range(128):
        alpha_vec.append(Float32(d))
        beta_vec.append(Float32(100 + d))
        gamma_vec.append(Float32(200 + d))

    _ = text_store.set_text(
        "doc:alpha", alpha_vec, "vector database for embeddings"
    )
    _ = text_store.set_text(
        "doc:beta", beta_vec, "text search with BM25 ranking"
    )
    _ = text_store.set_text(
        "doc:gamma", gamma_vec, "hybrid search combining vectors and text"
    )

    var text_results = text_store.search_text("BM25 ranking", k=2)
    print("\nText search 'BM25 ranking' (k=2):")
    for i in range(len(text_results)):
        var r = text_results[i].copy()
        print("  " + r.id + "  distance=" + String(r.distance))

    # --- 5. Persistence ---
    var path = "/tmp/omendb_quickstart_test"
    var persistent_store = VectorStore[128].create(path)
    var persist_ids = List[String]()
    persist_ids.append("doc:alpha")
    persist_ids.append("doc:beta")
    persist_ids.append("doc:gamma")
    for i in range(3):
        var vec = List[Float32]()
        for d in range(128):
            vec.append(Float32(i * 100 + d))
        _ = persistent_store.set(persist_ids[i], vec)
    persistent_store.flush()
    print("\nFlushed to " + path)

    var reopened = VectorStore[128].open(path)
    print("Reopened store, count=" + String(reopened.count()))

    var reopen_results = reopened.search(query, SearchOptions(k=2))
    print("Search after reopen:")
    for i in range(len(reopen_results)):
        var r = reopen_results[i].copy()
        print("  " + r.id + "  distance=" + String(r.distance))

    # --- 6. Memory budget ---
    print("\nMemory usage:\n" + reopened.memory_usage())

    reopened.close()
    persistent_store.close()
    text_store.close()
    store.close()
    print("\n=== Quickstart complete ===")
