"""Crash recovery and data integrity tests."""
import tempfile
import os
import random
import omendb_vector


def _random_vec(dim=128):
    return [random.random() for _ in range(dim)]


class TestFlushReopen:
    """Test that flush/reopen preserves all data and state."""

    def test_vectors_survive_reopen(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "db")
            db = omendb_vector.open(path)
            c = db.collection("test", config=omendb_vector.CollectionConfig(dim=128))
            vecs = {}
            for i in range(100):
                v = _random_vec()
                vecs[f"d-{i}"] = v
                c.set(f"d-{i}", vector=v, metadata={"i": i})
            c.flush()
            db.close()

            db2 = omendb_vector.open(path)
            c2 = db2.collection("test")
            for i in range(100):
                item = c2.get(f"d-{i}", include_vector=True)
                assert item is not None, f"d-{i} missing after reopen"
                assert item["metadata"]["i"] == i
            db2.close()

    def test_deleted_items_stay_deleted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "db")
            db = omendb_vector.open(path)
            c = db.collection("test", config=omendb_vector.CollectionConfig(dim=128))
            c.set("a", vector=_random_vec())
            c.set("b", vector=_random_vec())
            c.delete("a")
            c.flush()
            db.close()

            db2 = omendb_vector.open(path)
            c2 = db2.collection("test")
            assert c2.get("a") is None
            assert c2.get("b") is not None
            db2.close()

    def test_superseded_items_stay_superseded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "db")
            db = omendb_vector.open(path)
            c = db.collection("test", config=omendb_vector.CollectionConfig(dim=128))
            c.set("old", vector=_random_vec(), metadata={"v": 1})
            c.set("new", vector=_random_vec(), metadata={"v": 2})
            c.supersede("old", "new")
            c.flush()
            db.close()

            db2 = omendb_vector.open(path)
            c2 = db2.collection("test")
            assert c2.get("old") is None
            item = c2.get("new", include_vector=True)
            assert item is not None
            assert item["metadata"]["v"] == 2
            db2.close()

    def test_relationships_survive_reopen(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "db")
            db = omendb_vector.open(path)
            c = db.collection(
                "test", config=omendb_vector.CollectionConfig(dim=128, graph=True)
            )
            c.set("a", vector=_random_vec())
            c.set("b", vector=_random_vec())
            c.set("c", vector=_random_vec())
            c.add_relationship("a", "b", type="related")
            c.add_relationship("a", "c", type="cites")
            c.flush()
            db.close()

            db2 = omendb_vector.open(path)
            c2 = db2.collection("test")
            nbs = c2.neighbors("a")
            assert len(nbs) == 2
            db2.close()

    def test_timestamps_survive_reopen(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "db")
            db = omendb_vector.open(path)
            c = db.collection("test", config=omendb_vector.CollectionConfig(dim=128))
            c.set("a", vector=_random_vec())
            c.flush()
            db.close()

            db2 = omendb_vector.open(path)
            c2 = db2.collection("test")
            item = c2.get("a", include_timestamps=True)
            assert item is not None
            assert item["created_at"] is not None
            db2.close()

    def test_search_works_after_reopen(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "db")
            db = omendb_vector.open(path)
            c = db.collection("test", config=omendb_vector.CollectionConfig(dim=128))
            vecs = []
            for i in range(100):
                v = _random_vec()
                vecs.append(v)
                c.set(f"d-{i}", vector=v, metadata={"group": i % 5})
            c.flush()
            db.close()

            db2 = omendb_vector.open(path)
            c2 = db2.collection("test")
            results = c2.search_vector(vecs[0], k=5)
            assert len(results) == 5
            assert results[0].id == "d-0"  # exact match
            db2.close()

    def test_filtered_search_works_after_reopen(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "db")
            db = omendb_vector.open(path)
            c = db.collection("test", config=omendb_vector.CollectionConfig(dim=128))
            vecs = []
            for i in range(100):
                v = _random_vec()
                vecs.append(v)
                c.set(f"d-{i}", vector=v, metadata={"group": i % 5})
            c.flush()
            db.close()

            db2 = omendb_vector.open(path)
            c2 = db2.collection("test")
            results = c2.search_vector(vecs[0], k=10, filter={"group": 0})
            assert len(results) == 10
            for r in results:
                assert r.metadata["group"] == 0
            db2.close()

    def test_check_passes_after_reopen(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "db")
            db = omendb_vector.open(path)
            c = db.collection("test", config=omendb_vector.CollectionConfig(dim=128))
            for i in range(50):
                c.set(f"d-{i}", vector=_random_vec(), metadata={"i": i})
            c.delete("d-10")
            c.flush()
            db.close()

            db2 = omendb_vector.open(path)
            check = db2.check()
            assert check.ok, f"check failed: {check.issues}"
            db2.close()

    def test_cosine_metric_survives_reopen(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "db")
            db = omendb_vector.open(path)
            c = db.collection(
                "test", config=omendb_vector.CollectionConfig(dim=128, metric="cosine")
            )
            v = _random_vec()
            c.set("a", vector=v)
            c.flush()
            db.close()

            db2 = omendb_vector.open(path)
            c2 = db2.collection("test")
            assert c2.config.metric == "cosine"
            item = c2.get("a", include_vector=True)
            assert item is not None
            db2.close()
