from std.testing import assert_equal
from graph import PropertyGraph, PropertyValue


def test_sqlite_persistence() raises:
    var g = PropertyGraph()
    var db_path = "test_graph.db"

    # Setup store
    g.set_store(db_path)

    # Add node with properties
    var id = g.add_node("Agent")
    g.nodes[Int(id)].properties["name"] = PropertyValue(String("Omen"))
    g.nodes[Int(id)].properties["version"] = PropertyValue(Int(1))

    # Force re-save with properties
    if g.store:
        g.store.value().save_node(g.nodes[Int(id)])

    # Create new graph and load
    var g2 = PropertyGraph()
    g2.set_store(db_path)

    if g2.store:
        var loaded_nodes = g2.store.value().load_nodes()
        assert_equal(len(loaded_nodes), 1)
        assert_equal(loaded_nodes[0].label, "Agent")
        assert_equal(
            loaded_nodes[0].properties["name"].unsafe_get[String](), "Omen"
        )
        assert_equal(loaded_nodes[0].properties["version"].unsafe_get[Int](), 1)

    from std.os import remove

    try:
        remove(db_path)
    except:
        pass


def main() raises:
    test_sqlite_persistence()
    print("PASS test_sqlite_persistence")
