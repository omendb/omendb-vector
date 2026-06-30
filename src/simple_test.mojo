from std.memory import UnsafePointer
from distance import l2_distance
from flat_index import FlatIndex
from hnsw import Candidate


def main() raises:
    print("Testing basic FlatIndex and Distance...")
    var index = FlatIndex[4]()
    var v1: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var v2: List[Float32] = [4.0, 3.0, 2.0, 1.0]

    var s1 = Span(ptr=v1.unsafe_ptr(), length=4)
    var s2 = Span(ptr=v2.unsafe_ptr(), length=4)

    print("Inserting...")
    index.insert(s1)
    index.insert(s2)

    print("Searching...")
    var results = index.search(s1, 2)
    print("Results:", len(results))
    for i in range(len(results)):
        print("ID:", results[i].id, "Dist:", results[i].distance)
