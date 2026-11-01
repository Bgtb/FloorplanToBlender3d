import numpy as np
import sys

try:
    sys.path.insert(0, sys.path[0] + "/..")
    from FloorplanToBlenderLib import *  # floorplan to blender lib
except ImportError:
    raise ImportError  # floorplan to blender lib

box = np.array([[[1, 1]], [[200, 300]], [[50, 400]], [[100, 10]]], dtype=np.int32)


def test_rescale_rect():
    assert transform.rescale_rect([box], 0.5)


def test_flatten():
    assert ["j", "f", "q"] == transform.flatten(["j", ["f"], [["q"]]])


def test_rotate_round_origin_vector_2d():
    assert transform.rotate_round_origin_vector_2d((0, 1), (2, 2), 90)


def test_scale_model_point_to_origin():
    assert transform.scale_model_point_to_origin((0, 1), (2, 4), 0.2, 0.3)


def test_verts_to_poslist():
    assert transform.verts_to_poslist(
        [[[[[7.26, 5.37, 0], [7.26, 5.37, 1], [7.26, 5.71, 0], [7.26, 5.71, 1]]]]]
    )


def test_scale_point_to_vector():
    assert transform.scale_point_to_vector([[(2, 3)], [(1, 0)]])


def test_create_4xn_verts_and_faces():
    assert transform.create_4xn_verts_and_faces([[[(2, 3)]], [[(1, 0)]]])


def test_create_nx4_verts_and_faces():
    assert transform.create_nx4_verts_and_faces([[[(2, 3)]], [[(1, 0)]]])


def test_create_verts():
    assert transform.create_verts([[[(2, 3)]], [[(1, 0)]]], 1, 2)


def test_flatten_iterative_safe_preserves_order_types_and_input():
    import copy

    values = [[], [1, [2.5, [], -3]], True, "skip", None, (8, 9), [4]]
    original = copy.deepcopy(values)
    seed = [99]
    result = transform.flatten_iterative_safe(values, seed)
    assert result is seed
    assert result == [99, 1, 2.5, -3, True, 4]
    assert [type(value) for value in result] == [int, int, float, int, bool, int]
    assert values == original
    for value in (None, 7, (1, 2), "skip"):
        seed = [99]
        assert transform.flatten_iterative_safe(value, seed) is seed
        assert seed == [99]


def test_flatten_iterative_safe_matches_recursive_reference():
    import random

    def reference(values, result):
        if not isinstance(values, list):
            return result
        for value in values:
            if isinstance(value, list):
                reference(value, result)
            elif isinstance(value, int) or isinstance(value, float):
                result.append(value)
        return result

    rng = random.Random(45)

    def nested(depth):
        leaves = [None, "skip", (1, 2), False, -7, 2.5, 0]
        return [
            nested(depth - 1) if depth and rng.randrange(3) == 0
            else rng.choice(leaves)
            for _ in range(rng.randrange(8))
        ]

    for _ in range(100):
        values = nested(5)
        assert transform.flatten_iterative_safe(values, []) == reference(values, [])


def test_verts_to_poslist_exact_order_and_empty_lists():
    values = [[], [[[1, 2, 3], [], [4.5, 5, 6]]], [[7, 8, 9]]]
    assert transform.verts_to_poslist(values) == [
        [1, 2, 3], [4.5, 5, 6], [7, 8, 9]
    ]
    assert transform.verts_to_poslist([]) == []


def test_flatten_iterative_safe_wide_and_deep():
    import sys

    count = sys.getrecursionlimit() + 100
    wide = [[1, 2, 3] for _ in range(count)]
    assert transform.flatten_iterative_safe(wide, []) == [1, 2, 3] * count
    assert transform.verts_to_poslist(wide) == wide
    deep = [1, 2, 3]
    for _ in range(count):
        deep = [deep]
    assert transform.flatten_iterative_safe(deep, []) == [1, 2, 3]
    assert transform.verts_to_poslist(deep) == [[1, 2, 3]]
