import pytest
import tensorflow as tf
import numpy as np
from numpy.testing import assert_array_equal

from erasing.layer import ErasingLayer


@pytest.fixture()
def img():
    """Return a small test image with one channel"""
    return tf.constant(
        [
            [1, 2, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 2],
        ],
        dtype=tf.uint8,
    )


@pytest.mark.parametrize(
    ["x_loc", "y_loc", "height", "width", "expected"],
    [
        (0, 0, 1, 1, [[0, 2, 1, 1], [1, 1, 1, 1], [1, 1, 1, 2]]),
        (1, 0, 2, 2, [[1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 2]]),
        (1, 1, 2, 2, [[1, 2, 1, 1], [1, 0, 0, 1], [1, 0, 0, 2]]),
        (0, 1, 1, 4, [[1, 2, 1, 1], [0, 0, 0, 0], [1, 1, 1, 2]]),
        (3, 0, 3, 1, [[1, 2, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0]]),
    ],
)
def test_erase_target(img, x_loc, y_loc, height, width, expected):
    layer = ErasingLayer()
    result = layer.erase_target(
        img, x_loc=x_loc, y_loc=y_loc, target_height=height, target_width=width
    )
    expected = np.array(expected)
    assert_array_equal(result.numpy(), expected)
