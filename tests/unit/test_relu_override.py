from unittest.mock import MagicMock

from tinygrad import Tensor

from tinyexplain.utils.overrides import overwrite_relu, revert_relu


def test_relu_override():
    overwrite_relu()
    assert hasattr(Tensor, "_relu")
    assert "ov" in str(Tensor.relu)


def test_relu_revert():
    Tensor._relu = MagicMock("_relu")
    Tensor.relu = lambda x: x
    revert_relu()
    assert not hasattr(Tensor, "_relu")
    assert isinstance(Tensor.relu, MagicMock)
