from typing import Any
from unittest.mock import MagicMock

from tinygrad import Tensor, nn

from tinyexplain.utils.overrides import RELUS, overwrite_backward, revert_backward


class TestModel:
    def __init__(self) -> None:
        self.l1 = nn.Linear(10, 20)
        self.l2 = nn.Linear(20, 30)
        self.l3 = nn.Linear(30, 40)
        self.l4 = nn.Linear(40, 10)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.l1(x)
        x = self.l2(x)
        x = x.relu()
        x = self.l3(x)
        x = x.relu()
        x = self.l4(x)
        return x


def test_backward_override():
    model = TestModel()
    RELUS = [model.l2, model.l3]
    overwrite_backward()
    assert "_tinyexplain_backward_override" in str(model.l1.weight.backward)


def test_backward_revert():
    Tensor._backward = MagicMock("_backward")
    Tensor.backward = lambda x: x
    revert_backward()
    assert not hasattr(Tensor, "_backward")
    assert isinstance(Tensor.backward, MagicMock)
