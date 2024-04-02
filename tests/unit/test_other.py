import pytest

from tinygrad import Tensor, nn

from tinyexplain.utils.overrides import get_layer, is_tinygrad_layer, TINYGRAD_LAYERS, \
                                        get_model_layer_names, get_model_layers


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


def test_get_layer():
    m = TestModel()
    assert get_layer(m, "l3") is not None
    assert get_layer(m, "l3").weight.shape == (40, 30)
    assert get_layer(m, "l3").bias.shape == (40,)


def test_get_layer_incorrect():
    m = TestModel()
    with pytest.raises(AttributeError):
        assert get_layer(m, "l5")


def test_is_tinygrad_layer():
    for n in TINYGRAD_LAYERS:
        assert is_tinygrad_layer(n)


def test_is_tinygrad_layer_incorrect():
    assert is_tinygrad_layer("MyMockLayer") == False


def test_get_model_layer_names():
    m = TestModel()
    assert get_model_layer_names(m, nn.Linear) == ["l1", "l2", "l3", "l4"]


def test_get_model_layer_names_incorrect():
    m = TestModel()
    assert get_model_layer_names(m, nn.Conv2d) == []


def test_get_model_layers():
    m = TestModel()
    layers = get_model_layers(m, nn.Linear)
    assert len(layers) == 4
    assert list(layers.keys()) == ["l1", "l2", "l3", "l4"]
