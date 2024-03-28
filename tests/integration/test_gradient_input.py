from tinygrad import Tensor, dtypes

from tinyexplain.explain import GradientInput
from tinyexplain.extras.models import FashionMnistCnnClassifier, TabularDataModel, TimeSeriesModel, YOLOv8
from tinyexplain.extras.utils import yolov8_tbo
from tinyexplain.types import TinyExplainTask


def test_object_detection():
    model = YOLOv8(1, 1, 1, 1)
    explainer = GradientInput(model, TinyExplainTask.OBJECT_DETECTION)
    explanation = explainer.explain(
        Tensor.rand((1, 3, 640, 640), dtype=dtypes.float32), Tensor.rand((1, 5, 1), dtype=dtypes.float32), yolov8_tbo
    )
    assert explanation.shape == (1, 640, 640, 3)


def test_image_classification():
    model = FashionMnistCnnClassifier()
    explainer = GradientInput(model, TinyExplainTask.IMAGE_CLASSIFICATION)
    explanation = explainer.explain(Tensor.rand((1, 3, 28, 28)), Tensor.rand((1, 1)), lambda x: x)
    assert explanation.shape == (1, 28, 28, 3)


def test_tabular_data():
    model = TabularDataModel()
    explainer = GradientInput(model, TinyExplainTask.TABULAR_DATA)
    explanation = explainer.explain(Tensor.rand((1, 1, 4), dtype=dtypes.float32), Tensor.rand((1, 1, 1), dtype=dtypes.float32), lambda x: x)
    assert explanation.shape == (1, 1, 4)


def test_time_series():
    model = TimeSeriesModel()
    explainer = GradientInput(model, TinyExplainTask.TIME_SERIES)
    explanation = explainer.explain(
        Tensor.rand((1, 1, 16), dtype=dtypes.float32), Tensor.rand((1, 1, 16), dtype=dtypes.float32), lambda x: x
    )
    assert explanation.shape == (1, 1, 16)
