import numpy as np

from tinygrad import Tensor

from tinyexplain.score_functions import xcycwh_to_xyxy, relative_to_absolute, iou_score, drise_score


def test_xcycwh_to_xyxy():
    x = Tensor([0.1, 0.2, 0.05, 0.1]).reshape((1, 4, 1))
    x_expected = Tensor([0.075, 0.15, 0.125, 0.25]).reshape((1, 4, 1))
    x_converted = xcycwh_to_xyxy(x)
    np.testing.assert_allclose(x_converted.numpy(), x_expected.numpy(), 1e-6)


def test_relative_to_absolute():
    x = Tensor([0.1, 0.2, 0.05, 0.1]).reshape((1, 4, 1))
    x_expected = Tensor([10.0, 20.0, 5.0, 10.0]).reshape((1, 4, 1))
    x_converted = relative_to_absolute(x, 100, 100)
    np.testing.assert_allclose(x_converted.numpy(), x_expected.numpy(), 1e-6)


def test_iou_score():
    x1 = Tensor([100.0, 100.0, 120.0, 120.0]).reshape((4, 1))
    x2 = Tensor([110.0, 100.0, 120.0, 120.0]).reshape((4, 1))
    x_expected = Tensor(0.5)
    x_converted = iou_score(x1, x2)
    np.testing.assert_allclose(x_converted.numpy(), x_expected.numpy(), 1e-6)


def test_drise_score():
    x_gt = Tensor([100.0, 100.0, 120.0, 120.0, 1.0, 0.0, 0.0]).reshape((1, 7, 1))
    x_pred = Tensor([110.0, 100.0, 120.0, 120.0, 0.2, 0.6, 0.2]).reshape((1, 7, 1))
    np.testing.assert_allclose(drise_score(x_pred, x_gt, 100, 100, skip_xcycwh_to_xyxy=True).numpy(), 0.09045341)
