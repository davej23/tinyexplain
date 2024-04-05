import gzip
import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import requests
from tinygrad import Tensor, dtypes, nn

from tinyexplain.extras.models.yolov8 import non_max_suppression  # type: ignore[attr-defined]
from tinyexplain.utils import TinygradModel


def load_fashion_mnist_dataset() -> tuple[Tensor, Tensor]:
    images_url = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-images-idx3-ubyte.gz"
    labels_url = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-labels-idx1-ubyte.gz"

    images = np.frombuffer(gzip.decompress(requests.get(images_url, timeout=30).content), dtype=np.uint8, offset=16)
    labels = np.frombuffer(gzip.decompress(requests.get(labels_url, timeout=30).content), dtype=np.uint8, offset=8)
    images = Tensor(images).reshape((len(labels), 784)).cast(dtypes.float32)
    labels = Tensor(labels).cast(dtypes.float32)
    return images, labels


def process_voc_label(path: str | Path, fixed_size: int = 512) -> Tensor:
    """Load label image (H, W, 3) image and convert to (20, H, W) tensor"""

    label_np: npt.NDArray[np.float32] = plt.imread(path)[:, :, :3]
    label_np = cv2.resize(label_np, (fixed_size, fixed_size))[:, :, 0].astype(np.float32)  # pylint: disable=no-member
    label_np *= 255
    label_np[label_np > 20.0] = 0.0
    label = Tensor.stack([Tensor(label_np, dtype=dtypes.int32) == i for i in range(20)]).cast(dtypes.float32)
    return label.unsqueeze(0)


def process_voc_image(path: str | Path, fixed_size: int = 512) -> Tensor:
    """Load image as tensor"""

    image_np = plt.imread(path)
    image_np = cv2.resize(image_np, (fixed_size, fixed_size))  # pylint: disable=no-member
    image = Tensor(image_np, dtype=dtypes.float32).unsqueeze(0).permute((0, 3, 1, 2)) / 255.0
    return image


def load_mock_sine_dataset(sequence_length: int = 16) -> tuple[Tensor, Tensor]:
    data = Tensor(list(range(100000))).sin()
    rand_idx = int(Tensor.randint(1, high=data.shape[0]).numpy())
    return data[rand_idx : rand_idx + sequence_length].reshape((1, 1, sequence_length)), data[
        rand_idx + 1 : rand_idx + sequence_length + 1
    ].reshape((1, 1, sequence_length))


def yolov8_nms(x: Tensor) -> Tensor:  # This is very slow
    """This function carries out NMS on model outputs to get one bounding box"""
    _, idx = non_max_suppression(x.numpy(), max_det=1)
    x_act = Tensor(x[:, :, idx[0].item()].unsqueeze(0).numpy())
    objectness = x_act[:, :, 4:].max(2)
    x_out = x_act[:, :, :3]
    x_out.cat(objectness.reshape((1, 1)), dim=2)
    x_out.cat(x_act[:, :, 4:], dim=2)
    return x_out


def yolov8_tbo(x: Tensor) -> Tensor:  # This is much faster
    """This function selects top box by 'objectness' (box with highest probability)"""

    top_idxs = x[:, 4:, :].max(1).argmax(1)  # (BS,)  # need to create new tensor here or breaks thing

    if x.requires_grad:
        top_idxs = Tensor(top_idxs.numpy())

    top_x = Tensor.stack([x[i, :, top_idxs[i]].unsqueeze(1) for i in range(x.shape[0])])  # (BS, 4 + NC, 1)
    x_fast = top_x[:, :4, :]  # get coords
    x_fast = x_fast.cat(top_x[:, 4:, :].max(1).unsqueeze(-1), dim=1)  # add max prob as 'objectness'
    x_fast = x_fast.cat(top_x[:, 4:, :], dim=1)  # add probs
    # x_fast = x_fast.permute((0, 2, 1))  # permute to (B, N, 5 + NC)

    return x_fast


def save_torch_state_dict(model: torch.nn.Module, filename: str) -> None:
    """Save a PyTorch model state dict to file"""

    def custom_tensor_serialiser(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()

        return obj.__dict__

    with open(filename, "w") as state_dict:
        json.dump(model.state_dict(), state_dict, default=custom_tensor_serialiser)


def load_state_dict_from_torch(model: TinygradModel, state_dict_filename: str) -> TinygradModel:
    """Load PyTorch model state dict into Tinygrad model"""

    with open(state_dict_filename, "r") as state_dict:
        torch_state_dict = json.load(state_dict)

    def convert_to_tinygrad_state_dict(x: dict[str, Any]):
        for k, v in x.items():
            if isinstance(v, list):
                x[k] = Tensor(v)
            if isinstance(v, dict):
                x[k] = state_dict_to_tensor(v)
        return x

    tinygrad_state_dict = convert_to_tinygrad_state_dict(torch_state_dict)
    nn.state.load_state_dict(model, tinygrad_state_dict)
    return model
