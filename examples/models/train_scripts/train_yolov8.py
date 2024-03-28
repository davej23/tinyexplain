# Data from http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html

import xml.dom.minidom
from pathlib import Path
from typing import Iterator

import cv2
import matplotlib.pyplot as plt
import xmltodict
from tinygrad import Tensor, nn

from tinyexplain.extras.models import YOLOv8
from tinyexplain.utils import count_model_parameters

VOC_DATASET_FOLDER = Path("VOC2012")
LABELS = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


def create_generator(path: Path, batch_size: int = 32) -> Iterator[tuple[Tensor, Tensor]]:
    """Yield batches of input images and (xmin, ymin, xmax, ymax, clf) labels"""

    image_files = path / "JPEGImages"
    label_files = path / "Annotations"

    x_batch = None
    y_batch = None

    for image_file in [n for n in image_files.iterdir() if n.suffix == ".jpg"]:
        label_file = label_files / image_file.name.replace(".jpg", ".xml")
        if label_file.exists():
            image = plt.imread(image_file)
            image = cv2.resize(image, (512, 512))
            image = Tensor(image).unsqueeze(0).permute((0, 3, 1, 2))

            label_data = xml.dom.minidom.parse(str(label_file))
            label_data = xmltodict.parse(label_data.toxml("utf-8"))["annotation"]
            image_size = (int(label_data["size"]["height"]), int(label_data["size"]["width"]))

            obj = label_data["object"] if isinstance(label_data["object"], dict) else label_data["object"][0]
            clf = LABELS.index(obj["name"])
            coords = [int(n) for n in list(obj["bndbox"].values())]
            coords[0] /= image_size[1]
            coords[1] /= image_size[0]
            coords[2] /= image_size[1]
            coords[3] /= image_size[0]
            label = Tensor([coords + [clf]]).unsqueeze(0)

            if x_batch is not None and int(x_batch.shape[0]) % batch_size == 0:
                yield x_batch, y_batch
                x_batch = None
                y_batch = None
            else:
                x_batch = image if x_batch is None else x_batch.cat(image, dim=0)
                y_batch = label if y_batch is None else y_batch.cat(label, dim=0)


model = YOLOv8(1, 1, 1, len(LABELS))
print("model param count: ", count_model_parameters(model))

optim = nn.optim.Adam(nn.state.get_parameters(model))

gen = create_generator(VOC_DATASET_FOLDER)
for step in range(10):
    optim.zero_grad()

    x, y = next(gen)
    x.requires_grad = True
    y.requires_grad = True

    out = model(x)
    loss = (out - y).square().mean()  # TODO replace with YOLOv8 loss
    print(f"{step=} loss={loss.numpy()}")

    loss.backward()
    optim.step()

nn.state.safe_save(nn.state.get_state_dict(model), "pascal_voc_yolov8.bin")
