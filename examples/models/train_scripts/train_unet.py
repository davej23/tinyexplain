# Data from http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html

from pathlib import Path
from typing import Iterator

from tinygrad import Tensor, nn

from tinyexplain.extras.models import UNet
from tinyexplain.extras.utils import process_voc_image, process_voc_label
from tinyexplain.utils import count_model_parameters

VOC_DATASET_FOLDER = Path("VOC2012")


def create_generator(path: Path, batch_size: int = 8) -> Iterator[tuple[Tensor, Tensor]]:
    """Yield batches of input images and (xmin, ymin, xmax, ymax, clf) labels"""

    image_files = path / "JPEGImages"
    label_files = path / "SegmentationClass"

    x_batch = None
    y_batch = None

    for image_file in [n for n in image_files.iterdir() if n.suffix == ".jpg"]:
        label_file = label_files / image_file.name.replace(".jpg", ".png")
        if label_file.exists():
            image = process_voc_image(image_file)
            label = process_voc_label(label_file)
            if x_batch is not None and int(x_batch.shape[0]) % batch_size == 0:
                yield x_batch, y_batch
                x_batch = None
                y_batch = None
            else:
                x_batch = image if x_batch is None else x_batch.cat(image, dim=0)
                y_batch = label if y_batch is None else y_batch.cat(label, dim=0)


model = UNet(out_classes=20)
print("model param count: ", count_model_parameters(model))

optim = nn.optim.Adam(nn.state.get_parameters(model))

gen = create_generator(VOC_DATASET_FOLDER)
for step in range(20):
    optim.zero_grad()

    x, y = next(gen)
    x.requires_grad = True
    y.requires_grad = True

    out = model(x)
    loss = (out - y).square().mean()
    print(f"{step=} loss={loss.numpy()}")

    loss.backward()
    optim.step()

nn.state.safe_save(nn.state.get_state_dict(model), "pascal_voc_unet.bin")
